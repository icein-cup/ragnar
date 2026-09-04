"""Ingest HybridQA anchor tables + linked passages into a dedicated Qdrant
collection for the multi-hop benchmark.

Reads ``golden_hybridqa_draft.yaml`` to learn which tables/passages are
needed (so a sampled draft only ingests the reachable subgraph, not the full
13k-table corpus), fetches the corresponding ``tables_tok`` / ``request_tok``
JSON, flattens each to chunks, embeds, and upserts into a separate collection.

Chunk ``filename`` is set to the readable source title (the table title, or a
passage's wiki title), so citations returned by ``Search`` match the
``expected_sources`` labels in the golden set.

Usage:
    python eval/ingest_hybridqa.py --golden eval/golden_hybridqa_draft.yaml
"""
import argparse
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.models import Chunk
from ingestion.chunkers.semantic import _cosine_distances, _split_prose, _split_sentences
from ingestion.tables import chunk_table_markdown
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore

TABLE_URL = (
    "https://raw.githubusercontent.com/wenhuchen/WikiTables-WithLinks/"
    "master/tables_tok/{table_id}.json"
)
REQUEST_URL = (
    "https://raw.githubusercontent.com/wenhuchen/WikiTables-WithLinks/"
    "master/request_tok/{table_id}.json"
)

# Defaults reproduce the collection every existing report was measured
# against. Deliberately NOT read from config.yaml: this is the eval corpus
# builder, not the app's StructuralChunker, and coupling them silently would
# make old reports irreproducible the next time config.yaml moves.
CHARS_PER_TOKEN = 3.5
TARGET_TOKENS = 500
OVERLAP_TOKENS = 0
ROWS_PER_GROUP = 20


def table_to_markdown(table: dict) -> str:
    header = [col[0] for col in table["header"]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in table["data"]:
        cells = [cell[0].replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_prose(text: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Fixed character-window split. Mirrors StructuralChunker._split's
    overlap semantics (ingestion/chunkers/structural.py) so the two stay
    comparable when this axis is tuned."""
    if len(text) <= max_chars:
        return [text]
    step = max(max_chars - overlap_chars, 1)
    return [text[i:i + max_chars] for i in range(0, len(text), step)]


def split_prose_semantic(passages: dict[str, str], embedder,
                         max_chars: int,
                         breakpoint_percentile: float) -> dict[str, list[str]]:
    """Sentence-boundary splitting with topic-boundary cuts.

    Mirrors SemanticChunker: sentence-split each passage, batch-embed all
    sentences in one call, compute cosine distances between consecutive
    sentences, derive a percentile threshold from the pooled distances, then
    cut where divergence exceeds it (or where the size ceiling hits).
    Pooling distances across passages keeps the percentile stable even for
    short passages with only a handful of sentences.
    """
    per_passage: dict[str, list[str]] = {}
    all_sentences: list[str] = []
    for path, text in passages.items():
        sentences = _split_sentences(text)
        per_passage[path] = sentences
        all_sentences.extend(sentences)

    vectors = embedder.embed(all_sentences) if all_sentences else []
    if len(vectors) != len(all_sentences):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for "
            f"{len(all_sentences)} sentences -- cannot align them"
        )
    vector_iter = iter(vectors)

    per_passage_distances: dict[str, list[float]] = {}
    all_distances: list[float] = []
    for path, sentences in per_passage.items():
        vecs = [next(vector_iter) for _ in sentences]
        distances = _cosine_distances(vecs) if len(vecs) > 1 else []
        per_passage_distances[path] = distances
        all_distances.extend(distances)

    threshold = (
        float(np.percentile(all_distances, breakpoint_percentile))
        if all_distances else None
    )

    return {
        path: _split_prose(sentences, per_passage_distances[path],
                           threshold, max_chars)
        for path, sentences in per_passage.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path,
                        default=Path("eval/golden_hybridqa_draft.yaml"))
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection name. Defaults to 'hybridqa', "
                             "or 'hybridqa_semantic' with --semantic.")
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic chunking for prose (topic-boundary cuts)")
    parser.add_argument("--breakpoint-percentile", type=float, default=95.0,
                        help="Cosine-distance percentile for semantic topic "
                             "boundaries (only with --semantic).")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help="Prose chunk budget in tokens (ignored for "
                             "--semantic, whose cuts land on topic boundaries "
                             "instead of a fixed size).")
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS,
                        help="Prose chunk overlap in tokens. Ignored for "
                             "--semantic, same reason as --target-tokens.")
    parser.add_argument("--rows-per-group", type=int, default=ROWS_PER_GROUP,
                        help="Table rows per chunk (header repeated in each).")
    args = parser.parse_args()
    if args.collection is None:
        args.collection = "hybridqa_semantic" if args.semantic else "hybridqa"

    max_chars = int(args.target_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(args.overlap_tokens * CHARS_PER_TOKEN)

    import yaml
    entries = yaml.safe_load(args.golden.read_text())

    cfg = Config()
    store = QdrantStore(cfg.qdrant_url, args.collection, cfg.embedding_dim)
    # Rebuild the collection fresh so chunks from a prior golden set never act
    # as stale distractors for the current entries.
    try:
        store.drop_collection()
    except Exception:
        pass
    store.ensure_collection()
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)

    # table_id -> {table, passages}; dedupe tables shared across entries.
    tables: dict[str, dict] = {}
    for entry in entries:
        # Out-of-corpus probes carry no table_id: they exist to be refused,
        # so ingesting anything for them would defeat the point.
        table_id = entry.get("table_id")
        if not table_id or table_id in tables:
            continue
        with httpx.Client(follow_redirects=True) as client:
            table = client.get(TABLE_URL.format(table_id=table_id),
                               timeout=30.0)
            table.raise_for_status()
            table = table.json()
            passages = client.get(REQUEST_URL.format(table_id=table_id),
                                  timeout=30.0)
            passages.raise_for_status()
            passages = passages.json()
        tables[table_id] = {"table": table, "passages": passages}

    chunks: list[Chunk] = []
    for table_id, data in tables.items():
        table = data["table"]
        md = table_to_markdown(table)
        pieces = chunk_table_markdown(md, rows_per_group=args.rows_per_group) or [md]
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                doc_id=table_id, filename=table["title"], text=piece,
                chunk_index=i, is_table=True,
            ))
        if args.semantic:
            prose_pieces = split_prose_semantic(
                data["passages"], embedder, max_chars,
                args.breakpoint_percentile)
            for path, title in ((p, p.removeprefix("/wiki/").replace("_", " "))
                                for p in data["passages"]):
                for i, piece in enumerate(prose_pieces[path]):
                    chunks.append(Chunk(
                        doc_id=path, filename=title, text=piece,
                        chunk_index=i, is_table=False,
                    ))
        else:
            for path, text in data["passages"].items():
                title = path.removeprefix("/wiki/").replace("_", " ")
                for i, piece in enumerate(split_prose(text, max_chars, overlap_chars)):
                    chunks.append(Chunk(
                        doc_id=path, filename=title, text=piece,
                        chunk_index=i, is_table=False,
                    ))

    # Embed + upsert in batches so a single large table doesn't blow the
    # embedding request.
    batch = 32
    for start in range(0, len(chunks), batch):
        group = chunks[start:start + batch]
        vectors = embedder.embed([c.text for c in group])
        store.upsert(group, vectors)

    n_tables = len({c.doc_id for c in chunks if c.is_table})
    print(f"ingested {len(chunks)} chunks "
          f"({n_tables} tables + passages) "
          f"into collection '{args.collection}'", file=sys.stderr)


if __name__ == "__main__":
    main()