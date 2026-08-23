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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.models import Chunk
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

# Match the app's structural chunker prose budget.
CHARS_PER_CHUNK = 500 * 3.5
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


def split_prose(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path,
                        default=Path("eval/golden_hybridqa_draft.yaml"))
    parser.add_argument("--collection", default="hybridqa")
    args = parser.parse_args()

    import yaml
    entries = yaml.safe_load(args.golden.read_text())

    cfg = Config()
    store = QdrantStore(cfg.qdrant_url, args.collection, cfg.embedding_dim)
    store.ensure_collection()
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)

    # table_id -> {table, passages}; dedupe tables shared across entries.
    tables: dict[str, dict] = {}
    for entry in entries:
        table_id = entry["table_id"]
        if table_id in tables:
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
        pieces = chunk_table_markdown(md, rows_per_group=ROWS_PER_GROUP) or [md]
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                doc_id=table_id, filename=table["title"], text=piece,
                chunk_index=i, is_table=True,
            ))
        for path, text in data["passages"].items():
            title = path.removeprefix("/wiki/").replace("_", " ")
            for i, piece in enumerate(split_prose(text, int(CHARS_PER_CHUNK))):
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