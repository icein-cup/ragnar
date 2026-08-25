import re

import numpy as np

from core.models import Chunk
from ingestion.chunkers.grouping import group_blocks
from ingestion.chunkers.structural import CHARS_PER_TOKEN
from ingestion.parser import ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _cosine_distances(vectors: list[list[float]]) -> list[float]:
    """1 - cosine_similarity between each consecutive pair of vectors."""
    arr = np.array(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard a degenerate all-zero embedding
    unit = arr / norms
    sims = np.sum(unit[:-1] * unit[1:], axis=1)
    return (1.0 - sims).tolist()


def _split_prose(sentences: list[str], distances: list[float],
                  threshold: float | None, max_chars: int) -> list[str]:
    """Groups sentences into chunk texts, cutting at similarity drops.

    A cut happens where the distance to the previous sentence exceeds
    `threshold` (a topic boundary), or where adding the next sentence would
    exceed `max_chars` (the hard ceiling). A single sentence longer than
    `max_chars` is hard-split by character window — same fallback
    StructuralChunker uses for an oversized block.
    """
    pieces: list[str] = []
    current: list[str] = []

    def current_size(extra: str | None = None) -> int:
        sents = current + ([extra] if extra else [])
        if not sents:
            return 0
        return sum(len(s) for s in sents) + (len(sents) - 1)  # joining spaces

    def flush():
        if current:
            pieces.append(" ".join(current))
            current.clear()

    for i, sentence in enumerate(sentences):
        if len(sentence) > max_chars:
            flush()
            start = 0
            while start < len(sentence):
                pieces.append(sentence[start:start + max_chars])
                start += max_chars
            continue

        is_boundary = (
            i > 0 and threshold is not None and distances[i - 1] > threshold
        )
        if current and (is_boundary or current_size(sentence) > max_chars):
            flush()

        current.append(sentence)

    flush()
    return pieces


class SemanticChunker:
    """Cuts prose where consecutive sentences diverge in meaning, instead of
    at a blind character window.

    Page and table isolation are inherited from `group_blocks` — the same
    invariants StructuralChunker enforces. Only what happens *inside* a page
    of prose differs.

    No LLM is involved: the only model called is the embedding model, via
    `embedder.embed()` — the same call Pipeline already makes to embed
    chunks after chunking. "Semantic" here means vector geometry: where
    consecutive sentence embeddings point in noticeably different
    directions, that's the boundary.

    Overlap does not apply. The premise is that cuts land on topic
    boundaries; bleeding text across a boundary that was deliberately chosen
    would defeat the point.
    """

    def __init__(self, embedder, target_tokens: int = 500,
                 rows_per_group: int = 20, breakpoint_percentile: float = 95):
        self.embedder = embedder
        self.target_chars = int(target_tokens * CHARS_PER_TOKEN)
        self.rows_per_group = rows_per_group
        self.breakpoint_percentile = breakpoint_percentile

    def chunk(self, parsed: ParsedDocument, doc_id: str,
              filename: str) -> list[Chunk]:
        groups = group_blocks(parsed.blocks, max_chars=None)

        # Sentence-split every prose group up front so all embedding happens
        # in a single batched call — one HTTP round-trip per document,
        # matching how Pipeline batches the post-chunking embed pass.
        group_sentences: list[list[str] | None] = []
        all_sentences: list[str] = []
        for group in groups:
            if group[0].is_table:
                group_sentences.append(None)
                continue
            text = "\n\n".join(b.text.strip() for b in group)
            sentences = _split_sentences(text)
            group_sentences.append(sentences)
            all_sentences.extend(sentences)

        vectors = self.embedder.embed(all_sentences) if all_sentences else []
        if len(vectors) != len(all_sentences):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for "
                f"{len(all_sentences)} sentences -- cannot align them"
            )
        vector_iter = iter(vectors)

        # Distances per group, plus the document-wide pool the percentile
        # threshold is drawn from — a group with only a handful of sentences
        # shouldn't derive its own unstable cutoff.
        group_distances: list[list[float]] = []
        all_distances: list[float] = []
        for sentences in group_sentences:
            if sentences is None:
                group_distances.append([])
                continue
            vecs = [next(vector_iter) for _ in sentences]
            distances = _cosine_distances(vecs) if len(vecs) > 1 else []
            group_distances.append(distances)
            all_distances.extend(distances)

        threshold = (
            float(np.percentile(all_distances, self.breakpoint_percentile))
            if all_distances else None
        )

        chunks: list[Chunk] = []
        index = 0
        for group, sentences, distances in zip(groups, group_sentences,
                                                group_distances):
            head = group[0]

            if head.is_table:
                text = "\n\n".join(b.text.strip() for b in group)
                from ingestion.tables import chunk_table_markdown
                pieces = chunk_table_markdown(
                    text, rows_per_group=self.rows_per_group) or [text]
                if head.sheet:
                    # Same reasoning as StructuralChunker: put the sheet name
                    # into the embedded text, not just citation metadata, so
                    # "what's in the Q2 sheet?" actually retrieves.
                    pieces = [f"Sheet: {head.sheet}\n\n{p}" for p in pieces]
            else:
                pieces = _split_prose(sentences, distances, threshold,
                                       self.target_chars)

            for piece in pieces:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    text=piece,
                    chunk_index=index,
                    page=head.page,
                    sheet=head.sheet,
                    is_table=head.is_table,
                    low_confidence=parsed.low_confidence,
                    is_summary=head.is_summary,
                ))
                index += 1

        return chunks
