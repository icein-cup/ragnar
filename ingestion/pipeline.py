import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ingestion.parser import ParsedDocument

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    chunk_count: int
    markdown: str
    parsed: ParsedDocument


class Pipeline:
    """Orchestrates parse → chunk → embed → store for one document."""

    def __init__(self, parser, chunker, embedder, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def set_chunker(self, chunker) -> None:
        """Swap the chunker used by future ingest() calls.

        Safe to call from the UI thread while the background worker thread
        reads self._chunker in ingest() — a bare attribute reassignment is
        atomic under the GIL. Already-ingested documents are unaffected;
        this only changes how documents ingested after the call are chunked.
        """
        self._chunker = chunker

    def ingest(self, path: Path, doc_id: str,
               parsed: ParsedDocument | None = None,
               filename: str | None = None) -> IngestResult:
        """Parse (unless a cached ParsedDocument is supplied), chunk, embed, store.

        `parsed` lets the caller skip the (expensive) parse step by handing
        in a cached one — e.g. re-chunking after a settings change.

        `filename` is the document's logical name, which ends up on every
        chunk and therefore in every citation. It defaults to the path's own
        name, but staged uploads live under a content-stamped inbox name, so
        the worker passes the registry's filename explicitly — otherwise
        citations would read "report.1a2b3c4d.pdf".
        """
        t0 = time.perf_counter()
        if parsed is None:
            parsed = self._parser.parse(path)
        t1 = time.perf_counter()

        chunks = self._chunker.chunk(parsed, doc_id, filename or path.name)
        t2 = time.perf_counter()

        t3 = t4 = t2
        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            t3 = time.perf_counter()
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
                )
            self._store.upsert(chunks, vectors)
            t4 = time.perf_counter()

        # Delete stale chunks only after the new ones are safely embedded and
        # stored. A transient failure in embed()/upsert() now leaves the previous
        # index intact instead of wiping it.  We delete by chunk index, not by
        # doc_id, because deterministic point IDs mean the fresh upsert already
        # overwrote any old chunk with the same index.
        self._store.delete_stale(doc_id, {c.chunk_index for c in chunks})

        log.info(
            "ingest %s: parse=%.1fs chunk=%.1fs embed=%.1fs upsert=%.1fs chunks=%d",
            path.name, t1 - t0, t2 - t1, t3 - t2, t4 - t3, len(chunks),
        )

        return IngestResult(chunk_count=len(chunks),
                            markdown=parsed.markdown, parsed=parsed)
