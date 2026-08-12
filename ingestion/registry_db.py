import time
from pathlib import Path

from core.db import connect, write
from core.models import Document, IngestStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    started_at  REAL,
    finished_at REAL,
    added_at    REAL NOT NULL DEFAULT (julianday('now'))
);
"""

# Columns added after the first release; applied to pre-existing databases
# on open so upgrading in place never needs a manual migration.
_MIGRATIONS = [
    ("bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("started_at", "REAL"),
    ("finished_at", "REAL"),
]


def estimate_eta(done: list[tuple[int, float]], remaining_bytes: int,
                 remaining_count: int, current_elapsed: float) -> float | None:
    """Seconds until the queue drains, or None if not yet estimable.

    `done` is (bytes, processing_seconds) for each finished document.

    Prefer a bytes/second throughput (robust when the remaining documents
    differ wildly in size), but fall back to a per-document average while
    byte sizes are unknown - older queue entries predate size tracking.
    `current_elapsed` discounts time already spent on the in-progress
    document so the estimate counts down instead of stalling.
    """
    if remaining_count <= 0:
        return None

    byte_done = [(b, d) for b, d in done if b and b > 0 and d > 0]
    eta: float | None = None

    if remaining_bytes > 0 and byte_done:
        total_bytes = sum(b for b, _ in byte_done)
        total_secs = sum(d for _, d in byte_done)
        eta = remaining_bytes / (total_bytes / total_secs)
    else:
        timed = [d for _, d in done if d > 0]
        if timed:
            eta = (sum(timed) / len(timed)) * remaining_count

    if eta is None:
        return None
    return max(eta - current_elapsed, 0.0)


class Registry:
    """SQLite document registry.

    Also the sole communication channel between the ingestion worker thread
    and the Streamlit UI, so every method must be thread-safe.
    """

    def __init__(self, path: Path):
        self._conn, self._lock = connect(path, SCHEMA)
        with self._lock:
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        existing = {r["name"] for r in
                    self._conn.execute("PRAGMA table_info(documents)")}
        for name, ddl in _MIGRATIONS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE documents ADD COLUMN {name} {ddl}"
                )

    def _write(self, sql: str, params: tuple) -> None:
        write(self._conn, self._lock, sql, params)

    def _row_to_doc(self, row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            status=IngestStatus(row["status"]),
            error=row["error"],
            chunk_count=row["chunk_count"],
        )

    def add(self, doc_id: str, filename: str, size_bytes: int = 0) -> None:
        self._write(
            "INSERT OR IGNORE INTO documents (doc_id, filename, status, bytes) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, filename, IngestStatus.QUEUED.value, size_bytes),
        )

    def get(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def all(self) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY added_at"
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def next_queued(self) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE status = ? "
                "ORDER BY added_at LIMIT 1",
                (IngestStatus.QUEUED.value,),
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def mark_processing(self, doc_id: str) -> None:
        # Stamp the start and clear any prior finish time so a re-run (e.g. a
        # retry) is timed from scratch rather than showing a stale duration.
        self._write(
            "UPDATE documents SET status = ?, started_at = ?, "
            "finished_at = NULL WHERE doc_id = ?",
            (IngestStatus.PROCESSING.value, time.time(), doc_id),
        )

    def mark_done(self, doc_id: str, chunk_count: int) -> None:
        self._write(
            "UPDATE documents SET status = ?, error = NULL, chunk_count = ?, "
            "finished_at = ? WHERE doc_id = ?",
            (IngestStatus.DONE.value, chunk_count, time.time(), doc_id),
        )

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._write(
            "UPDATE documents SET status = ?, error = ?, finished_at = ? "
            "WHERE doc_id = ?",
            (IngestStatus.FAILED.value, error, time.time(), doc_id),
        )

    def requeue(self, doc_id: str) -> None:
        """Put an existing document back in the queue (e.g. to re-chunk it)."""
        self._write(
            "UPDATE documents SET status = ?, error = NULL WHERE doc_id = ?",
            (IngestStatus.QUEUED.value, doc_id),
        )

    def remove(self, doc_id: str) -> None:
        self._write("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def reset_stale_processing(self) -> int:
        """Recover documents wedged by a crash mid-ingest."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE documents SET status = ? WHERE status = ?",
                (IngestStatus.QUEUED.value, IngestStatus.PROCESSING.value),
            )
            self._conn.commit()
            return cursor.rowcount

    def ingest_eta(self) -> tuple[int, int, float | None]:
        """(processing count, queued count, estimated seconds remaining).

        The estimate is None until at least one document has finished with a
        measured duration this run; see estimate_eta for the method.
        """
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, bytes, started_at, finished_at FROM documents"
            ).fetchall()

        processing = sum(1 for r in rows
                         if r["status"] == IngestStatus.PROCESSING.value)
        queued = sum(1 for r in rows
                     if r["status"] == IngestStatus.QUEUED.value)

        done = [
            (r["bytes"] or 0, r["finished_at"] - r["started_at"])
            for r in rows
            if r["started_at"] is not None and r["finished_at"] is not None
            and r["finished_at"] > r["started_at"]
        ]
        remaining_bytes = sum(
            r["bytes"] or 0 for r in rows
            if r["status"] in (IngestStatus.QUEUED.value,
                               IngestStatus.PROCESSING.value)
        )
        in_flight = [r["started_at"] for r in rows
                     if r["status"] == IngestStatus.PROCESSING.value
                     and r["started_at"] is not None]
        current_elapsed = (now - min(in_flight)) if in_flight else 0.0

        eta = estimate_eta(done, remaining_bytes, processing + queued,
                           current_elapsed)
        return processing, queued, eta
