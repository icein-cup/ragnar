"""Shared SQLite wiring for the two on-disk stores.

Registry (ingestion) and ChatStore (history) are both single-connection,
lock-guarded SQLite tables read from the Streamlit UI thread and written from
the ingestion worker thread. This is the setup they had written out twice.
"""
import sqlite3
import threading
from pathlib import Path


def connect(path: Path,
            schema: str) -> tuple[sqlite3.Connection, threading.Lock]:
    """Open a lock-guarded, row-dict SQLite connection and apply `schema`.

    check_same_thread=False because the connection is shared across the UI
    and worker threads; the returned lock is what actually serializes access,
    so every caller must hold it around both reads and writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    lock = threading.Lock()
    with lock:
        conn.executescript(schema)
        conn.commit()
    return conn, lock


def write(conn: sqlite3.Connection, lock: threading.Lock,
          sql: str, params: tuple) -> None:
    """Run a single statement and commit, under the connection's lock."""
    with lock:
        conn.execute(sql, params)
        conn.commit()
