import json
import sqlite3
import threading
import time
from pathlib import Path

from core.models import SavedChat

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id    TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    messages   TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

TITLE_MAX = 40


def chat_title(messages: list[dict]) -> str:
    """A short title from the first user message, or a placeholder.

    Whitespace is collapsed and the text is truncated so long questions
    don't overflow the sidebar.
    """
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(message.get("content", "").split())
            if text:
                return text[:TITLE_MAX] + "…" if len(text) > TITLE_MAX else text
    return "New chat"


def dataclass_to_dict(obj):
    """Convert dataclass instances to plain dicts so they survive json.dumps.

    Doubles as json.dumps' `default=` hook, hence the TypeError on anything
    that isn't a dataclass.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ChatStore:
    """SQLite persistence for conversation history.

    Separate from the ingestion registry so chat history and document state
    stay decoupled. Only the UI thread touches it, but it mirrors Registry's
    lock-guarded, single-connection pattern for consistency and safety.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _row_to_chat(self, row) -> SavedChat:
        return SavedChat(
            chat_id=row["chat_id"],
            title=row["title"],
            messages=json.loads(row["messages"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save(self, chat_id: str, title: str, messages: list[dict]) -> None:
        """Insert a new chat or update an existing one in place.

        created_at is preserved across updates (ON CONFLICT keeps it); only
        the title, messages, and updated_at move.
        """
        now = time.time()
        payload = json.dumps(messages, ensure_ascii=False,
                             default=dataclass_to_dict)
        with self._lock:
            self._conn.execute(
                "INSERT INTO chats "
                "(chat_id, title, messages, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET "
                "title = excluded.title, messages = excluded.messages, "
                "updated_at = excluded.updated_at",
                (chat_id, title, payload, now, now),
            )
            self._conn.commit()

    def all(self) -> list[SavedChat]:
        """Saved chats, most recently updated first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_chat(r) for r in rows]

    def delete(self, chat_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM chats WHERE chat_id = ?", (chat_id,)
            )
            self._conn.commit()
