from dataclasses import dataclass
from enum import Enum


class IngestStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Chunk:
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    page: int | None = None
    sheet: str | None = None
    is_table: bool = False
    low_confidence: bool = False
    is_summary: bool = False

    def citation_label(self) -> str:
        if self.page is not None:
            return f"{self.filename}, p. {self.page}"
        if self.sheet is not None:
            return f"{self.filename}, sheet {self.sheet}"
        return self.filename


@dataclass
class Citation:
    """A rich citation with navigation metadata for the UI."""
    label: str
    doc_id: str
    filename: str
    page: int | None = None
    sheet: str | None = None
    chunk_index: int = 0


@dataclass
class Document:
    doc_id: str
    filename: str
    status: IngestStatus = IngestStatus.QUEUED
    error: str | None = None
    chunk_count: int = 0


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    # Raw vector-similarity score from the store, preserved through
    # reranking as a second signal — some content (e.g. table rows) scores
    # near-neutral on the cross-encoder despite being genuinely relevant.
    vector_score: float | None = None


@dataclass
class SavedChat:
    """A persisted conversation.

    `messages` mirrors the Streamlit chat history: a list of
    {"role", "content", "citations"} dicts, kept as plain dicts so it
    round-trips through JSON without a bespoke schema.
    """
    chat_id: str
    title: str
    messages: list[dict]
    created_at: float
    updated_at: float
