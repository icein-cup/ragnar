from core.models import Chunk, SearchResult


class FakeEmbedder:
    """Deterministic embeddings without a server."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((hash(t) >> (i * 4)) % 10) for i in range(self.dim)]
            for t in texts
        ]


class ScriptedEmbedder:
    """Maps exact text to a caller-supplied vector.

    FakeEmbedder's hash-derived vectors carry no controllable notion of
    similarity, so semantic chunker tests — which need a known similarity
    cliff between specific sentences — script the vector per text instead.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[t] for t in texts]


class FakeChunker:
    """Naive character-window chunker, for driving Pipeline/Worker tests.

    Deliberately dumber than the real chunkers — those have their own tests;
    this one just needs to turn blocks into a predictable number of chunks.
    """

    def __init__(self, target_chars: int = 2000, overlap_chars: int = 200):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(self, parsed, doc_id: str, filename: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = max(self.target_chars - self.overlap_chars, 1)
        for block in parsed.blocks:
            text = block.text.strip()
            for start in range(0, len(text), step):
                chunks.append(Chunk(
                    doc_id=doc_id, filename=filename,
                    text=text[start:start + self.target_chars],
                    chunk_index=len(chunks), page=block.page,
                    sheet=block.sheet, is_table=block.is_table,
                    low_confidence=parsed.low_confidence,
                ))
        return chunks


class FakeStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.deleted: list[str] = []

    def ensure_collection(self):
        pass

    def upsert(self, chunks, vectors):
        self.chunks.extend(chunks)

    def search(self, vector, limit, doc_ids=None):
        pool = self.chunks
        if doc_ids is not None:
            pool = [c for c in pool if c.doc_id in doc_ids]
        return [SearchResult(chunk=c, score=1.0) for c in pool[:limit]]

    def delete_by_doc(self, doc_id):
        self.deleted.append(doc_id)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]

    def delete_stale(self, doc_id, keep_indices):
        self.chunks = [
            c for c in self.chunks
            if c.doc_id != doc_id or c.chunk_index in keep_indices
        ]


class FakeParser:
    def __init__(self, blocks=None, fail=False):
        self.blocks = blocks or []
        self.fail = fail

    def parse(self, path):
        if self.fail:
            raise ValueError("unreadable file")
        from ingestion.parser import ParsedDocument
        return ParsedDocument(markdown="# doc", blocks=self.blocks,
                              page_count=1)
