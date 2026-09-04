import pytest

from ingestion.parser import Block, ParsedDocument
from ingestion.pipeline import Pipeline
from tests.fakes import FakeChunker, FakeEmbedder, FakeStore, FakeParser


def test_set_chunker_changes_chunking_for_next_ingest(tmp_path):
    store = FakeStore()
    parser = FakeParser(blocks=[Block(text="word " * 100, page=1)])
    pipeline = Pipeline(parser, FakeChunker(target_chars=1000), FakeEmbedder(), store)

    path = tmp_path / "a.pdf"
    path.write_bytes(b"data")
    first = pipeline.ingest(path, "d1")

    pipeline.set_chunker(FakeChunker(target_chars=50, overlap_chars=0))
    second = pipeline.ingest(path, "d2")

    assert second.chunk_count > first.chunk_count


def test_set_chunker_does_not_affect_already_ingested_chunks():
    store = FakeStore()
    parser = FakeParser(blocks=[Block(text="hello world", page=1)])
    pipeline = Pipeline(parser, FakeChunker(target_chars=1000), FakeEmbedder(), store)

    from pathlib import Path
    pipeline.ingest(Path("a.pdf"), "d1")
    before = list(store.chunks)

    pipeline.set_chunker(FakeChunker(target_chars=1))
    # no new ingest() call - already-stored chunks must be untouched
    assert store.chunks == before


class FailingEmbedder(FakeEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("ollama unavailable")


def test_embed_failure_leaves_previous_index_intact(tmp_path):
    store = FakeStore()
    parser = FakeParser(blocks=[Block(text="hello world", page=1)])

    # First ingest succeeds and leaves a chunk for this doc.
    pipeline = Pipeline(parser, FakeChunker(target_chars=1000), FakeEmbedder(), store)
    path = tmp_path / "a.pdf"
    path.write_bytes(b"data")
    pipeline.ingest(path, "d1")
    before = list(store.chunks)
    assert any(c.doc_id == "d1" for c in before)

    # Second ingest (re-chunk) fails during embedding.
    failing_pipeline = Pipeline(
        parser, FakeChunker(target_chars=1), FailingEmbedder(), store
    )
    with pytest.raises(RuntimeError):
        failing_pipeline.ingest(path, "d1")

    # Old chunks must still be present; the failed run must not have deleted them.
    assert store.chunks == before
    assert store.deleted == []


def test_mismatched_vector_count_raises_before_store_is_touched(tmp_path):
    store = FakeStore()
    parser = FakeParser(blocks=[Block(text="hello world one two", page=1)])

    class ShortEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return super().embed(texts)[:-1]

    pipeline = Pipeline(parser, FakeChunker(target_chars=5), ShortEmbedder(), store)
    path = tmp_path / "a.pdf"
    path.write_bytes(b"data")

    with pytest.raises(ValueError, match="embedder returned"):
        pipeline.ingest(path, "d1")

    assert store.chunks == []
    assert store.deleted == []
