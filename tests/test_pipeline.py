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
