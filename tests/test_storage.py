import pytest
from ingestion.parser import Block
from ingestion.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path)


def test_directories_are_created(storage, tmp_path):
    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "originals").is_dir()
    assert (tmp_path / "converted").is_dir()


def test_doc_id_is_content_hash_not_filename(storage):
    a = storage.inbox / "a.pdf"
    b = storage.inbox / "b.pdf"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")

    assert storage.doc_id(a) == storage.doc_id(b)


def test_different_content_yields_different_doc_id(storage):
    a = storage.inbox / "a.pdf"
    b = storage.inbox / "b.pdf"
    a.write_bytes(b"one")
    b.write_bytes(b"two")

    assert storage.doc_id(a) != storage.doc_id(b)


def test_archive_moves_file_and_suffixes_with_hash(storage):
    src = storage.inbox / "report.pdf"
    src.write_bytes(b"content")
    doc_id = storage.doc_id(src)

    archived = storage.archive(src, doc_id)

    assert not src.exists()
    assert archived.exists()
    assert archived.name == f"report.{doc_id[:8]}.pdf"


def test_archiving_same_name_different_content_does_not_collide(storage):
    first = storage.inbox / "report.pdf"
    first.write_bytes(b"v1")
    a = storage.archive(first, storage.doc_id(first))

    second = storage.inbox / "report.pdf"
    second.write_bytes(b"v2")
    b = storage.archive(second, storage.doc_id(second))

    assert a != b
    assert a.exists() and b.exists()


def test_converted_artifacts_written_and_read_back(storage):
    storage.write_converted("d1", markdown="# Title")
    assert storage.read_markdown("d1") == "# Title"


def test_remove_converted_deletes_artifacts(storage):
    storage.write_converted("d1", markdown="# Title")
    storage.remove_converted("d1")
    assert storage.read_markdown("d1") is None


def test_write_parsed_round_trips_blocks(storage):
    from ingestion.parser import ParsedDocument

    parsed = ParsedDocument(
        markdown="# Title",
        blocks=[
            Block(text="hello", page=1, is_table=False, sheet=None, is_summary=False),
            Block(text="| a | b |", page=None, is_table=True, sheet="Q1", is_summary=False),
        ],
        low_confidence=True,
    )

    storage.write_parsed("d1", parsed)
    result = storage.read_parsed("d1")

    assert result.low_confidence is True
    assert [b.text for b in result.blocks] == ["hello", "| a | b |"]
    assert result.blocks[1].sheet == "Q1"
    assert result.blocks[1].is_table is True


def test_read_parsed_returns_none_when_no_cache_exists(storage):
    assert storage.read_parsed("unknown") is None


def test_remove_converted_deletes_parsed_cache_too(storage):
    from ingestion.parser import ParsedDocument

    storage.write_parsed("d1", ParsedDocument(markdown="", blocks=[], low_confidence=False))
    storage.remove_converted("d1")
    assert storage.read_parsed("d1") is None


def test_restore_to_inbox_round_trips_an_archived_original(storage):
    src = storage.inbox / "report.pdf"
    src.write_bytes(b"the original content")
    doc_id = storage.doc_id(src)
    storage.archive(src, doc_id)
    assert not (storage.inbox / "report.pdf").exists()

    ok = storage.restore_to_inbox("report.pdf", doc_id)

    assert ok is True
    restored = storage.inbox / "report.pdf"
    assert restored.exists()
    assert restored.read_bytes() == b"the original content"


def test_restore_to_inbox_returns_false_when_original_missing(storage):
    assert storage.restore_to_inbox("gone.pdf", "deadbeef" * 8) is False
