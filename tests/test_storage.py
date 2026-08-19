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
    # Restored under the content stamp, which is where inbox_path looks.
    restored = storage.inbox_path("report.pdf", doc_id)
    assert restored.exists()
    assert restored.read_bytes() == b"the original content"


def test_stage_keeps_same_named_uploads_apart(storage):
    """Two different documents that happen to share a filename. Staging both
    under the bare name let the second clobber the first, so the first
    doc_id was ingested from the wrong bytes."""
    first_id, first = storage.stage("report.pdf", b"v1")
    second_id, second = storage.stage("report.pdf", b"v2")

    assert first_id != second_id
    assert first != second
    assert first.read_bytes() == b"v1"
    assert second.read_bytes() == b"v2"


def test_stage_doc_id_matches_the_content_hash(storage):
    doc_id, path = storage.stage("report.pdf", b"content")
    assert doc_id == storage.doc_id(path)


def test_inbox_path_falls_back_to_the_bare_name_for_dropped_files(storage):
    """Files copied straight into the watched folder never went through
    stage(), so they carry no stamp."""
    dropped = storage.inbox / "dropped.pdf"
    dropped.write_bytes(b"content")

    assert storage.inbox_path("dropped.pdf", storage.doc_id(dropped)) == dropped


def test_archive_does_not_stamp_an_already_stamped_path(storage):
    """archive() gets the staged path, whose name already carries the stamp;
    stamping it again would file it where archived_path never looks."""
    doc_id, staged = storage.stage("report.pdf", b"content")

    archived = storage.archive(staged, doc_id, "report.pdf")

    assert archived == storage.archived_path("report.pdf", doc_id)
    assert archived.exists()


def test_restore_to_inbox_returns_false_when_original_missing(storage):
    assert storage.restore_to_inbox("gone.pdf", "deadbeef" * 8) is False
