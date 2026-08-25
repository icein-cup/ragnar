import pytest
import uuid
from core.models import Chunk
from retrieval.store import QdrantStore


@pytest.fixture
def store():
    import os
    name = f"test_{uuid.uuid4().hex[:8]}"
    s = QdrantStore(os.environ["QDRANT_URL"], name, dim=1024)
    s.ensure_collection()
    yield s
    s.drop_collection()


def _chunk(doc_id, text, index=0, page=1):
    return Chunk(doc_id=doc_id, filename="f.pdf", text=text,
                 chunk_index=index, page=page)


@pytest.mark.integration
def test_store_roundtrips_chunk_and_payload(store):
    store.upsert([_chunk("d1", "hello")], [[0.1] * 1024])

    results = store.search([0.1] * 1024, limit=5)

    assert len(results) == 1
    assert results[0].chunk.text == "hello"
    assert results[0].chunk.page == 1
    assert results[0].chunk.doc_id == "d1"


@pytest.mark.integration
def test_delete_by_doc_removes_only_that_document(store):
    store.upsert(
        [_chunk("d1", "keep"), _chunk("d2", "remove", index=1)],
        [[0.1] * 1024, [0.2] * 1024],
    )

    store.delete_by_doc("d2")
    results = store.search([0.1] * 1024, limit=10)

    assert [r.chunk.doc_id for r in results] == ["d1"]


@pytest.mark.integration
def test_reupsert_same_chunk_does_not_duplicate(store):
    chunk = _chunk("d1", "v1")
    store.upsert([chunk], [[0.1] * 1024])
    store.upsert([chunk], [[0.1] * 1024])

    assert len(store.search([0.1] * 1024, limit=10)) == 1


@pytest.mark.integration
def test_search_with_doc_ids_scopes_to_selected_documents(store):
    store.upsert(
        [_chunk("d1", "alpha"), _chunk("d2", "beta"), _chunk("d3", "gamma")],
        [[0.1] * 1024, [0.1] * 1024, [0.1] * 1024],
    )

    results = store.search([0.1] * 1024, limit=10, doc_ids=["d1", "d3"])

    assert {r.chunk.doc_id for r in results} == {"d1", "d3"}


@pytest.mark.integration
def test_search_with_empty_doc_ids_returns_nothing_without_erroring(store):
    store.upsert([_chunk("d1", "alpha")], [[0.1] * 1024])

    results = store.search([0.1] * 1024, limit=10, doc_ids=[])

    assert results == []


@pytest.mark.integration
def test_search_with_none_doc_ids_is_unfiltered(store):
    store.upsert(
        [_chunk("d1", "alpha"), _chunk("d2", "beta")],
        [[0.1] * 1024, [0.1] * 1024],
    )

    results = store.search([0.1] * 1024, limit=10, doc_ids=None)

    assert {r.chunk.doc_id for r in results} == {"d1", "d2"}


@pytest.mark.integration
def test_delete_stale_removes_indices_not_in_keep_set(store):
    store.upsert(
        [_chunk("d1", "keep", index=0), _chunk("d1", "stale", index=1),
         _chunk("d2", "untouched", index=1)],
        [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024],
    )

    store.delete_stale("d1", keep_indices={0})
    results = store.search([0.1] * 1024, limit=10)

    assert sorted((r.chunk.doc_id, r.chunk.chunk_index) for r in results) == [
        ("d1", 0), ("d2", 1),
    ]


@pytest.mark.integration
def test_delete_stale_with_empty_keep_indices_removes_whole_doc(store):
    store.upsert(
        [_chunk("d1", "gone", index=0), _chunk("d2", "untouched", index=0)],
        [[0.1] * 1024, [0.2] * 1024],
    )

    store.delete_stale("d1", keep_indices=set())
    results = store.search([0.1] * 1024, limit=10)

    assert [r.chunk.doc_id for r in results] == ["d2"]
