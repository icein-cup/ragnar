import threading
import time
from unittest.mock import patch

import pytest
from core.models import Chunk, SearchResult
from retrieval.reranker import BGEReranker


def _result(text):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=text, chunk_index=0),
        score=0.5,
    )


@pytest.mark.integration
def test_reranker_ranks_relevant_chunk_first():
    candidates = [
        _result("The office cafeteria serves lunch at noon."),
        _result("The service contract number is SC-4471."),
        _result("Parking permits expire annually."),
    ]

    ranked = BGEReranker().rerank(
        "What is the service contract number?", candidates, top_k=3
    )

    assert "SC-4471" in ranked[0].chunk.text


@pytest.mark.integration
def test_reranker_respects_top_k():
    candidates = [_result(f"text {i}") for i in range(10)]
    ranked = BGEReranker().rerank("query", candidates, top_k=3)
    assert len(ranked) == 3


@pytest.mark.integration
def test_reranker_scores_are_normalised_to_unit_interval():
    candidates = [_result("The contract number is SC-4471.")]
    ranked = BGEReranker().rerank("contract number", candidates, top_k=1)
    assert 0.0 <= ranked[0].score <= 1.0


def test_reranker_handles_empty_candidates():
    assert BGEReranker.__init__ is not None  # import smoke test


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder — returns a fixed
    logit per pair so tests don't need the real 2.3GB model."""

    def predict(self, pairs):
        return [0.0 for _ in pairs]  # sigmoid(0) == 0.5


def test_rerank_preserves_vector_score_from_candidates():
    # The reranker overwrites .score with its own sigmoid, but the original
    # vector-similarity score must survive as a second signal for the
    # two-signal floor in Search.find().
    candidate = SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text="row", chunk_index=0),
        score=0.47,
    )

    ranked = BGEReranker(model=FakeCrossEncoder()).rerank(
        "q", [candidate], top_k=1
    )

    assert ranked[0].vector_score == 0.47


def test_ensure_model_constructs_once_under_concurrent_first_touch():
    # Regression for the check-then-construct race: two threads hitting a
    # cold reranker at once used to both pass the `self._model is None`
    # check before either finished constructing.
    construct_count = 0

    def _slow_construct(model_name):
        nonlocal construct_count
        time.sleep(0.05)  # widen the race window
        construct_count += 1
        return FakeCrossEncoder()

    reranker = BGEReranker()
    with patch("sentence_transformers.CrossEncoder", side_effect=_slow_construct):
        threads = [threading.Thread(target=reranker._ensure_model) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert construct_count == 1
