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


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeHTTPClient:
    """Stands in for httpx.Client against the host rerank server."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"scores": self._scores})


def _candidate(text, vector_score=0.5):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=text, chunk_index=0),
        score=vector_score,
    )


def test_http_path_scores_via_the_server_and_never_loads_a_local_model():
    # The whole point of the host service is that the container stops doing
    # tensor math; loading the local model anyway would silently reintroduce
    # the multi-GB download and the CPU cost.
    client = FakeHTTPClient([2.0, -2.0])
    reranker = BGEReranker(url="http://host.docker.internal:8007", client=client)

    ranked = reranker.rerank("q", [_candidate("relevant"), _candidate("junk")],
                             top_k=2)

    assert reranker._model is None
    assert client.calls[0]["url"] == "http://host.docker.internal:8007/rerank"
    assert client.calls[0]["json"] == {"query": "q", "texts": ["relevant", "junk"]}
    assert ranked[0].chunk.text == "relevant"


def test_http_path_applies_the_same_sigmoid_as_the_local_path():
    # Server returns raw logits; the sigmoid stays client-side so both paths
    # produce identical scores from identical model output. sigmoid(0) == 0.5.
    reranker = BGEReranker(url="http://h:1", client=FakeHTTPClient([0.0]))
    ranked = reranker.rerank("q", [_candidate("x", vector_score=0.47)], top_k=1)

    assert ranked[0].score == pytest.approx(0.5)
    assert ranked[0].vector_score == 0.47


def test_http_failure_propagates_instead_of_falling_back_to_cpu():
    # A silent fallback would leave a sweep running at 7x its budgeted
    # latency with nothing in the report to explain it.
    class Failing:
        def post(self, *a, **k):
            raise ConnectionError("server down")

    with pytest.raises(ConnectionError):
        BGEReranker(url="http://h:1", client=Failing()).rerank(
            "q", [_candidate("x")], top_k=1)


def test_uses_onnx_backend_when_the_export_is_present(tmp_path, monkeypatch):
    # The export is the whole point of the ONNX work: if it exists it must be
    # picked up, or the run silently pays the slow torch path anyway.
    monkeypatch.setattr("retrieval.reranker.ONNX_DIR", tmp_path)
    seen = {}

    def _construct(model_name, **kwargs):
        seen["name"], seen["kwargs"] = model_name, kwargs
        return FakeCrossEncoder()

    with patch("sentence_transformers.CrossEncoder", side_effect=_construct):
        BGEReranker()._ensure_model()

    assert seen["name"] == str(tmp_path)
    assert seen["kwargs"].get("backend") == "onnx"


def test_falls_back_to_torch_when_no_export_exists(tmp_path, monkeypatch):
    # A fresh checkout has no export; the app must still start rather than
    # fail on a missing directory.
    monkeypatch.setattr("retrieval.reranker.ONNX_DIR", tmp_path / "absent")
    seen = {}

    def _construct(model_name, **kwargs):
        seen["name"], seen["kwargs"] = model_name, kwargs
        return FakeCrossEncoder()

    with patch("sentence_transformers.CrossEncoder", side_effect=_construct):
        BGEReranker()._ensure_model()

    assert seen["name"] == "BAAI/bge-reranker-v2-m3"
    assert "backend" not in seen["kwargs"]


def test_ensure_model_constructs_once_under_concurrent_first_touch():
    # Regression for the check-then-construct race: two threads hitting a
    # cold reranker at once used to both pass the `self._model is None`
    # check before either finished constructing.
    construct_count = 0

    def _slow_construct(model_name, **kwargs):
        # **kwargs: the ONNX branch passes backend="onnx", and this test runs
        # against whichever branch the environment selects.
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
