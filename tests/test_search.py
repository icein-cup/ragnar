from core.models import Chunk, SearchResult
from retrieval.search import Search
from tests.fakes import FakeEmbedder


class StubStore:
    def __init__(self, results):
        self._results = results
        self.last_doc_ids = "not called"

    def search(self, vector, limit, doc_ids=None):
        self.last_doc_ids = doc_ids
        if doc_ids is not None and not doc_ids:
            return []
        if doc_ids is not None:
            return [r for r in self._results
                    if r.chunk.doc_id in doc_ids][:limit]
        return self._results[:limit]


class StubReranker:
    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, candidates, top_k):
        # Mirrors the real BGEReranker: rerank score replaces .score, but
        # the incoming vector score is preserved on vector_score.
        scored = [
            SearchResult(chunk=c.chunk, score=s, vector_score=c.score)
            for c, s in zip(candidates, self._scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _result(text, vector_score=0.5):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=text, chunk_index=0),
        score=vector_score,
    )


def test_results_below_floor_are_rejected():
    # vector_floor=1.1 is unreachable (scores are in (0, 1)) so this test
    # exercises the rerank-score floor in isolation, unaffected by the
    # vector-score rescue covered separately below.
    search = Search(
        FakeEmbedder(), StubStore([_result("irrelevant")]),
        reranker=StubReranker([0.05]), candidates=25, top_k=5,
        score_floor=0.3, vector_floor=1.1,
    )

    outcome = search.find("unrelated question")

    assert outcome.refused is True
    assert outcome.results == []
    assert len(outcome.related) == 1


def test_results_above_floor_are_returned():
    search = Search(
        FakeEmbedder(), StubStore([_result("relevant")]),
        reranker=StubReranker([0.9]), candidates=25, top_k=5,
        score_floor=0.3,
    )

    outcome = search.find("good question")

    assert outcome.refused is False
    assert len(outcome.results) == 1


def test_only_results_above_floor_survive_a_mixed_batch():
    search = Search(
        FakeEmbedder(),
        StubStore([_result("a"), _result("b"), _result("c")]),
        reranker=StubReranker([0.9, 0.1, 0.5]),
        candidates=25, top_k=5, score_floor=0.3, vector_floor=1.1,
    )

    outcome = search.find("q")

    assert [round(r.score, 1) for r in outcome.results] == [0.9, 0.5]


def test_empty_corpus_refuses_without_error():
    search = Search(FakeEmbedder(), StubStore([]),
                    reranker=StubReranker([]), score_floor=0.3)

    outcome = search.find("q")

    assert outcome.refused is True
    assert outcome.related == []


def test_score_floor_is_settable_at_runtime():
    # A single result scoring 0.5: refused under a strict floor, kept
    # under a lenient one, without rebuilding the Search object.
    search = Search(
        FakeEmbedder(), StubStore([_result("x")]),
        reranker=StubReranker([0.5]), candidates=25, top_k=5,
        score_floor=0.8, vector_floor=1.1,
    )
    assert search.find("q").refused is True

    search.score_floor = 0.3
    assert search.find("q").refused is False


def test_find_with_no_doc_ids_searches_everything():
    search = Search(FakeEmbedder(), StubStore([_result("a")]),
                    reranker=StubReranker([0.9]), score_floor=0.3)
    search.find("q")
    assert search._store.last_doc_ids is None


def test_find_scopes_results_to_selected_doc_ids():
    a = SearchResult(chunk=Chunk(doc_id="docA", filename="a.pdf", text="a",
                                  chunk_index=0), score=0.5)
    b = SearchResult(chunk=Chunk(doc_id="docB", filename="b.pdf", text="b",
                                  chunk_index=0), score=0.5)
    search = Search(FakeEmbedder(), StubStore([a, b]),
                    reranker=StubReranker([0.9, 0.9]), score_floor=0.3)

    outcome = search.find("q", doc_ids=["docA"])

    assert [r.chunk.doc_id for r in outcome.results] == ["docA"]


def test_find_with_empty_doc_ids_refuses_without_reranker_call():
    reranker = StubReranker([0.9])
    search = Search(FakeEmbedder(), StubStore([_result("a")]),
                    reranker=reranker, score_floor=0.3)

    outcome = search.find("q", doc_ids=[])

    assert outcome.refused is True


def test_per_request_score_floor_overrides_the_instance_default():
    # Instance floor 0.8 would refuse a 0.5 result; a per-call 0.3 keeps it,
    # without mutating the shared instance.
    search = Search(
        FakeEmbedder(), StubStore([_result("x")]),
        reranker=StubReranker([0.5]), candidates=25, top_k=5,
        score_floor=0.8, vector_floor=1.1,
    )
    assert search.find("q").refused is True
    assert search.find("q", score_floor=0.3).refused is False
    assert search.score_floor == 0.8  # unchanged


def test_disabling_the_reranker_returns_top_candidates_without_the_floor():
    # Reranker would score this 0.05 (below floor) and refuse; with reranking
    # off, the floor is bypassed and the top candidate comes straight back.
    reranker = StubReranker([0.05])
    search = Search(
        FakeEmbedder(), StubStore([_result("a")]),
        reranker=reranker, candidates=25, top_k=5, score_floor=0.3,
        vector_floor=1.1,
    )
    assert search.find("q").refused is True
    outcome = search.find("q", use_reranker=False)
    assert outcome.refused is False
    assert len(outcome.results) == 1


def test_disabling_the_reranker_still_refuses_an_empty_corpus():
    search = Search(FakeEmbedder(), StubStore([]),
                    reranker=StubReranker([]), score_floor=0.3)
    assert search.find("q", use_reranker=False).refused is True


# --- two-signal floor: rerank score OR vector score must clear its bar -----

def test_low_rerank_score_rescued_by_high_vector_score():
    # Mirrors the real corpus: a table-row chunk the cross-encoder scores as
    # neutral (0.50, below the 0.55 rerank floor) but that the embedder
    # placed clearly near the query (0.6 vector score, above the vector
    # floor). Refusing this outright is the bug — it must be kept.
    search = Search(
        FakeEmbedder(), StubStore([_result("table row", vector_score=0.60)]),
        reranker=StubReranker([0.50]), candidates=25, top_k=5,
        score_floor=0.55, vector_floor=0.45,
    )

    outcome = search.find("q")

    assert outcome.refused is False
    assert len(outcome.results) == 1


def test_refused_when_both_rerank_and_vector_scores_are_weak():
    search = Search(
        FakeEmbedder(), StubStore([_result("noise", vector_score=0.30)]),
        reranker=StubReranker([0.50]), candidates=25, top_k=5,
        score_floor=0.55, vector_floor=0.45,
    )

    outcome = search.find("q")

    assert outcome.refused is True


def test_high_rerank_score_kept_regardless_of_vector_score():
    # Unchanged behavior: a strong rerank score is sufficient on its own.
    search = Search(
        FakeEmbedder(), StubStore([_result("clear match", vector_score=0.10)]),
        reranker=StubReranker([0.90]), candidates=25, top_k=5,
        score_floor=0.55, vector_floor=0.45,
    )

    outcome = search.find("q")

    assert outcome.refused is False
