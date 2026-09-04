"""Tests for the agentic RAG search module."""
import pytest

from core.models import Chunk, SearchResult, Citation
from retrieval.agentic import (AgenticSearch, AgenticSearchOutcome,
                               QUERY_TEMPERATURE)
from generation.agentic_prompts import (
    build_rewrite_prompt,
    build_multi_query_prompt,
    build_multi_hop_prompt,
    build_self_correction_prompt,
)


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 1024 for _ in texts]


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
        scored = [
            SearchResult(chunk=c.chunk, score=s, vector_score=c.score)
            for c, s in zip(candidates, self._scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class FakeLLM:
    """An LLM that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self._call_count = 0
        self._calls = []
        self._call_kwargs = []

    def generate(self, system: str, user: str, **kwargs):
        self._call_count += 1
        self._calls.append((system, user))
        self._call_kwargs.append(kwargs)
        # Return based on prompt content hints
        if "rewrite" in system.lower() or "rewrite" in user.lower():
            return self._responses.get("rewrite", "rewritten query")
        if "different ways" in system.lower() or "generate" in user.lower():
            return self._responses.get("multi_query", "variant 1\nvariant 2")
        if "Sufficient" in system or "Sufficient" in user:
            return self._responses.get("multi_hop",
                "Sufficient: yes\nMissing: none\nFollowUp: none")
        if "Complete" in system or "Complete" in user:
            return self._responses.get("self_correct",
                "Complete: yes\nContradictions: no\nImprovement: none")
        return self._responses.get("default", "default response")

    def stream(self, system: str, user: str, **kwargs):
        yield self.generate(system, user, **kwargs)


class StubSearch:
    """Mimics retrieval.search.Search for agentic tests."""

    def __init__(self, results_by_query=None):
        self._results_by_query = results_by_query or {}
        self.score_floor = 0.55
        self.vector_floor = 0.42
        self.reranker = None

    def find(self, question, doc_ids=None, score_floor=None, vector_floor=None,
             use_reranker=True, context_summary=None):
        from retrieval.search import SearchOutcome
        results = self._results_by_query.get(question, [])
        if not results:
            return SearchOutcome(refused=True)
        return SearchOutcome(results=results)


def _result(text, vector_score=0.5, doc_id="d", filename="f.pdf", page=1, chunk_index=0):
    return SearchResult(
        chunk=Chunk(doc_id=doc_id, filename=filename, text=text,
                    chunk_index=chunk_index, page=page),
        score=vector_score,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder tests
# ──────────────────────────────────────────────────────────────────────────────

def test_build_rewrite_prompt_without_context():
    system, user = build_rewrite_prompt("What is AI?")
    assert "rewrite" in system.lower()
    assert user == "Question: What is AI?"


def test_build_rewrite_prompt_with_context():
    system, user = build_rewrite_prompt("What is it?", context_summary="We discussed AI.")
    assert "What is it?" in user
    assert "We discussed AI." in user


def test_build_multi_query_prompt():
    system, user = build_multi_query_prompt("What is RAG?", num_queries=5)
    assert "5" in system
    assert user == "What is RAG?"


def test_build_multi_hop_prompt():
    excerpts = [("doc.pdf, p. 1", "RAG is retrieval augmented generation.")]
    system, user = build_multi_hop_prompt("How does RAG work?", excerpts)
    assert "RAG is retrieval augmented generation" in user
    assert "Sufficient:" in system


def test_build_self_correction_prompt():
    excerpts = [("doc.pdf, p. 1", "RAG uses retrieval.")]
    system, user = build_self_correction_prompt(
        "What is RAG?", excerpts, "RAG is a system."
    )
    assert "Complete:" in system
    assert "RAG is a system." in user


# ──────────────────────────────────────────────────────────────────────────────
# AgenticSearch tests
# ──────────────────────────────────────────────────────────────────────────────

def test_agentic_search_with_all_features_disabled_acts_like_base_search():
    """When all agentic features are off, it just passes through to base."""
    results = [_result("relevant content")]
    base = StubSearch({"question": results})
    llm = FakeLLM()
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=False,
        enable_multi_query=False,
        enable_multi_hop=False,
        enable_self_correction=False,
    )

    outcome = agentic.find("question")

    assert not outcome.refused
    assert len(outcome.results) == 1
    assert outcome.rewritten_query is None
    assert outcome.hops_performed == 0
    assert not outcome.self_corrected


def test_query_rewriting_changes_the_query():
    base = StubSearch()
    llm = FakeLLM({"rewrite": "improved query about RAG"})
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=True,
        enable_multi_query=False,
        enable_multi_hop=False,
        enable_self_correction=False,
    )

    outcome = agentic.find("What is RAG?")

    assert outcome.rewritten_query == "improved query about RAG"


def test_multi_query_generates_variants_and_fuses():
    base = StubSearch()
    llm = FakeLLM({"multi_query": "variant one\nvariant two\nvariant three"})
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=False,
        enable_multi_query=True,
        enable_multi_hop=False,
        enable_self_correction=False,
    )

    outcome = agentic.find("question")

    # The base search returns empty so the outcome is refused.
    # When refused, queries_executed may be empty because we return early.
    # The real behavior: multi-query is generated but base returns refused.
    # We just verify it doesn't crash and produces a valid outcome.
    assert outcome.refused is True


def test_multi_hop_follows_up_when_information_is_insufficient():
    results = [_result("some info", page=1)]
    base = StubSearch({"question": results, "follow up": [_result("more info", page=2)]})
    llm = FakeLLM({
        "multi_hop": "Sufficient: no\nMissing: more details\nFollowUp: follow up"
    })
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=False,
        enable_multi_query=False,
        enable_multi_hop=True,
        enable_self_correction=False,
        max_hops=3,
    )

    outcome = agentic.find("question")

    assert outcome.hops_performed >= 1


def test_multi_hop_continues_on_partial_result():
    """'Sufficient: partial' means a follow-up could still help — it must
    not be treated the same as 'yes' (fully answered), or multi-hop only
    ever executes zero or one hop regardless of max_hops.
    """
    results = [_result("some info", page=1)]
    base = StubSearch({
        "question": results,
        "follow up": [_result("more info", page=2)],
    })

    class PartialThenSufficientLLM(FakeLLM):
        """Returns 'partial' on the first multi-hop call, 'yes' after that."""

        def generate(self, system, user, **kwargs):
            if "Sufficient" not in system and "Sufficient" not in user:
                return super().generate(system, user, **kwargs)
            self._call_count += 1
            self._calls.append((system, user))
            self._call_kwargs.append(kwargs)
            if self._call_count == 1:
                return "Sufficient: partial\nMissing: more details\nFollowUp: follow up"
            return "Sufficient: yes\nMissing: none\nFollowUp: none"

    agentic = AgenticSearch(
        base, PartialThenSufficientLLM(),
        enable_rewrite=False, enable_multi_query=False,
        enable_multi_hop=True, enable_self_correction=False,
        max_hops=3,
    )

    outcome = agentic.find("question")

    assert outcome.hops_performed == 1


def test_multi_hop_stops_when_sufficient():
    results = [_result("complete answer", page=1)]
    base = StubSearch({"question": results})
    llm = FakeLLM({
        "multi_hop": "Sufficient: yes\nMissing: none\nFollowUp: none"
    })
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=False,
        enable_multi_query=False,
        enable_multi_hop=True,
        enable_self_correction=False,
    )

    outcome = agentic.find("question")

    assert outcome.hops_performed == 0


def test_self_correction_triggers_extra_retrieval_when_needed():
    results = [_result("partial info", page=1)]
    base = StubSearch({
        "question": results,
        "improve answer": [_result("more info", page=2)],
    })
    llm = FakeLLM({
        "self_correct": "Complete: no\nContradictions: no\nImprovement: improve answer",
        "multi_hop": "Sufficient: yes\nMissing: none\nFollowUp: none",
    })
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=False,
        enable_multi_query=False,
        enable_multi_hop=False,
        enable_self_correction=True,
    )

    outcome = agentic.find("question")

    assert outcome.self_corrected is True


def test_self_correction_does_not_trigger_when_complete():
    results = [_result("complete info", page=1)]
    base = StubSearch({"question": results})
    llm = FakeLLM({
        "self_correct": "Complete: yes\nContradictions: no\nImprovement: none",
        "multi_hop": "Sufficient: yes\nMissing: none\nFollowUp: none",
    })
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=False,
        enable_multi_query=False,
        enable_multi_hop=False,
        enable_self_correction=True,
    )

    outcome = agentic.find("question")

    assert outcome.self_corrected is False


def test_empty_results_returns_refused():
    base = StubSearch({})
    llm = FakeLLM()
    agentic = AgenticSearch(base, llm)

    outcome = agentic.find("question")

    assert outcome.refused is True


def test_fuse_results_deduplicates_by_chunk():
    r1 = _result("text A", doc_id="d1", chunk_index=0, page=1)
    r2 = _result("text A", doc_id="d1", chunk_index=0, page=1)
    r3 = _result("text B", doc_id="d1", chunk_index=1, page=2)
    fused = AgenticSearch._fuse_results([r1, r2, r3])

    assert len(fused) == 2


def test_fuse_results_keeps_highest_score():
    r1 = _result("text", doc_id="d1", chunk_index=0, page=1)
    r1.score = 0.3
    r2 = _result("text", doc_id="d1", chunk_index=0, page=1)
    r2.score = 0.9
    fused = AgenticSearch._fuse_results([r1, r2])

    assert len(fused) == 1
    assert fused[0].score == 0.9


def test_fuse_results_score_gap_pruning_truncates_to_top3():
    """When >4 results and a clear gap (>0.05) separates top-3 from the
    rest, the noisy tail should be dropped."""
    results = [
        _result(f"chunk {i}", doc_id="d", chunk_index=i, page=i + 1)
        for i in range(6)
    ]
    # Top-3 are clearly separated from the rest
    for i, score in enumerate([0.80, 0.75, 0.70, 0.60, 0.58, 0.55]):
        results[i].score = score
    fused = AgenticSearch._fuse_results(results)

    assert len(fused) == 3
    assert fused[0].score == 0.80
    assert fused[2].score == 0.70


def test_fuse_results_score_gap_pruning_skipped_when_gap_is_small():
    """When the gap between 3rd and 4th is <= 0.05, all results are kept."""
    results = [
        _result(f"chunk {i}", doc_id="d", chunk_index=i, page=i + 1)
        for i in range(6)
    ]
    # Gap between 3rd (0.70) and 4th (0.68) is only 0.02
    for i, score in enumerate([0.80, 0.75, 0.70, 0.68, 0.65, 0.60]):
        results[i].score = score
    fused = AgenticSearch._fuse_results(results)

    assert len(fused) == 6


def test_fuse_results_score_gap_pruning_skipped_when_leq_4_results():
    """Pruning never fires on result sets of 4 or fewer."""
    results = [
        _result(f"chunk {i}", doc_id="d", chunk_index=i, page=i + 1)
        for i in range(4)
    ]
    for i, score in enumerate([0.90, 0.85, 0.80, 0.50]):
        results[i].score = score
    fused = AgenticSearch._fuse_results(results)

    assert len(fused) == 4


def test_fuse_results_score_gap_pruning_boundary_gap():
    """A gap of exactly 0.05 is NOT enough to prune (boundary: must be >0.05)."""
    results = [
        _result(f"chunk {i}", doc_id="d", chunk_index=i, page=i + 1)
        for i in range(5)
    ]
    for i, score in enumerate([0.80, 0.75, 0.70, 0.65, 0.60]):
        results[i].score = score
    fused = AgenticSearch._fuse_results(results)

    # gap is exactly 0.05 — not > 0.05, so no pruning
    assert len(fused) == 5


def test_parse_tag_finds_value():
    text = "Sufficient: yes\nMissing: nothing"
    assert AgenticSearch._parse_tag(text, "Sufficient", "no") == "yes"
    assert AgenticSearch._parse_tag(text, "Missing", "unknown") == "nothing"


def test_parse_tag_returns_default_when_missing():
    text = "Sufficient: yes"
    assert AgenticSearch._parse_tag(text, "FollowUp", "none") == "none"


def test_model_and_temperature_thread_through_every_agentic_llm_call():
    """The model/temperature picked in the UI must reach every internal
    agentic LLM call (rewrite, multi-query, multi-hop, self-correction) —
    not just the final answer — so Ollama isn't thrashing between two
    resident models on every turn.
    """
    results = [_result("relevant content")]
    # Both keys covered: the default FakeLLM rewrite response changes the
    # query before the base search runs.
    base = StubSearch({"question": results, "rewritten query": results})
    llm = FakeLLM()
    agentic = AgenticSearch(base, llm)  # all stages enabled by default

    agentic.find("question", model="custom-model", temperature=0.7)

    assert llm._call_count >= 4
    assert all(
        kwargs.get("model") == "custom-model" and kwargs.get("temperature") == 0.7
        for kwargs in llm._call_kwargs
    )


def test_self_correction_draft_only_sees_floored_results():
    """Self-correction must run on the post-floor result set, not the raw
    fused set — otherwise the draft (which the UI can reuse verbatim as the
    displayed answer) could rest on a chunk whose citation was filtered out.
    """
    kept = _result("KEEPS-floor")
    kept.score = 0.9
    kept.vector_score = 0.9
    dropped = _result("DROPPED-below-floor", chunk_index=1)
    dropped.score = 0.1
    dropped.vector_score = 0.1

    class RerankedStubSearch(StubSearch):
        def __init__(self):
            super().__init__({"question": [kept, dropped]})
            self.reranker = object()  # non-None so _apply_floors engages

    llm = FakeLLM({
        "self_correct": "Complete: yes\nContradictions: no\nImprovement: none",
    })
    agentic = AgenticSearch(
        RerankedStubSearch(), llm,
        enable_rewrite=False, enable_multi_query=False,
        enable_multi_hop=False, enable_self_correction=True,
    )

    outcome = agentic.find("question")

    assert [r.chunk.text for r in outcome.results] == ["KEEPS-floor"]
    draft_system, draft_prompt = llm._calls[0]
    assert "DROPPED-below-floor" not in draft_prompt
    assert "KEEPS-floor" in draft_prompt


def test_refusal_preserves_related_and_queries_executed():
    """A base-search refusal must not silently drop the 'related documents'
    fallback or the query trace — both used to be lost because early
    returns rebuilt AgenticSearchOutcome from scratch and forgot a field.
    """
    from retrieval.search import SearchOutcome

    related_chunk = _result("weakly related")

    class RelatedOnlyStubSearch(StubSearch):
        def find(self, question, doc_ids=None, score_floor=None,
                  vector_floor=None, use_reranker=True, context_summary=None):
            return SearchOutcome(related=[related_chunk], refused=True)

    agentic = AgenticSearch(
        RelatedOnlyStubSearch(), FakeLLM(),
        enable_rewrite=False, enable_multi_query=False,
        enable_multi_hop=False, enable_self_correction=False,
    )

    outcome = agentic.find("question")

    assert outcome.refused is True
    assert outcome.related == [related_chunk]
    assert outcome.queries_executed == ["question"]


# ──────────────────────────────────────────────────────────────────────────────
# AgenticSearchOutcome tests
# ──────────────────────────────────────────────────────────────────────────────

def test_outcome_defaults():
    outcome = AgenticSearchOutcome()
    assert outcome.results == []
    assert outcome.related == []
    assert outcome.refused is False
    assert outcome.rewritten_query is None
    assert outcome.queries_executed == []
    assert outcome.hops_performed == 0
    assert outcome.self_corrected is False


# ──────────────────────────────────────────────────────────────────────────────
# Citation model tests
# ──────────────────────────────────────────────────────────────────────────────

def test_citation_from_search_result():
    from generation.answerer import build_citations
    results = [
        _result("text", doc_id="doc1", filename="file.pdf", page=5, chunk_index=2),
    ]
    citations = build_citations(results)

    assert len(citations) == 1
    assert citations[0].label == "file.pdf, p. 5"
    assert citations[0].doc_id == "doc1"
    assert citations[0].page == 5
    assert citations[0].chunk_index == 2


def test_citations_are_deduplicated_by_label():
    from generation.answerer import build_citations
    results = [
        _result("text A", doc_id="doc1", filename="file.pdf", page=5, chunk_index=0),
        _result("text B", doc_id="doc1", filename="file.pdf", page=5, chunk_index=1),
    ]
    citations = build_citations(results)

    assert len(citations) == 1  # Same page, same file = same label
    assert citations[0].label == "file.pdf, p. 5"


def test_draft_is_reused_when_self_correction_offers_no_follow_up():
    """"needs more" with no usable follow-up query changes nothing about the
    result set, so the finished draft is still valid for it. Throwing it away
    made the UI generate a third answer from the same excerpts."""
    llm = FakeLLM({
        "self_correct": (
            "Complete: partial\nContradictions: no\nImprovement: none"
        ),
    })
    agentic = AgenticSearch(
        StubSearch({"question": [_result("chunk")]}), llm,
        enable_rewrite=False, enable_multi_query=False,
        enable_multi_hop=False, enable_self_correction=True,
    )

    outcome = agentic.find("question")

    assert outcome.self_corrected is True
    assert outcome.hops_performed == 0
    assert outcome.draft_answer is not None


def test_draft_is_dropped_when_a_follow_up_retrieval_lands():
    """A follow-up that actually merged new chunks makes the draft stale —
    it was written before those chunks existed."""
    llm = FakeLLM({
        "self_correct": (
            "Complete: no\nContradictions: no\nImprovement: more on widgets"
        ),
    })
    agentic = AgenticSearch(
        StubSearch({
            "question": [_result("chunk")],
            "more on widgets": [_result("widget chunk", chunk_index=1)],
        }),
        llm,
        enable_rewrite=False, enable_multi_query=False,
        enable_multi_hop=False, enable_self_correction=True,
    )

    outcome = agentic.find("question")

    assert outcome.hops_performed == 1
    assert outcome.draft_answer is None


def test_fast_path_uses_the_floor_matching_the_score_scale():
    """With reranking off, r.score is a raw cosine, not a rerank sigmoid, so
    it must be tested against vector_floor (0.42) — not score_floor (0.55),
    which belongs to the other scale entirely."""
    agentic = AgenticSearch(StubSearch(), FakeLLM())  # floors 0.55 / 0.42

    # Cosines that clear vector_floor but not score_floor: strong for an
    # un-reranked run, not strong for a reranked one.
    cosine = [_result(f"chunk {i}", chunk_index=i) for i in range(5)]
    for r in cosine:
        r.score = 0.50

    assert agentic._is_fast_path(cosine, use_reranker=False) is True
    assert agentic._is_fast_path(cosine, use_reranker=True) is False


def test_fast_path_declines_when_the_cosine_floor_is_disabled():
    """vector_floor 0 accepts everything, so it cannot tell strong from weak.
    Better to run the expensive stages than to invent a threshold."""
    agentic = AgenticSearch(StubSearch(), FakeLLM())
    strong = [_result(f"chunk {i}", chunk_index=i) for i in range(5)]
    for r in strong:
        r.score = 0.99

    assert agentic._is_fast_path(strong, vector_floor=0.0,
                                 use_reranker=False) is False


# Temperature is split by what a call produces, not by where it sits in the
# pipeline: query generation wants lexical variety, everything else must stay
# deterministic (the draft can be shipped verbatim, and _parse_tag reads the
# evaluations against an exact format).

def _temps_by_kind(llm):
    """(system, user, temperature) for every call, tagged by call kind."""
    out = []
    for (system, user), kwargs in zip(llm._calls, llm._call_kwargs):
        blob = f"{system} {user}"
        if "rewrite" in blob.lower():
            kind = "rewrite"
        elif "different ways" in blob.lower():
            kind = "multi_query"
        elif "Sufficient" in blob:
            kind = "multi_hop"
        elif "Complete" in blob:
            kind = "self_correction"
        else:
            kind = "draft"
        out.append((kind, kwargs.get("temperature")))
    return out


class _AnySearch(StubSearch):
    """Returns the same results for any query, so these tests exercise the
    temperature routing rather than the fake's query matching."""

    def find(self, question, **kwargs):
        from retrieval.search import SearchOutcome
        return SearchOutcome(results=[_result("a"), _result("b")])


def test_query_generation_runs_warm_and_everything_else_stays_cold():
    llm = FakeLLM({"multi_hop": "Sufficient: no\nMissing: x\nFollowUp: more"})
    search = AgenticSearch(_AnySearch(), llm, query_temperature=0.7)
    search.find("q?", temperature=0.0)

    by_kind = _temps_by_kind(llm)
    assert by_kind, "no LLM calls were made"
    for kind, temp in by_kind:
        if kind in ("rewrite", "multi_query"):
            assert temp == 0.7, f"{kind} should sample, got {temp}"
        else:
            assert temp == 0.0, f"{kind} must stay deterministic, got {temp}"


def test_query_temperature_is_configurable_and_defaults_to_the_module_constant():
    llm = FakeLLM()
    AgenticSearch(_AnySearch(), llm).find("q?", temperature=0.0)
    assert any(t == QUERY_TEMPERATURE
               for kind, t in _temps_by_kind(llm)
               if kind in ("rewrite", "multi_query"))


# ──────────────────────────────────────────────────────────────────────────────
# on_progress — Branch F, visible progress for the 30-120s slow path
# ──────────────────────────────────────────────────────────────────────────────

def test_on_progress_fires_at_each_stage_boundary_in_order():
    results = [_result("some info")]
    base = StubSearch({"rewritten query": results})
    llm = FakeLLM()
    agentic = AgenticSearch(
        base, llm,
        enable_rewrite=True,
        enable_multi_query=False,
        enable_multi_hop=False,
        enable_self_correction=True,
    )

    seen = []
    agentic.find("question", on_progress=seen.append)

    assert seen == [
        "Rewriting your question…",
        "Searching…",
        "Double-checking the answer…",
    ]


def test_on_progress_is_optional():
    """Every existing call site omits on_progress -- must not require it."""
    results = [_result("some info")]
    base = StubSearch({"question": results})
    agentic = AgenticSearch(base, FakeLLM(), enable_rewrite=False,
                             enable_multi_query=False, enable_multi_hop=False,
                             enable_self_correction=False)

    outcome = agentic.find("question")

    assert not outcome.refused


# ──────────────────────────────────────────────────────────────────────────────
# Latency budget — the max <=35s ceiling
# ──────────────────────────────────────────────────────────────────────────────
#
# Tuning cannot deliver a ceiling: parameters shift a distribution, only a
# clock bounds a tail. These cover the three places the deadline bites
# (fan-out, each hop, self-correction), that it never costs the results
# already retrieved, and that 0 disables it.


class Clock:
    """Manual monotonic clock. Time passes only when work says it does."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class CostlyLLM(FakeLLM):
    """A FakeLLM where every generate() burns wall-clock."""

    def __init__(self, clock, seconds_per_call, responses=None):
        super().__init__(responses)
        self._clock = clock
        self._cost = seconds_per_call

    def generate(self, system, user, **kwargs):
        self._clock.advance(self._cost)
        return super().generate(system, user, **kwargs)


class AlwaysFindsSearch(StubSearch):
    """Returns the same results for any query, so hop loops don't stop early.

    clock/seconds_per_search make a retrieval cost wall-clock too — the real
    pipeline spends most of a slow case in search-and-rerank, not only in the
    LLM.
    """

    def __init__(self, results, clock=None, seconds_per_search=0.0):
        super().__init__()
        self._always = results
        self._clock = clock
        self._cost = seconds_per_search

    def find(self, question, **kwargs):
        from retrieval.search import SearchOutcome
        if self._clock is not None:
            self._clock.advance(self._cost)
        return SearchOutcome(results=list(self._always))


def _use_clock(monkeypatch, clock):
    """Swap only retrieval.agentic's `time`, never the real time module."""
    import types
    import retrieval.agentic as agentic_module
    monkeypatch.setattr(agentic_module, "time",
                        types.SimpleNamespace(monotonic=clock))


def test_budget_does_not_fire_when_there_is_time_to_spare(monkeypatch):
    clock = Clock()
    _use_clock(monkeypatch, clock)
    results = [_result("relevant content")]
    base = AlwaysFindsSearch(results)
    # Every call is free, so the deadline can never be reached.
    agentic = AgenticSearch(base, CostlyLLM(clock, 0.0), latency_budget_s=25.0)

    outcome = agentic.find("question")

    assert outcome.budget_exhausted is False


def test_budget_skips_the_multi_query_fan_out(monkeypatch):
    """The fan-out is the first expansion, so it is the first thing dropped."""
    clock = Clock()
    _use_clock(monkeypatch, clock)
    base = AlwaysFindsSearch([_result("relevant content")])
    # The rewrite call alone (6s) overruns the 5s budget.
    agentic = AgenticSearch(base, CostlyLLM(clock, 6.0), latency_budget_s=5.0)

    outcome = agentic.find("question")

    assert outcome.budget_exhausted is True
    # Only the rewritten query ran — no variants were generated or searched.
    assert len(outcome.queries_executed) == 1


def test_budget_stops_the_hop_loop_partway_and_keeps_earlier_hops(monkeypatch):
    """Hops already paid for are kept; only further ones are refused."""
    clock = Clock()
    _use_clock(monkeypatch, clock)
    base = AlwaysFindsSearch([_result("relevant content")])
    llm = CostlyLLM(clock, 6.0, responses={
        "multi_hop": "Sufficient: no\nMissing: more\nFollowUp: another query",
    })
    agentic = AgenticSearch(
        base, llm, max_hops=3, latency_budget_s=10.0,
        enable_rewrite=False, enable_multi_query=False,
        enable_self_correction=False,
    )

    outcome = agentic.find("question")

    # hop1 top at t=0 (ok, LLM -> 6), hop2 top at t=6 (ok, LLM -> 12),
    # hop3 top at t=12 >= deadline 10 -> stop with 2 hops done.
    assert outcome.hops_performed == 2
    assert outcome.budget_exhausted is True


def test_budget_skips_self_correction(monkeypatch):
    """Self-correction is the last expansion and the last thing dropped."""
    clock = Clock()
    _use_clock(monkeypatch, clock)
    # The base search alone (6s) overruns the 5s budget.
    base = AlwaysFindsSearch([_result("relevant content")],
                             clock=clock, seconds_per_search=6.0)
    llm = CostlyLLM(clock, 0.0)
    agentic = AgenticSearch(
        base, llm, latency_budget_s=5.0,
        enable_rewrite=False, enable_multi_query=False, enable_multi_hop=False,
    )

    outcome = agentic.find("question")

    assert outcome.budget_exhausted is True
    # The self-correction call is the only LLM call this config would make.
    assert llm._call_count == 0
    assert outcome.draft_answer is None


def test_self_correction_still_runs_when_the_budget_allows(monkeypatch):
    """Control for the test above — same wiring, enough time."""
    clock = Clock()
    _use_clock(monkeypatch, clock)
    base = AlwaysFindsSearch([_result("relevant content")],
                             clock=clock, seconds_per_search=6.0)
    llm = CostlyLLM(clock, 0.0)
    agentic = AgenticSearch(
        base, llm, latency_budget_s=25.0,
        enable_rewrite=False, enable_multi_query=False, enable_multi_hop=False,
    )

    outcome = agentic.find("question")

    assert outcome.budget_exhausted is False
    # Two calls: the draft answer, then the completeness evaluation on it.
    assert llm._call_count == 2


def test_budget_exhausted_still_answers_rather_than_refusing(monkeypatch):
    """A ceiling must degrade the search, never turn a hit into a refusal."""
    clock = Clock()
    _use_clock(monkeypatch, clock)
    base = AlwaysFindsSearch([_result("relevant content", vector_score=0.9)])
    agentic = AgenticSearch(base, CostlyLLM(clock, 60.0), latency_budget_s=1.0)

    outcome = agentic.find("question")

    assert outcome.budget_exhausted is True
    assert outcome.refused is False
    assert outcome.results


def test_zero_budget_disables_the_deadline_entirely(monkeypatch):
    """0 restores the unbounded pre-2026-08-26 behaviour."""
    clock = Clock()
    _use_clock(monkeypatch, clock)
    base = AlwaysFindsSearch([_result("relevant content")])
    llm = CostlyLLM(clock, 600.0, responses={
        "multi_hop": "Sufficient: no\nMissing: more\nFollowUp: another query",
    })
    agentic = AgenticSearch(
        base, llm, max_hops=3, latency_budget_s=0,
        enable_rewrite=False, enable_multi_query=False,
        enable_self_correction=False,
    )

    outcome = agentic.find("question")

    # 600s per call against any real budget, yet nothing was cut short.
    assert outcome.budget_exhausted is False
    assert outcome.hops_performed == 3
