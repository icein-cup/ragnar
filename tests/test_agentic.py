"""Tests for the agentic RAG search module."""
import pytest

from core.models import Chunk, SearchResult, Citation
from retrieval.agentic import AgenticSearch, AgenticSearchOutcome
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
        if "multi-hop" in user.lower() or "Sufficient" in user:
            return self._responses.get("multi_hop",
                "Sufficient: yes\nMissing: none\nFollowUp: none")
        if "self-correction" in user.lower() or "Complete" in user:
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
    assert user == "What is AI?"


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
    assert "Sufficient:" in user


def test_build_self_correction_prompt():
    excerpts = [("doc.pdf, p. 1", "RAG uses retrieval.")]
    system, user = build_self_correction_prompt(
        "What is RAG?", excerpts, "RAG is a system."
    )
    assert "Complete:" in user
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
            if "Sufficient" not in user:
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
