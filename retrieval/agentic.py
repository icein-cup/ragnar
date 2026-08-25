"""Agentic RAG orchestration: query rewriting, multi-query retrieval,
multi-hop reasoning, and self-correction.
"""
from __future__ import annotations

import functools
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from core.models import SearchResult
from retrieval.search import RELATED_COUNT, SearchOutcome, clears_floor
from generation.answerer import build_excerpts, _recent
from generation.prompts import SYSTEM_PROMPT, build_user_prompt
from generation.agentic_prompts import (
    build_rewrite_prompt,
    build_multi_query_prompt,
    build_multi_hop_prompt,
    build_self_correction_prompt,
)

logger = logging.getLogger(__name__)

# Need at least N results clearing the floor. Loosely coupled to
# agentic.multi_query_count in config.yaml (3 by default): fast-path fires
# on the base search alone, or on base + multi-query fan-out together, so
# raising multi_query_count without also reconsidering this number changes
# how often the expensive multi-hop/self-correction stages get skipped.
FAST_PATH_MIN_RESULTS = 3

# Cap on concurrent Qdrant searches during multi-query fan-out. Currently
# always <= multi_query_count (3 by default), but stated explicitly rather
# than left implicit in the min() call below, so a future config bump to
# multi_query_count doesn't silently raise how many searches hit the store
# at once.
MAX_PARALLEL_QUERIES = 4

# Fallback rerank floor for the fast-path strength check, when the caller's
# score_floor is 0 (accepts everything, so it cannot distinguish strong from
# weak). Mirrors config.yaml's retrieval.score_floor default — kept as a
# named constant here rather than a bare literal so the two don't silently
# drift if one changes without the other.
FALLBACK_SCORE_FLOOR = 0.55

# Temperature for the two calls that emit a retrieval query (_rewrite and
# _generate_multi_queries). Everything else in this file stays at the
# caller's temperature — see the note where the partials are built.
QUERY_TEMPERATURE = 0.7

# Strips leading bullets/numbering ("1.", "-", "*", "1)") that a model adds
# despite being told not to.
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _clean_llm_lines(text: str) -> list[str]:
    """Clean each line of a multi-line LLM reply, stripping bullets/numbering
    and surrounding quotes, and dropping blanks and label lines like
    "Rewritten query:" that a model prepends despite instructions.
    """
    cleaned = []
    for raw in text.splitlines():
        line = _BULLET_RE.sub("", raw.strip()).strip("\"'").strip()
        if not line or line.endswith(":"):
            continue
        cleaned.append(line)
    return cleaned


@dataclass
class AgenticSearchOutcome(SearchOutcome):
    """A SearchOutcome that also carries the agentic reasoning trace."""
    # Reasoning trace for transparency / debugging
    rewritten_query: str | None = None
    queries_executed: list[str] = field(default_factory=list)
    hops_performed: int = 0
    self_corrected: bool = False
    correction_notes: str | None = None
    fast_path: bool = False  # True when expensive stages were skipped
    # Draft answer generated during self-correction. Reused by the UI to
    # avoid a duplicate LLM call when self-correction deemed the answer
    # complete. None when self-correction didn't run or triggered a
    # follow-up retrieval (the draft is stale in that case).
    draft_answer: str | None = None


class AgenticSearch:
    """Wraps a base Search with agentic capabilities.

    The agentic layer sits *above* the existing retrieval stack — it uses
    the same embedder, store, and reranker, but adds:

    1. Query rewriting — transforms vague / ambiguous queries before retrieval.
    2. Multi-query retrieval — generates N query variants, fuses results.
    3. Multi-hop retrieval — iteratively retrieves based on intermediate findings.
    4. Self-correction — evaluates answer completeness and re-retrieves if needed.

    Fast-path: when the base search returns strong results, skip the expensive
    LLM-based multi-hop and self-correction stages entirely.
    """

    def __init__(
        self,
        base_search,          # retrieval.search.Search instance
        llm,                    # generation.llm.OllamaLLM instance
        max_hops: int = 3,
        multi_query_count: int = 3,
        enable_rewrite: bool = True,
        enable_multi_query: bool = True,
        enable_multi_hop: bool = True,
        enable_self_correction: bool = True,
        fast_path_min_results: int = FAST_PATH_MIN_RESULTS,
        query_temperature: float = QUERY_TEMPERATURE,
    ):
        self._base_search = base_search
        self._llm = llm
        self._max_hops = max_hops
        self._multi_query_count = multi_query_count
        self._enable_rewrite = enable_rewrite
        self._enable_multi_query = enable_multi_query
        self._enable_multi_hop = enable_multi_hop
        self._enable_self_correction = enable_self_correction
        self._fast_path_min_results = fast_path_min_results
        self._query_temperature = query_temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        score_floor: float | None = None,
        vector_floor: float | None = None,
        use_reranker: bool = True,
        context_summary: str | None = None,
        history: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        enable_rewrite: bool | None = None,
        enable_multi_query: bool | None = None,
        enable_multi_hop: bool | None = None,
        enable_self_correction: bool | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> AgenticSearchOutcome:
        """Agentic retrieval pipeline.

        model/temperature/enable_* are threaded through per call, resolved
        against the instance defaults below, rather than mutated on self —
        this instance is shared (st.cache_resource) across concurrent
        Streamlit sessions, and _retrieve_multi_query also fans out across
        threads on it, so per-call state must never be assigned to self.

        on_progress, when given, is called with a short human-readable
        string at each stage boundary below — the pipeline can otherwise run
        30-120s (slow-path, see COMPARISONS.md) with no visible output at
        all. Called at stage granularity only, not per hop/query — enough
        for the UI to show it is working, not a full trace.

        Pipeline order:
        1. Optional query rewriting.
        2. Base search (fast-path check after this).
        3. Optional multi-query retrieval (parallel).
        4. Fast-path check: if results are strong, skip to answer.
        5. Optional multi-hop retrieval.
        6. Apply floors.
        7. Optional self-correction, evaluated against the floored results
           so a reused draft answer never rests on a chunk that got
           filtered out of the citations shown to the user.
        """
        def notify(message: str) -> None:
            if on_progress:
                on_progress(message)

        enable_rewrite = self._enable_rewrite if enable_rewrite is None else enable_rewrite
        enable_multi_query = (
            self._enable_multi_query if enable_multi_query is None else enable_multi_query
        )
        enable_multi_hop = (
            self._enable_multi_hop if enable_multi_hop is None else enable_multi_hop
        )
        enable_self_correction = (
            self._enable_self_correction if enable_self_correction is None
            else enable_self_correction
        )

        # Two generators, split by what the call produces rather than by
        # where it sits in the pipeline.
        #
        # gen_query drives the two calls whose entire output is a retrieval
        # query. Those want lexical variety — the point of asking for three
        # phrasings is that they differ, and greedy decoding gives three
        # near-identical ones. Published RAG setups run query rewriting warm
        # (CQC-RAG at 0.7) and synthesis cold, splitting on call type, not on
        # hop number; nothing supports varying by hop, so nothing here does.
        #
        # gen keeps the caller's temperature (0.0 by default) for everything
        # else: the self-correction draft, because it can be shipped verbatim
        # as the answer, and the multi-hop / self-correction evaluations,
        # because _parse_tag reads them against an exact "Sufficient:" /
        # "Complete:" format that sampling would put at risk.
        gen = functools.partial(self._llm.generate, model=model,
                                temperature=temperature)
        gen_query = functools.partial(self._llm.generate, model=model,
                                      temperature=self._query_temperature)
        outcome = AgenticSearchOutcome()

        # 1. Query rewriting
        if enable_rewrite:
            notify("Rewriting your question…")
        query = self._rewrite(question, context_summary, gen_query) \
                if enable_rewrite else question
        outcome.rewritten_query = query if enable_rewrite else None

        # 2. Base search (single query first — cheap)
        notify("Searching…")
        base_result = self._base_search.find(
            query, doc_ids=doc_ids,
            score_floor=score_floor, vector_floor=vector_floor,
            use_reranker=use_reranker, context_summary=context_summary,
        )
        outcome.queries_executed.append(query)
        outcome.related = base_result.related

        if not base_result.results and not base_result.related:
            outcome.refused = True
            return outcome

        all_results: list[SearchResult] = list(base_result.results)

        # 3. Multi-query retrieval (parallel, only if enabled and base wasn't great)
        if enable_multi_query and not self._is_fast_path(
                all_results, score_floor, vector_floor, use_reranker):
            notify("Expanding the search…")
            extra = self._retrieve_multi_query(
                query, doc_ids, score_floor, vector_floor, use_reranker,
                context_summary, outcome, gen_query,
            )
            all_results.extend(extra)

        # 4. Fast-path check: skip expensive stages if results are already strong
        if self._is_fast_path(all_results, score_floor, vector_floor,
                              use_reranker):
            outcome.fast_path = True
            final_results = self._fuse_results(all_results)
            return self._apply_floors(
                final_results, score_floor, vector_floor, use_reranker, outcome,
            )

        # 5. Multi-hop retrieval (expensive — only if needed)
        if enable_multi_hop:
            notify("Looking deeper…")
            all_results = self._multi_hop(
                question, all_results, doc_ids, score_floor, vector_floor,
                use_reranker, context_summary, outcome, gen,
            )

        # 6. Deduplicate, sort, and apply floors
        final_results = self._fuse_results(all_results)

        if not final_results:
            outcome.refused = True
            return outcome

        outcome = self._apply_floors(
            final_results, score_floor, vector_floor, use_reranker, outcome,
        )

        # 7. Self-correction check (expensive — only if multi-hop didn't
        # already run), evaluated against outcome.results — the floored
        # set the user will actually see cited.
        if enable_self_correction and outcome.hops_performed == 0 and outcome.results:
            notify("Double-checking the answer…")
            correction = self._self_correct(
                question, outcome.results, gen,
                context_summary=context_summary, history=history,
            )
            if correction:
                retrieved_more = False
                if correction.get("needs_more"):
                    outcome.self_corrected = True
                    outcome.correction_notes = correction.get("reason")
                    follow_up = correction.get("follow_up")
                    if follow_up and follow_up.lower() != "none":
                        extra = self._base_search.find(
                            follow_up, doc_ids=doc_ids,
                            score_floor=score_floor, vector_floor=vector_floor,
                            use_reranker=use_reranker, context_summary=context_summary,
                        )
                        if extra.results:
                            merged = self._fuse_results(outcome.results + extra.results)
                            outcome.hops_performed += 1
                            outcome = self._apply_floors(
                                merged, score_floor, vector_floor, use_reranker, outcome,
                            )
                            retrieved_more = True

                # The draft was generated from outcome.results and stays
                # valid for exactly as long as that set does. Only a
                # follow-up retrieval that actually landed makes it stale.
                # Discarding it whenever the grader merely *said* "needs
                # more" — while offering no usable follow-up query — threw
                # away a finished answer and made the UI generate a third
                # one from the same excerpts.
                if not retrieved_more:
                    outcome.draft_answer = correction.get("draft")

        return outcome

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_fast_path(self, results: list[SearchResult],
                      score_floor: float | None = None,
                      vector_floor: float | None = None,
                      use_reranker: bool = True) -> bool:
        """Check if results are already strong enough to skip expensive stages.

        A result set is "strong" when at least FAST_PATH_MIN_RESULTS chunks
        clear the floor. This means the base search already found relevant
        content, so multi-hop reasoning and self-correction are unlikely to
        add value.

        Which floor depends on what `r.score` actually is. With reranking on
        it is a rerank sigmoid and score_floor applies. With reranking off it
        is a raw cosine (~0.4-0.6) on an unrelated scale, so vector_floor —
        the floor calibrated for exactly that scale — is what it gets tested
        against. Testing a cosine against the rerank floor, as this used to,
        made the fast path fire essentially at random in the one mode whose
        whole purpose is speed.
        """
        if not results:
            return False

        if use_reranker:
            floor = (self._base_search.score_floor if score_floor is None
                     else score_floor)
            # A floor of 0 accepts everything and so cannot separate strong
            # from weak; fall back to the calibrated default for this test.
            if floor <= 0:
                floor = FALLBACK_SCORE_FLOOR
        else:
            floor = (self._base_search.vector_floor if vector_floor is None
                     else vector_floor)
            # No usable cosine threshold — decline to call anything strong
            # rather than invent a number for a scale we have not calibrated.
            if floor <= 0:
                return False

        strong = [r for r in results if r.score >= floor]
        return len(strong) >= self._fast_path_min_results

    def _apply_floors(
        self,
        final_results: list[SearchResult],
        score_floor: float | None,
        vector_floor: float | None,
        use_reranker: bool,
        outcome: AgenticSearchOutcome,
    ) -> AgenticSearchOutcome:
        """Apply score floors, updating results/refused/related on outcome in place."""
        if not final_results:
            outcome.results = []
            outcome.refused = True
            return outcome

        if use_reranker and self._base_search.reranker is not None:
            floor = self._base_search.score_floor if score_floor is None else score_floor
            vfloor = self._base_search.vector_floor if vector_floor is None else vector_floor
            kept = [r for r in final_results if clears_floor(r, floor, vfloor)]
            if not kept:
                outcome.results = []
                outcome.related = final_results[:RELATED_COUNT]
                outcome.refused = True
                return outcome
            final_results = kept

        outcome.results = final_results
        outcome.refused = False
        return outcome

    def _rewrite(
        self, question: str, context_summary: str | None, gen: Callable[..., str],
    ) -> str:
        """Use the LLM to rewrite the query for better retrieval."""
        system, user = build_rewrite_prompt(question, context_summary)
        try:
            raw = gen(system, user).strip()
            lines = _clean_llm_lines(raw)
            rewritten = lines[-1] if lines else ""
            if rewritten and len(rewritten) > 5:
                logger.debug("Rewrote query: %r -> %r", question, rewritten)
                return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
        return question

    def _generate_multi_queries(self, question: str, gen: Callable[..., str]) -> list[str]:
        """Generate N query variants."""
        system, user = build_multi_query_prompt(question, self._multi_query_count)
        try:
            raw = gen(system, user).strip()
            queries = _clean_llm_lines(raw)
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique = []
            for q in queries:
                key = q.lower().strip("?.")
                if key not in seen:
                    seen.add(key)
                    unique.append(q)
            if unique:
                logger.debug("Generated %d multi-queries", len(unique))
                return unique
        except Exception as exc:
            logger.warning("Multi-query generation failed: %s", exc)
        return [question]

    def _retrieve_multi_query(
        self,
        query: str,
        doc_ids: list[str] | None,
        score_floor: float | None,
        vector_floor: float | None,
        use_reranker: bool,
        context_summary: str | None,
        outcome: AgenticSearchOutcome,
        gen: Callable[..., str],
    ) -> list[SearchResult]:
        """Execute multi-query retrieval in parallel using a thread pool.

        Base searches are independent (different query embeddings), so we
        run them concurrently. The LLM multi-query generation still happens
        sequentially since it's one prompt.
        """
        queries = self._generate_multi_queries(query, gen)
        all_results: list[SearchResult] = []

        # Run all base searches concurrently
        with ThreadPoolExecutor(
            max_workers=min(len(queries), MAX_PARALLEL_QUERIES)
        ) as executor:
            future_to_query: dict = {}
            for q in queries:
                # Skip the original query if it was already searched
                if q == query and query in outcome.queries_executed:
                    continue
                outcome.queries_executed.append(q)
                future = executor.submit(
                    self._base_search.find,
                    q, doc_ids=doc_ids,
                    score_floor=score_floor, vector_floor=vector_floor,
                    use_reranker=use_reranker, context_summary=context_summary,
                )
                future_to_query[future] = q

            for future in as_completed(future_to_query):
                try:
                    result = future.result()
                    if result.results:
                        all_results.extend(result.results)
                    elif result.related:
                        all_results.extend(result.related)
                except Exception as exc:
                    q = future_to_query[future]
                    logger.warning("Multi-query search failed for %r: %s", q, exc)

        return all_results

    def _multi_hop(
        self,
        original_question: str,
        current_results: list[SearchResult],
        doc_ids: list[str] | None,
        score_floor: float | None,
        vector_floor: float | None,
        use_reranker: bool,
        context_summary: str | None,
        outcome: AgenticSearchOutcome,
        gen: Callable[..., str],
    ) -> list[SearchResult]:
        """Iteratively retrieve additional information if gaps remain."""
        accumulated = list(current_results)

        for hop in range(1, self._max_hops + 1):
            system, user = build_multi_hop_prompt(
                original_question, build_excerpts(accumulated))
            try:
                raw = gen(system, user).strip()
            except Exception as exc:
                logger.warning("Multi-hop reasoning failed at hop %d: %s", hop, exc)
                break

            sufficient = self._parse_tag(raw, "Sufficient", "no").lower()
            follow_up = self._parse_tag(raw, "FollowUp", "none")

            # "partial" means more hops could still help — only "yes" (fully
            # answered) or the absence of a follow-up query should stop the
            # loop. Treating "partial" as a stop condition made max_hops
            # effectively cap out at 1 hop regardless of its configured value.
            if sufficient == "yes" or follow_up.lower() in ("none", "", "n/a"):
                break

            outcome.hops_performed += 1
            outcome.queries_executed.append(f"hop{hop}: {follow_up}")

            extra = self._base_search.find(
                follow_up, doc_ids=doc_ids,
                score_floor=score_floor, vector_floor=vector_floor,
                use_reranker=use_reranker, context_summary=context_summary,
            )
            if extra.results:
                accumulated.extend(extra.results)
            elif extra.related:
                accumulated.extend(extra.related)
            else:
                # No more info available — stop
                break

        return accumulated

    def _self_correct(
        self,
        question: str,
        results: list[SearchResult],
        gen: Callable[..., str],
        *,
        context_summary: str | None = None,
        history: list[dict] | None = None,
    ) -> dict[str, str | bool | None] | None:
        """Evaluate whether the current results adequately answer the question.

        The draft is built with the same prompt construction that
        ``Answerer.answer`` uses — ``SYSTEM_PROMPT`` +
        ``build_user_prompt`` with ``context_summary`` and ``history``
        — so there is no drift between the draft and what the final
        answer path would produce.  When self-correction deems the
        answer complete, the draft is carried through the outcome and
        reused by the UI, avoiding a duplicate LLM call.
        """
        excerpts = build_excerpts(results)
        draft_prompt = build_user_prompt(
            question, excerpts, context_summary=context_summary,
        )
        try:
            draft = gen(
                SYSTEM_PROMPT, draft_prompt, history=_recent(history),
            )
        except Exception as exc:
            logger.warning("Self-correction draft generation failed: %s", exc)
            return None

        system, user = build_self_correction_prompt(question, excerpts, draft)
        try:
            raw = gen(system, user).strip()
        except Exception as exc:
            logger.warning("Self-correction evaluation failed: %s", exc)
            return None

        complete = self._parse_tag(raw, "Complete", "partial")
        contradictions = self._parse_tag(raw, "Contradictions", "no")
        improvement = self._parse_tag(raw, "Improvement", "none")

        needs_more = complete.lower() in ("no", "partial") or contradictions.lower() == "yes"
        follow_up = improvement if needs_more and improvement.lower() not in ("none", "", "n/a") else None
        return {
            "needs_more": needs_more,
            "reason": improvement,
            "follow_up": follow_up,
            "draft": draft,
        }

    @staticmethod
    def _fuse_results(results: list[SearchResult]) -> list[SearchResult]:
        """Deduplicate by chunk text and keep the best score per unique chunk.

        After sorting by score (descending), apply score-gap pruning: if there
        are more than 4 results and a clear gap (>0.05) separates the top-3
        from the rest, truncate to the top-3.  This drops fusion noise — the
        low-scoring chunks that multi-query and multi-hop retrieval inject
        without adding relevant signal — while preserving relevant chunks when
        scores are close enough that a gap-based cut would be unsafe.
        """
        best: dict[str, SearchResult] = {}
        for r in results:
            key = f"{r.chunk.doc_id}:{r.chunk.chunk_index}"
            existing = best.get(key)
            if existing is None or r.score > existing.score:
                best[key] = r
        fused = sorted(best.values(), key=lambda x: x.score, reverse=True)

        # Score-gap pruning: keep only the top-3 when there's a clear
        # separation (>0.05) between the 3rd and 4th results.  Only fires
        # when we have 6+ results, so small result sets (common in
        # multi-hop where every chunk may matter) are never cut.
        if len(fused) >= 6:
            gap = fused[2].score - fused[3].score
            if gap > 0.05:
                logger.debug(
                    "Score-gap pruning: top-3 score=%.3f, 4th=%.3f, gap=%.3f "
                    "— truncating %d results to 3",
                    fused[2].score, fused[3].score, gap, len(fused),
                )
                fused = fused[:3]

        return fused

    @staticmethod
    def _parse_tag(text: str, tag: str, default: str) -> str:
        """Parse a tagged value from structured LLM output.

        Matches "Tag: value" at the start of a line, case-insensitive. Tolerates
        a code fence or preamble the model adds despite instructions, and a
        value the model put on the line after the tag instead of beside it.
        """
        tag_lower = tag.lower()
        lines = [ln.strip() for ln in text.splitlines()]
        # Drop a leading/trailing code fence (``` or ~~~) if present.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        for i, line in enumerate(lines):
            if line.lower().startswith(f"{tag_lower}:"):
                value = line.split(":", 1)[1].strip()
                # A value that ends with ":" is a label, not a value — the
                # model put the real content on the following line.
                if value.endswith(":"):
                    value = value[:-1].strip()
                # An empty value means the model put it on the next line.
                if not value and i + 1 < len(lines):
                    value = lines[i + 1].strip()
                return value
        return default
