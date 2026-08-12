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

FAST_PATH_MIN_RESULTS = 3         # Need at least N results clearing the floor

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
    ) -> AgenticSearchOutcome:
        """Agentic retrieval pipeline.

        model/temperature/enable_* are threaded through per call, resolved
        against the instance defaults below, rather than mutated on self —
        this instance is shared (st.cache_resource) across concurrent
        Streamlit sessions, and _retrieve_multi_query also fans out across
        threads on it, so per-call state must never be assigned to self.

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

        gen = functools.partial(self._llm.generate, model=model, temperature=temperature)
        outcome = AgenticSearchOutcome()

        # 1. Query rewriting
        query = self._rewrite(question, context_summary, gen) \
                if enable_rewrite else question
        outcome.rewritten_query = query if enable_rewrite else None

        # 2. Base search (single query first — cheap)
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
        if enable_multi_query and not self._is_fast_path(all_results, score_floor):
            extra = self._retrieve_multi_query(
                query, doc_ids, score_floor, vector_floor, use_reranker,
                context_summary, outcome, gen,
            )
            all_results.extend(extra)

        # 4. Fast-path check: skip expensive stages if results are already strong
        if self._is_fast_path(all_results, score_floor):
            outcome.fast_path = True
            final_results = self._fuse_results(all_results)
            return self._apply_floors(
                final_results, score_floor, vector_floor, use_reranker, outcome,
            )

        # 5. Multi-hop retrieval (expensive — only if needed)
        if enable_multi_hop:
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
            correction = self._self_correct(
                question, outcome.results, gen,
                context_summary=context_summary, history=history,
            )
            if correction:
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
                else:
                    # Answer was deemed complete — carry the draft so the
                    # UI can display it without a duplicate LLM call.
                    outcome.draft_answer = correction.get("draft")

        return outcome

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_fast_path(self, results: list[SearchResult], score_floor: float | None = None) -> bool:
        """Check if results are already strong enough to skip expensive stages.

        A result set is "strong" when at least FAST_PATH_MIN_RESULTS chunks
        have a score above the configured score_floor (or a default of 0.55).
        This means the base search already found relevant content, so multi-hop
        reasoning and self-correction are unlikely to add value.
        """
        if not results:
            return False
        floor = score_floor if score_floor is not None else self._base_search.score_floor
        # If floor is 0 (reranker disabled), use a reasonable default
        if floor <= 0:
            floor = 0.55
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
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
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
        """Deduplicate by chunk text and keep the best score per unique chunk."""
        best: dict[str, SearchResult] = {}
        for r in results:
            key = f"{r.chunk.doc_id}:{r.chunk.chunk_index}"
            existing = best.get(key)
            if existing is None or r.score > existing.score:
                best[key] = r
        fused = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return fused

    @staticmethod
    def _parse_tag(text: str, tag: str, default: str) -> str:
        """Parse a tagged value from structured LLM output.

        Matches "Tag: value" at the start of a line, case-insensitive.
        """
        tag_lower = tag.lower()
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith(f"{tag_lower}:"):
                return line.split(":", 1)[1].strip()
        return default
