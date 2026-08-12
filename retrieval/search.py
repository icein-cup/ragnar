from dataclasses import dataclass, field

from core.models import SearchResult

RELATED_COUNT = 3


def clears_floor(result: SearchResult, floor: float, vfloor: float) -> bool:
    """Keep a result if EITHER signal clears its floor.

    sigmoid(0) == 0.5 on the rerank scale, so "no opinion" and "irrelevant"
    land in the same place there; the raw vector score is the second opinion
    that rescues genuinely relevant content (table rows especially).
    """
    return result.score >= floor or (
        result.vector_score is not None and result.vector_score >= vfloor
    )


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    related: list[SearchResult] = field(default_factory=list)
    refused: bool = False


class Search:
    """Retrieve, rerank, then apply the similarity floor.

    When nothing clears the floor the LLM is never called — that is what
    makes the refusal trustworthy, since there is no opportunity for the
    model to improvise.
    """

    def __init__(self, embedder, store, reranker=None, candidates: int = 25,
                 top_k: int = 5, score_floor: float = 0.0,
                 vector_floor: float = 0.0):
        self._embedder = embedder
        self._store = store
        self.reranker = reranker
        self._candidates = candidates
        self._top_k = top_k
        # Public so the UI can adjust them per-query without rebuilding
        # Search.
        self.score_floor = score_floor
        self.vector_floor = vector_floor

    def find(self, question: str,
             doc_ids: list[str] | None = None,
             score_floor: float | None = None,
             vector_floor: float | None = None,
             use_reranker: bool = True,
             context_summary: str | None = None) -> SearchOutcome:
        """Retrieve, optionally rerank, then apply the similarity floor.

        doc_ids=None searches the whole corpus; a list scopes retrieval to
        just those documents (e.g. a user-selected subset in the UI).

        score_floor/vector_floor override the instance defaults for this one
        call, so the UI can pass per-request values without mutating shared
        state.

        A result is kept if EITHER the rerank score clears score_floor OR
        the original vector-similarity score clears vector_floor — refusal
        only fires when both signals are weak. This rescues content (table
        rows in particular) that the cross-encoder scores as near-neutral
        despite being genuinely relevant: sigmoid(0) == 0.5, so "no opinion"
        and "irrelevant" land in the same place on the rerank scale alone.

        use_reranker=False skips the cross-encoder entirely (much faster).
        Neither floor is applied in that mode: raw vector-similarity scores
        are on a different, uncalibrated scale, so refusal then happens only
        when retrieval finds nothing at all.
        """
        floor = self.score_floor if score_floor is None else score_floor
        vfloor = self.vector_floor if vector_floor is None else vector_floor
        search_query = f"{context_summary}\n\n{question}" if context_summary else question
        vector = self._embedder.embed([search_query])[0]
        candidates = self._store.search(
            vector, limit=self._candidates, doc_ids=doc_ids
        )

        if not candidates:
            return SearchOutcome(refused=True)

        if not (use_reranker and self.reranker is not None):
            return SearchOutcome(results=candidates[: self._top_k])

        ranked = self.reranker.rerank(search_query, candidates, self._top_k)
        kept = [r for r in ranked if clears_floor(r, floor, vfloor)]

        if not kept:
            return SearchOutcome(
                related=ranked[:RELATED_COUNT], refused=True
            )

        return SearchOutcome(results=kept)
