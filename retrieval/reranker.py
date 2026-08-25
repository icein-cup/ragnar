import math
import threading

from core.models import SearchResult

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """Cross-encoder reranker running on CPU inside the container.

    Costs roughly 1-3s for 25 candidates. If that proves too slow, the ONNX
    export of the same model is a drop-in replacement that removes the torch
    dependency entirely.
    """

    def __init__(self, model_name: str = MODEL_NAME, model=None):
        self._model_name = model_name
        self._model = model
        self._lock = threading.Lock()

    def _ensure_model(self):
        # Double-checked locking: this instance is shared (st.cache_resource)
        # across Streamlit sessions, and multi-query fan-out hits it from a
        # ThreadPoolExecutor, so two threads can both see self._model is None
        # and race to construct the multi-GB CrossEncoder.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[SearchResult],
               top_k: int) -> list[SearchResult]:
        if not candidates:
            return []

        model = self._ensure_model()
        pairs = [(query, c.chunk.text) for c in candidates]
        raw_scores = model.predict(pairs)

        # The cross-encoder emits logits, not probabilities. Sigmoid maps
        # them to (0, 1) so a single interpretable floor can be configured.
        rescored = [
            SearchResult(chunk=c.chunk, score=1 / (1 + math.exp(-float(s))),
                        vector_score=c.score)
            for c, s in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
