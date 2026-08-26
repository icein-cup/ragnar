import math
import os
import threading
from pathlib import Path

from core.models import SearchResult

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# When set, scoring is delegated to retrieval/rerank_server.py running on the
# host, where the cross-encoder can use the Mac's GPU. Docker on macOS has no
# access to Metal, and the gap is large: 1.48s vs 10.34s per 30 candidates on
# an M5 Pro. Unset (the default) keeps everything in-process.
RERANKER_URL = os.environ.get("RERANKER_URL", "").rstrip("/")

# One rerank is ~1.5s on the GPU, but a cold server still has to load the
# model, and fan-out queues requests behind the server's inference lock.
RERANK_TIMEOUT = 180.0

# Where eval/export_reranker_onnx.py writes the exported model. Under data/
# because that is bind-mounted (docker-compose.yml) and gitignored: the
# container's HuggingFace cache is ephemeral, so anything not written here is
# re-downloaded or re-exported on every image rebuild.
ONNX_DIR = Path(
    os.environ.get("RERANKER_ONNX_DIR", "/app/data/models/bge-reranker-v2-m3-onnx")
)

# Token cap per (query, chunk) pair. Left unset, sentence-transformers uses
# the model's own limit (8192 for bge-reranker-v2-m3) and nothing truncates.
#
# That matters because a cross-encoder pads every pair in a batch to the
# longest sequence in it, so ONE long chunk makes all 30 candidates cost as
# if they were that long. Measured on the HybridQA corpus (2026-08-26), same
# query count and near-identical total text in both cases:
#
#     longest chunk 1750 chars -> rerank 1.74s
#     longest chunk 6648 chars -> rerank 9.00s
#
# 5 chunks out of 2846 (0.2%) exceed 4000 chars, and those five set the
# latency ceiling for the whole system: any query retrieving one pays ~5x on
# every rerank in its fan-out. The corpus p99 is 1750 chars (~500 tokens),
# so a 512-token cap truncates only those outliers and leaves 99.8% of
# chunks scored exactly as before.
#
# This bounds rerank cost; it does not fix the oversized chunks themselves,
# which are an ingestion concern (see eval/docs/tuning-runbook.md Phase 4).
MAX_LENGTH = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))


class BGEReranker:
    """Cross-encoder reranker running on CPU inside the container.

    Measured at ~10s for 30 candidates on the torch backend, and the agentic
    path pays that once per generated query — the single largest cost in an
    eval run. When an ONNX export exists at ONNX_DIR this loads that instead,
    which runs the same model through a graph-optimized runtime.

    Falls back to torch when the export is absent so a fresh checkout still
    works before anyone has run the export script.

    When RERANKER_URL is set, scoring is delegated over HTTP to
    retrieval/rerank_server.py on the host instead, where the same model runs
    on the GPU (1.48s vs 10.34s per 30 candidates). Only the model call moves;
    the sigmoid and top_k cut below stay here, so both paths run identical
    code from the raw logits onward.
    """

    def __init__(self, model_name: str = MODEL_NAME, model=None,
                 url: str | None = None, client=None):
        self._model_name = model_name
        self._model = model
        self._url = RERANKER_URL if url is None else url.rstrip("/")
        self._client = client
        self._lock = threading.Lock()

    def _http_scores(self, query: str, texts: list[str]) -> list[float]:
        """Score via the host service. Raises rather than falling back.

        A silent fallback to the in-container model would be the worst
        outcome here: a sweep would keep running at 7x the latency it was
        budgeted for, and any score difference between the two devices would
        land in the middle of a run with nothing in the report to show it.
        Failing loudly costs one run; failing quietly costs the experiment.
        """
        import httpx
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.Client()
        response = self._client.post(
            f"{self._url}/rerank",
            json={"query": query, "texts": texts},
            timeout=RERANK_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["scores"]

    def _ensure_model(self):
        # Double-checked locking: this instance is shared (st.cache_resource)
        # across Streamlit sessions, and multi-query fan-out hits it from a
        # ThreadPoolExecutor, so two threads can both see self._model is None
        # and race to construct the multi-GB CrossEncoder.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    if ONNX_DIR.is_dir():
                        self._model = CrossEncoder(str(ONNX_DIR),
                                                   backend="onnx",
                                                   max_length=MAX_LENGTH)
                    else:
                        self._model = CrossEncoder(self._model_name,
                                                   max_length=MAX_LENGTH)
        return self._model

    def rerank(self, query: str, candidates: list[SearchResult],
               top_k: int) -> list[SearchResult]:
        if not candidates:
            return []

        if self._url:
            raw_scores = self._http_scores(
                query, [c.chunk.text for c in candidates])
        else:
            model = self._ensure_model()
            raw_scores = model.predict([(query, c.chunk.text) for c in candidates])

        # The cross-encoder emits logits, not probabilities. Sigmoid maps
        # them to (0, 1) so a single interpretable floor can be configured.
        rescored = [
            SearchResult(chunk=c.chunk, score=1 / (1 + math.exp(-float(s))),
                        vector_score=c.score)
            for c, s in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
