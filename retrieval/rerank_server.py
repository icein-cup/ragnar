"""Host-side reranker service, so the cross-encoder can use the Mac's GPU.

Docker on macOS runs a Linux VM with no access to Metal or the Neural Engine,
so anything inside the app container is CPU-only by construction. Measured on
an M5 Pro, 30 candidates per rerank:

    container CPU (torch)   10.34s
    container CPU (ONNX)     8.57s
    host CPU (torch)         6.58s
    host GPU (MPS)           1.48s   <- 7x faster than the container

The agentic path pays that cost once per generated query, which made it the
single largest cost in an eval run. This serves the same model from the host
so the container can keep its reproducible, pinned environment while the
expensive tensor math runs on hardware the container cannot reach — exactly
the arrangement Ollama already uses here.

Returns raw logits, not finished scores. BGEReranker applies the sigmoid and
the top_k cut itself, and keeping that math on the client side means the HTTP
path and the in-process path run identical code from the model output onward.

Run it on the HOST (not in Docker), using the repo's own virtualenv:

    .venv/bin/python retrieval/rerank_server.py
    .venv/bin/python retrieval/rerank_server.py --device cpu --port 8008

Then point the container at it (docker-compose.yml already does this):

    RERANKER_URL=http://host.docker.internal:8007
"""
import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8007
MAX_BODY_BYTES = 32 * 1024 * 1024  # ~30 chunks of table/prose text, generously

_model = None
# Multi-query fan-out sends up to MAX_PARALLEL_QUERIES concurrent requests
# (retrieval/agentic.py). GPU work serialises in the driver regardless, and
# concurrent MPS forward passes from multiple threads are not reliably safe,
# so inference is serialised here and the wins come from the device, not from
# overlapping requests.
_lock = threading.Lock()


def load_model(model_name: str, device: str):
    global _model
    from sentence_transformers import CrossEncoder
    logger.info("loading %s on %s ...", model_name, device)
    _model = CrossEncoder(model_name, device=device)
    # Warm up: the first forward pass compiles Metal kernels and allocates
    # buffers, and is several times slower than steady state. Doing it here
    # means the first real query does not eat that cost.
    _model.predict([("warmup query", "warmup document text")])
    logger.info("ready on %s", device)


def _score(query: str, texts: list[str]) -> list[float]:
    with _lock:
        raw = _model.predict([(query, t) for t in texts])
    return [float(s) for s in raw]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "loaded": _model is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "bad Content-Length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
            query, texts = payload["query"], payload["texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise TypeError("texts must be a list of strings")
        except (ValueError, KeyError, TypeError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return

        if not texts:
            self._send(200, {"scores": []})
            return

        try:
            self._send(200, {"scores": _score(query, texts)})
        except Exception as exc:                       # noqa: BLE001
            # A failure here must be visible: the client refuses to silently
            # fall back to the slow in-container path, because that would
            # change timings (and possibly scores) mid-sweep without a trace.
            logger.exception("rerank failed")
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 keeps this off the LAN; Docker still "
                             "reaches it via host.docker.internal.")
    parser.add_argument("--device", default="mps",
                        help="mps (Apple GPU), cpu, or cuda")
    parser.add_argument("--model", default=None,
                        help="defaults to config.yaml's models.reranker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    model_name = args.model
    if model_name is None:
        from core.config import Config
        model_name = Config().reranker_model

    load_model(model_name, args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
