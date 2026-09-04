"""Serve archived originals over HTTP so the browser can open them.

The app runs inside a Linux Docker container, where macOS-only commands like
``open`` don't exist, so citations can't shell out to a host PDF viewer.
Serving the originals on a local port and linking to them with a ``#page=N``
fragment hands the work to the browser, which runs on the host and has its
own PDF viewer.

The socket binds ``0.0.0.0`` by default because it has to: inside a
container, binding 127.0.0.1 would make Docker's published port unreachable.
What keeps the archive off the network is the *publish* — docker-compose maps
``127.0.0.1:8510:8510``, so only the host reaches it. Running the app
natively (``streamlit run ui/app.py``) has no such mapping and would put every
archived original on the LAN unauthenticated; set ``FILE_SERVER_HOST=127.0.0.1``
for that case. There is no auth on this server, by design and by assumption.
"""
from __future__ import annotations

import logging
import os
import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FILE_SERVER_PORT = 8510
# 0.0.0.0 is required under Docker; override to 127.0.0.1 when running the
# app directly on a host that is not alone on its network.
FILE_SERVER_HOST = os.environ.get("FILE_SERVER_HOST", "0.0.0.0")

_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None
_lock = threading.Lock()


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve files without spamming the log on every request."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def start_file_server(root: Path) -> str:
    """Start (once) a local HTTP server for ``root``; return its base URL.

    Idempotent — a module-level guard means later calls reuse the first
    server. If the serving thread died, restart it.
    """
    global _server, _server_thread

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    with _lock:
        if _server is not None and (_server_thread is None or not _server_thread.is_alive()):
            try:
                _server.server_close()
            except Exception:
                pass
            _server = None
            _server_thread = None

        if _server is None:
            handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
                *args, directory=str(root), **kwargs
            )
            _server = ThreadingHTTPServer(
                (FILE_SERVER_HOST, FILE_SERVER_PORT), handler)

            def _serve() -> None:
                try:
                    _server.serve_forever()
                except Exception as exc:
                    logging.exception("File server died: %s", exc)

            _server_thread = threading.Thread(target=_serve, daemon=True)
            _server_thread.start()

    return f"http://localhost:{FILE_SERVER_PORT}"


def file_url(base_url: str, archived_name: str, page: int | None = None) -> str:
    """URL for an archived file, with a ``#page=N`` fragment for PDFs.

    The fragment is honored by browser PDF viewers (Chrome, Firefox, Safari),
    which is what makes "open at the relevant page" work without a host-side
    viewer command.
    """
    url = f"{base_url}/{urllib.parse.quote(archived_name)}"
    if page is not None:
        url += f"#page={page}"
    return url