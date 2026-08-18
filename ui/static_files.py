"""Serve archived originals over loopback HTTP so the browser can open them.

The app runs inside a Linux Docker container, where macOS-only commands like
``open`` don't exist, so citations can't shell out to a host PDF viewer.
Serving the originals on a loopback port and linking to them with a
``#page=N`` fragment hands the work to the browser, which runs on the host and
has its own PDF viewer.
"""
from __future__ import annotations

import logging
import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FILE_SERVER_PORT = 8510

_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None
_lock = threading.Lock()


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve files without spamming the log on every request."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def start_file_server(root: Path) -> str:
    """Start (once) a loopback HTTP server for ``root``; return its base URL.

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
            _server = ThreadingHTTPServer(("0.0.0.0", FILE_SERVER_PORT), handler)

            def _serve() -> None:
                try:
                    _server.serve_forever()
                except Exception as exc:
                    logging.exception("Loopback file server died: %s", exc)

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