"""Serve archived originals over loopback HTTP so the browser can open them.

The app runs inside a Linux Docker container, where macOS-only commands like
``open`` don't exist, so citations can't shell out to a host PDF viewer.
Serving the originals on a loopback port and linking to them with a
``#page=N`` fragment hands the work to the browser, which runs on the host and
has its own PDF viewer.
"""
from __future__ import annotations

import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FILE_SERVER_PORT = 8510

_server: ThreadingHTTPServer | None = None
_lock = threading.Lock()


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve files without spamming the log on every request."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def start_file_server(root: Path) -> str:
    """Start (once) a loopback HTTP server for ``root``; return its base URL.

    Idempotent — a module-level guard means later calls with an unchanged root
    reuse the first server instead of failing on an already-bound port.
    """
    global _server

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if _server is None:
        with _lock:
            if _server is None:
                handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
                    *args, directory=str(root), **kwargs
                )
                _server = ThreadingHTTPServer(("0.0.0.0", FILE_SERVER_PORT), handler)
                threading.Thread(target=_server.serve_forever, daemon=True).start()

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