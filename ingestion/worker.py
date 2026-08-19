import logging
import threading

log = logging.getLogger(__name__)


class IngestWorker:
    """Drains the registry queue one document at a time.

    Sequential by design: Docling on CPU would only contend with itself in
    parallel, and concurrency would add failure modes for no gain at this
    scale.

    Runs on a background thread and touches only the registry, storage, and
    pipeline — never Streamlit APIs, which are not thread-safe.
    """

    def __init__(self, storage, registry, pipeline, poll_seconds: float = 1.0):
        self._storage = storage
        self._registry = registry
        self._pipeline = pipeline
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def process_next(self) -> bool:
        """Process one queued document. Returns False if the queue is empty."""
        doc = self._registry.next_queued()
        if doc is None:
            return False

        self._registry.mark_processing(doc.doc_id)
        path = self._storage.inbox_path(doc.filename, doc.doc_id)

        try:
            if not path.exists():
                raise FileNotFoundError(f"{doc.filename} missing from inbox")

            # A cached parse (from a prior ingest of this exact content, keyed
            # by the content-hash doc_id) lets re-chunking skip Docling
            # entirely. Only write the converted markdown / cache on a fresh
            # parse — on a cache hit both are already correct on disk.
            cached = self._storage.read_parsed(doc.doc_id)
            result = self._pipeline.ingest(path, doc.doc_id, parsed=cached,
                                           filename=doc.filename)
            if cached is None:
                self._storage.write_converted(doc.doc_id, result.markdown)
                self._storage.write_parsed(doc.doc_id, result.parsed)
            self._storage.archive(path, doc.doc_id, doc.filename)
            self._registry.mark_done(doc.doc_id, result.chunk_count)
        except Exception as exc:
            # Original deliberately stays in inbox for inspection.
            log.exception("ingestion failed for %s", doc.filename)
            self._registry.mark_failed(doc.doc_id, str(exc))

        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.process_next():
                    self._stop.wait(self._poll)
            except Exception:
                log.exception("worker loop error")
                self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._registry.reset_stale_processing()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
