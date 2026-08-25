# Refactor Suggestions — RAGnar (branch `iteration_ai`)

Non-bugging improvements to structure, robustness, and maintainability.
Ordered roughly by impact. No code was changed.

---

## 1. Make the SQLite layer harder to misuse

`core/db.py` shares a **single connection across the UI and worker threads**
(`check_same_thread=False`) guarded by a hand-passed `threading.Lock`. The
contract — "every caller must hold the lock around reads and writes" — is
enforced only by docstring. The moment a future read forgets the lock, you get
"database is locked" or "recursive use of cursors" errors that surface
intermittently under concurrency.

Suggestions:
- Enable `PRAGMA journal_mode=WAL` and set a `busy_timeout` on connect.
- Or move to connection-per-thread (the worker is already a fixed thread pool;
  the UI is one thread).
- Consider a small repository base class so `Registry` and `ChatStore` stop
  re-implementing the same row-mapping / lock boilerplate independently.

---

## 2. Centralize `Config` with a schema

`core/config.py` is ~20 thin `@property` passthroughs into `self._raw`, all
keyed by string literals (`self._raw["models"]["llm"]`, `["retrieval"]["top_k"]`,
…). A typo'd key raises `KeyError` deep inside a request rather than at startup.
`agentic:` keys are instead splatted into `AgenticSearch(**cfg.agentic)` — the
one place a bad key *does* fail loudly at startup, and the code comment calls
out the inconsistency.

Suggestions: a dataclass/TypedDict/pydantic model for the config shape, loaded
once and validated at startup, so every key is checked before the first request
and property access is type-safe.

---

## 3. Move the file-server side effect out of import time

`ui/services.py:32` calls `start_file_server(...)` at module import, which also
constructs a *second* `Config()` and `Storage()` that `build_services()` then
re-builds. Importing `list_chat_models` for a test or a utility now also spins
up an HTTP server. Move the server start into `build_services()` (it's already
cached) and pass the root through.

---

## 4. Snapshot mutable floor fallbacks instead of reading shared state

`retrieval/agentic.py` — `_is_fast_path` and `_apply_floors` fall back to
`self._base_search.score_floor` / `.vector_floor` when the caller passes `None`.
Those are *public mutable* attributes (`search.py` notes the UI "can adjust
them per-query"). Reading them as defaults means a prior request's mutation can
leak into a later fast-path decision. Capture the defaults at construction, or
always pass explicit floors and drop the shared-state fallback.

---

## 5. Guard the lazy cross-encoder load

`retrieval/reranker.py` — `_ensure_model()` constructs the `CrossEncoder`
lazily with no lock. If two threads first-touch it concurrently (e.g. two
Streamlit sessions on a shared `Search`), the model can be constructed twice
(a multi-GB spike). Load it eagerly in `build_services()` or guard the lazy
load with a lock.

---

## 6. Explicitly handle the Linux-only memory/CPU detection

`core/config.py` — `_cgroup_memory_limit_bytes()` and `_cpu_count()` read
`/proc` and `/sys/fs/cgroup`, which do not exist on macOS. Native runs
(`streamlit run ui/app.py`, documented in `static_files.py`) silently return
`None` and `auto` worker count collapses to the `max_workers` cap. That may be
acceptable, but make it explicit: detect the platform and log "native run —
worker auto-sizing unavailable", rather than silently hitting the fallback.

---

## 7. Defensive guards in `auto_worker_count` and `QdrantStore.search`

- `auto_worker_count(worker_memory_gb, ...)`: `mem_gb // worker_memory_gb`
  raises `ZeroDivisionError` if `worker_memory_gb` is `0`. Guard the input.
- `QdrantStore.search` reads `h.payload["doc_id"]`, `["filename"]`, `["text"]`,
  `["chunk_index"]` with `[]` (KeyError on a missing key) while the optional
  fields use `.get()`. Make the required reads `.get(...)` with a sane default
  or validate at upsert time so an older/mismatched point can't crash search.

---

## 8. Consolidate the two floor-application sites

The rerank floor is applied twice: once inside `Search.find` (returns only
`kept`) and again in `AgenticSearch._apply_floors` over the fused set. It is
idempotent today, but two owners of the same policy invites drift (e.g. someone
changes one `clears_floor` call and not the other). Pick one place — ideally
`Search.find` for single-shot, with the agentic layer only fusing/deduplicating.

---

## 9. Add regression tests for the two sharpest edges

- `generation/prompts.py::_table_to_sentences` with a repeated separator row
  (currently hangs — see bugs_found.md #2).
- `ingestion/pipeline.py::ingest` where `embed()` raises *after*
  `delete_by_doc()` — assert the old chunks survive or the doc is explicitly
  marked "index empty" (see bugs_found.md #1).

The test suite is otherwise broad (27 test modules covering the parser,
chunkers, agentic flow, guards, store, worker) — these two gaps are exactly the
failure modes the current coverage misses.
