# Bugs Found — RAGnar (branch `iteration_ai`) — 2026-08-25 (second pass)

Second review pass. The first pass (`bugs_found.md` above, entries 1–5) is now
all fixed in the current tree: the ingest pipeline is delete-after-store,
`_table_to_sentences` advances `i`, the grounding verdict is scoped by
`current_chat_id`, the OCR trigger is gated on `page_count > 1`, and the two
junk files are gone. This file records what that pass missed.

Severity: **High** / **Medium** / **Low**. Confidence: **confirmed**
(reproduced against the real code path) vs **conditional** (needs a specific
input/state).

---

## 1. [FALSE POSITIVE] Agentic fast-path ignores the per-request similarity floor

- **File:** `retrieval/agentic.py:316-353` (`_is_fast_path`), reached from
  `find()` at `retrieval/agentic.py:223-230` (and again for the fused set).
- **Status:** not a bug. The fast-path branch (`retrieval/agentic.py:244-250`)
  ends in `self._apply_floors(...)`, which does apply the caller's
  `score_floor`/`vector_floor` — it was never skipped. `git blame` puts that
  call at commit `583ccde` (2026-08-07), predating this review.

  The original repro leaked weak results only because the fake `Search` used
  to reproduce it (`StubSearch` in `tests/test_agentic.py`) doesn't behave
  like the real one: `StubSearch.reranker = None` makes `_apply_floors` a
  no-op (it only filters `if use_reranker and self._base_search.reranker is
  not None`), and `StubSearch.find` ignores its own `score_floor`/
  `vector_floor` arguments. The real `Search` built in
  `ui/services.py:59-63` always carries a `BGEReranker`.

  Verified against the real `Search` (fake embedder/store/reranker, 4
  chunks, rerank score 0.60, cosine 0.60):

  ```
  score_floor=0.5 vector_floor=0.42: refused=False n=4 fast_path=True
  score_floor=0.9 vector_floor=0.42: refused=False n=4 fast_path=False
  score_floor=0.9 vector_floor=0.99: refused=True  n=0 fast_path=False
  ```

  Row 2 is what the report reads as the bug; it's `clears_floor`'s
  documented OR rule (`retrieval/search.py:8-17`) — the vector floor rescues
  it. Row 3 shows raising both floors does refuse correctly.

  Original text below, kept for the record:

`AgenticSearch.find()` threads a per-call `score_floor` / `vector_floor`
down into every `self._base_search.find(...)` call, and `_apply_floors()`
(used on the multi-hop and self-correction paths) correctly honors that
per-call value:

```python
# _apply_floors, retrieval/agentic.py:371-372
floor  = self._base_search.score_floor if score_floor is None else score_floor
vfloor = self._base_search.vector_floor if vector_floor is None else vector_floor
```

But the fast-path gate does **not**:

```python
# _is_fast_path, retrieval/agentic.py:339-340
floor = (self._base_search.score_floor if score_floor is None
         else score_floor)
if floor <= 0:
    floor = FALLBACK_SCORE_FLOOR          # 0.55
```

The `<= 0` fallback exists to stop a *disabled* floor (0) from declaring
everything "strong". It is applied unconditionally, so it also overrides a
**legitimate strict** per-request floor — and more importantly, the fast path
skips `_apply_floors` entirely, so the caller's strict floor is never applied
to the returned result set on that branch.

**Reproduction** (fake `Search` whose `find` honors the caller's floor, fake
LLM; 4 candidate chunks each scoring 0.6):

```
FAST-PATH branch (multi_query OFF):
  caller score_floor=0.5: ANSWERED (4 chunks), fast_path=True   [want ANSWERED]
  caller score_floor=0.9: ANSWERED (4 chunks), fast_path=False  [want REFUSED]  <-- BUG
MULTI-QUERY branch (multi_query ON):
  caller score_floor=0.9: ANSWERED (4 chunks), fast_path=False  [want REFUSED]  <-- BUG
```

**Reachability:** the UI passes a per-request floor on every query
(`ui/app.py:331` — `score_floor=query["floor"]`), and that value comes
straight from the "Similarity floor" slider (`ui/panels/settings.py`,
`min_value=0.0`). So a user who cranks the floor up to make refusals stricter
silently gets weak results anyway whenever the fast path fires — which is the
common case (3+ decent chunks on the base search). The floor slider, the
app's primary "answer more / refuse more" knob, is effectively a no-op on the
fast path. The multi-query branch is also affected: the fast-path check runs
before `_apply_floors`, so a strict caller floor that the fused set would have
refused on is bypassed.

Note the non-agentic path is correct: `Search.find()` itself applies the
floor, so only the agentic layer (the default, since `agentic_search` is
always built in `ui/services.py:59`) has the gap.

**Suggested fix:** pass the per-call `score_floor` / `vector_floor` through to
`_is_fast_path`'s decision *and* make the `<= 0` guard only substitute the
fallback when the caller explicitly disabled the floor (floor is `None` or
`<= 0`), not when the caller passed a real positive value. Simplest: have the
fast-path gate re-run the exact same `clears_floor` test that `_apply_floors`
uses, on the caller's resolved floors.

---

## 2. [Medium] Lazy cross-encoder load is not thread-safe on a shared instance

- **File:** `retrieval/reranker.py:20-24` (`_ensure_model`).
- **Confidence:** conditional (requires two threads first-touching the model;
  a real memory spike, not a crash).

```python
def _ensure_model(self):
    if self._model is None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self._model_name)
    return self._model
```

`BGEReranker` is built once in `build_services()` (`ui/services.py:57`,
`@st.cache_resource`) and shared across all Streamlit sessions, and
`AgenticSearch._retrieve_multi_query` fans retrieval out across a
`ThreadPoolExecutor` (`retrieval/agentic.py`, `MAX_PARALLEL_QUERIES`). If two
sessions (or two multi-query workers) hit a cold reranker at the same time,
the check-then-construct is not atomic and the multi-GB `CrossEncoder` can be
constructed twice. No lock, no `threading.Lock` guard, and no eager
construction in `build_services()`. This is already called out as a known
refactor item, but it is a live race on a shared mutable object, not merely a
style point.

**Suggested fix:** construct the reranker eagerly in `build_services()`, or
guard `_ensure_model` with a lock (double-checked locking). Same applies to
`DoclingParser`'s thread-local converters — that one is *correctly*
thread-local, so it's the model of what the reranker should do.

---

## 3. [Medium] `auto_worker_count` divides by a per-worker RAM budget that can be 0

- **File:** `core/config.py:54` (`auto_worker_count`).
- **Confidence:** conditional (needs `worker_memory_gb: 0` in `config.yaml`).

```python
mem_based = int(mem_gb // worker_memory_gb)
```

`worker_memory_gb` comes straight from `config.yaml`
(`ing.get("worker_memory_gb", 2.0)`, `core/config.py:157`) with no
guard. A `0` (a plausible typo for "unlimited") raises `ZeroDivisionError`
inside `Config.worker_count`, which is read in `build_services()` at startup
(`ui/services.py:56`) — so the whole app fails to boot rather than
falling back to the CPU-based count. The default is 2.0, so this only bites
on a hand-edited config, hence conditional.

**Suggested fix:** `mem_based = max_workers if not worker_memory_gb else int(mem_gb // worker_memory_gb)`.

---

## 4. [Low] `delete_stale` scroll is capped at 10000 points and not paginated

- **File:** `retrieval/store.py:140-160` (`delete_stale`).
- **Confidence:** conditional (needs a document whose re-chunk changes enough
  chunk indices that >10000 stale points remain).

```python
points, _next = self._client.scroll(
    collection_name=self.collection,
    scroll_filter=Filter(must=[... doc_id ...]),
    limit=10000,
    with_payload=True,
)
stale_ids = [p.id for p in points if p.payload.get("chunk_index") not in keep_indices]
```

The `_next` page token is discarded — only the first page of up to 10000
points for that `doc_id` is ever scanned, so a document with more than 10000
old chunk indices can leave stale points behind after a re-chunk. In practice
a 10000-chunk document is large enough that this won't be hit soon, but it is
an unbounded-by-design pagination that silently stops at 10k.

**Suggested fix:** loop on `_next` until it is `None`.

---
