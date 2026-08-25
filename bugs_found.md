# Bugs Found — RAGnar (branch `iteration_ai`)

Findings from a manual review of the Python source, 2026-08-25.
Severity: **High** / **Medium** / **Low**. Confidence: **confirmed** (reproduced or read
directly from the code path) vs **conditional** (requires a specific input/state).

No code was changed. Line numbers refer to the current checkout.

---

## 1. [High] Ingest pipeline wipes a document's search index on transient failure

- **File:** `ingestion/pipeline.py:59-66`
- **Confidence:** confirmed (code order)

```python
# Replace wholesale so stale and fresh chunks never coexist.
self._store.delete_by_doc(doc_id)

t3 = t4 = t2
if chunks:
    vectors = self._embedder.embed([c.text for c in chunks])
    ...
    self._store.upsert(chunks, vectors)
```

`delete_by_doc()` runs **before** `embed()` and `upsert()`. If either of those
throws — Ollama is briefly down, Qdrant is unreachable, the embedder returns a
mismatched vector count — the worker catches it in `worker.py:57` and calls
`mark_failed()`, but the document's old chunks are already gone from Qdrant.

**Result:** the document silently vanishes from search until a manual retry
succeeds. A one-second Ollama hiccup during a re-chunk turns a previously
answerable document into "no results" with nothing telling the user why.

**Suggested fix:** make the replace atomic. Embed first, upsert to the new
vector set, and only then delete stale chunks — or delete-and-rebuild inside a
guard that restores/re-queues on failure. At minimum, if the build fails, leave
the old chunks in place and surface a clear "index not updated" state rather
than an empty one.

---

## 2. [High] Infinite loop in `_table_to_sentences` on a repeated separator row

- **File:** `generation/prompts.py:233-235`
- **Confidence:** confirmed (reproduced — the function never terminates)

```python
# In a table — skip separator rows.
if _is_separator(cells):
    continue          # <-- i is never incremented
```

Inside the "already in a table" branch, a line that classifies as a separator
(`|---|---|`) hits `continue` without advancing `i`, so the loop re-reads the
same line forever. Confirmed: feeding a table whose body contains a second
separator row (`header, sep, row, sep, row`) hangs the process.

**Reachability:** current chunking emits exactly one separator per table chunk,
so the normal path never triggers this. But `_table_to_sentences` runs on every
excerpt of every query — a single malformed table (repeated header row from a
Docling export, a hand-edited markdown file) would hang the *entire answer
path*, not just one document. There is no guard and no timeout.

**Suggested fix:** add `i += 1` before the `continue`, and add a regression
test that feeds a repeated-separator table.

---

## 3. [Medium] Grounding verdict can be written into the wrong chat

- **File:** `ui/app.py:237-257` (fold-in) and `ui/app.py:457-459` (registration)
- **Confidence:** conditional (requires a chat switch while a check is in flight)

`pending_grounding` is a dict keyed by **positional message index**
(`_msg_idx = len(messages) - 1`). The grounding check runs on a daemon thread
(`answerer.ground_async`) and its verdict is folded in on a *later* rerun:

```python
if _idx < len(st.session_state.messages):
    st.session_state.messages[_idx]["grounded"] = _answer.grounded
```

If the user opens another chat (or starts a new one) between the answer and the
verdict landing, `st.session_state.messages` now belongs to a *different* chat,
and the stale index writes the old answer's verdict onto an unrelated message —
then persists it via `chats.save(current_chat_id, ...)`.

**Suggested fix:** scope the pending entry by `current_chat_id` (key on
`(chat_id, msg_idx)` or store the target message's id) and only fold the
verdict in when the chat id still matches.

---

## 4. [Low] OCR fallback triggers on legitimately short documents

- **File:** `ingestion/parser.py:185`
- **Confidence:** conditional (short but valid PDFs)

```python
if parsed.chars_per_page < OCR_TRIGGER_CHARS_PER_PAGE or not parsed.blocks:
```

`OCR_TRIGGER_CHARS_PER_PAGE = 50`. A genuinely short page (a cover sheet, a
one-line memo, a title page) falls under 50 chars/page and gets re-parsed with
OCR — slow, may pull OCR model weights, and sets `low_confidence = True` on a
document that was extracted correctly.

**Suggested fix:** prefer "no text blocks" as the OCR trigger, or gate the
density check on `page_count`/expected length rather than a flat 50-char floor.

---

## 5. [Low] Stray repo junk files

- **Files:** `import` (root), `IDEA.md` (root)
- **Confidence:** confirmed

`import` contains a leftover Python traceback (`KeyboardInterrupt` from a
`/tmp/async_test.py`); `IDEA.md` is a one-line placeholder ("a local RAG
system"). Neither is imported or referenced. Cleanup only.
