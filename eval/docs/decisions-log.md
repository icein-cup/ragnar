# RAGnar Findings Log

Running log of discoveries, patterns, and decisions. Updated each iteration.

## 2026-08-25 — Tuning runbook fixed: harness bugs, phase order, and agentic grid cut from 16 combos to 5

Reviewing `eval/docs/tuning-runbook.md` against `eval/tune_params.py` and prior eval results before
running any of the four tuning phases turned up problems serious enough that the retrieval and
agentic phases had never actually run:

- `tune_params.py` never passed `--golden` to `run_eval.py`, so any `--collection hybridqa` sweep
  silently scored the wrong golden set (`eval/golden_set.yaml`, the default) against the hybridqa
  collection. Fixed: `--golden` threaded through `_run_eval`/`_run_grid`/`main`.
- `_run_eval` parsed the *entire* stdout of `run_eval.py` as JSON, but `run_eval.py` prints the
  report object and then a trailing `wrote eval/reports/<stamp>.json` line — `json.loads` raised
  `Extra data` on that, killing the whole sweep at combo 1. Fixed: parse only up to the report's
  closing brace, with a `try/except` so one bad combo degrades instead of aborting the sweep.
- Deleted the dead `--phase floors` path (`_run_floors`, `FLOOR_GRID`) — Phase 3 (floors) has always
  called `eval/replay_floor.py --grid` directly; the `tune_params.py` copy was never invoked and
  parsed replay output by column position, which would silently break on any header change.

**Phase order was wrong.** The runbook called the four phases independent. They are not:
`score_floor`/`vector_floor` filter the score population that `candidates`/`top_k` produce, so a
floor result measured before retrieval is tuned is invalidated the moment retrieval changes.
Reordered to retrieval → agentic → floors → chunking; floors last because they're the cheapest to
redo.

**Agentic grid cut from 16 combos (~20h) to 5 new runs (~4.5h).** Two results already in this repo,
never stated together: the current default `max_hops=3, multi_query_count=3` measures p90 **53.5s**
against an agreed p90 target of 15-20s (`eval/reports/20260824-132940.json`, see
experiment-results.md "Latency reality"), and `multi_query_count=5` was separately measured at
**+47s/case** and reverted (`256c051`). Together those disqualify `multi_query_count` 5/7 (latency)
and `max_hops=4` (baseline already 3x over budget) — 12 of the planned 16 combos were never viable to
ship regardless of coverage. Also reframes the phase: the open question is how far *down*
`max_hops`/`multi_query_count` can go before coverage breaks, not how much further up buys — more is
already unaffordable. See experiment-results.md Phase 2 for the full reasoning and the disqualified-
combo grid.

**Retrieval grid narrowed:** dropped `top_k=12` from the sweep — adds context to a generation call
already over the latency budget above, and Branch D's own recall numbers already flatten past
`top_k=10`. Both phases now use coordinate descent (hold one axis at default, sweep the other, sweep
the winner) instead of the full grid: ~11 runs total across both phases, ~10h serial, down from ~35h.

**Parallel execution ruled out for the agentic phase, viable but not recommended for retrieval.**
The agentic phase's whole output is a latency measurement; sharding combos across one Ollama makes
per-request latency a function of contention rather than the config, which would distort the ranking
(contention scales with call count, not just rescale it by a constant) — not just add noise. Retrieval
could shard safely (`answer_coverage` doesn't move under contention) after fixing two harness bugs —
`run_cases`' `llm.unload()` is a *global* Ollama eviction, not per-client, and the report stamp is
second-resolution so concurrent shards collide on one output file — but coordinate descent already
gets retrieval down to ~6-7 runs, making the fix-and-shard path not worth it.

**Added a RAGAS finalist check.** `decisions-log`'s own "Eval comparison tracking" table already
listed `faithfulness`/`answer_correctness`/`context_precision` as pending, but the tuning runbook
never called `eval/run_ragas.py` anywhere — a phase could pick a winner on `answer_coverage` alone
while faithfulness quietly dropped, since `answer_coverage` only checks whether expected words appear
in the answer. Running RAGAS's judged, paid metrics on the full ~11-run sweep isn't worth the cost;
instead each phase's finalist (not every grid row) gets one RAGAS pass, recorded alongside the
deterministic metrics, with a faithfulness regression against the 0.70 baseline treated as a reason to
keep the previous default even if `answer_coverage` improved.

**Correction, same day:** the RAGAS finalist check as first written had two real problems, caught in
review before any run happened. `run_ragas.py:86` called `run_cases(...)` directly — it does not read
a saved report — so "score the finalist with RAGAS" was actually a *second* full ~55-min pipeline run
per phase, not the cheap post-process the runbook implied. And `run_ragas.py` had no floor flags, so
that second run always used whatever `config.yaml` held at the time — meaning the Phase 1/2 finalist
checks would have been scored against the untuned 0.55/0.42 stopgap, since floors aren't tuned until
Phase 3. Fixed by adding `--report` to `run_ragas.py`: `build_samples()` now accepts a saved report
path and reads its `question`/`answer`/`expected_answer`/`contexts` fields directly (already saved by
`run_eval.py` on every run) instead of calling `run_cases()` again. This makes the finalist check a
true post-process — no extra pipeline run, and it's inherently scored against whatever config produced
that report, so the floor-mismatch problem doesn't arise. Also added `0.42` (the shipped
`vector_floor` default) to `replay_floor.py`'s `VECTOR_GRID`, which previously only bracketed it —
`0.55/0.42` is now a directly comparable row in the Phase 3 grid, not an interpolation. Also corrected
the runbook's Phase 3 header, which quoted "~55 min" for what is actually two full runs (uncensored +
confirm) — see the total-budget table now at the top of `tuning-runbook.md`.

`eval/run_eval.py`, `eval/metrics.py`, and `config.yaml` are still untouched — this remains a
harness-and-runbook fix, not a new measurement. `eval/run_ragas.py` and `eval/replay_floor.py` did
change (see above). See [`eval/docs/tuning-runbook.md`](tuning-runbook.md) for the corrected procedure
and [`eval/docs/experiment-results.md`](experiment-results.md) for the updated phase tables.

## 2026-08-25 — Systematic hyperparameter tuning begins

**Problem:** every retrieval/chunking/agentic default is an untested stopgap:
- `score_floor` 0.55 / `vector_floor` 0.42 — calibrated on a broken sweep (vector_floor left at 0.0,
  so the OR'd floors passed everything), never re-derived. `config.yaml` flags this itself.
- `target_tokens` 350 / `overlap_tokens` 50 / `table_rows_per_group` 10 — cavecrew recommendations,
  never measured.
- `max_hops` 3 / `multi_query_count` 3 — arbitrary defaults.

**Plan:** four phases, cheapest first. Full execution instructions and runtime budget live in
[`eval/docs/tuning-runbook.md`](tuning-runbook.md); results go in `eval/docs/experiment-results.md`.

**Tooling added:** `eval/replay_floor.py` (offline floor replay, refuses censored reports),
`eval/tune_params.py` (orchestrator, ranks by `answer_coverage` subject to `refusal_accuracy >= 0.85`).

**Key correction to the first draft of this plan:** the floor sweep was originally proposed as 25
full `--agentic` runs (~23 hours). `calibrate_floor()`'s own docstring already documents the offline
route — one run at floors 0.0 saves every score, and any floor is then arithmetic over that report.
`replay_floor.py` implements it. Also dropped: a made-up composite score (no precedent in the
codebase — `answer_coverage` is primary, `refusal_accuracy` the constraint), and a re-test of
semantic chunking (already measured and closed as a 1-point wash).

**Risks:**
- Golden-set quality: if probes are too easy/hard, `refusal_accuracy` is meaningless.
- Interaction effects: optimal floors may depend on chunking strategy (deferred to Phase 4).
- Runtime: Phases 2-3 are ~15h serial each; the runbook documents a parallel path (~7-8h).

**Success criteria:**
- Documented floor heatmap (coverage vs refusal) in `eval/docs/experiment-results.md`.
- A defensible production `config.yaml` recommendation.
- Parameters identified as negligible (freeze their defaults).

**Next:** run Phase 1 (uncensored run + `replay_floor.py --grid`), log results in experiment-results.md.

## 2026-08-24 — Baseline analysis (125-case HybridQA, qwen2.5:7b, floors 0.0)

### Failure category breakdown (90 in-corpus cases)

| Category | Count | % | Root cause |
|---|---|---|---|
| CITATION_NOISE | 39 | 43% | Multi-query/multi-hop fusion piles 5-16 chunks, all cited |
| RETRIEVAL_MISS | 22 | 24% | Expected source not retrieved at all |
| OVER_REFUSAL | 16 | 18% | Model says NO_ANSWER when answer IS in contexts |
| WRONG_ANSWER | 8 | 9% | Model answers but misreads retrieved text |
| MULTI_HOP_MISS | 4 | 4% | Answer correct but not all sources cited |
| CORRECT | 1 | 1% | Fully correct (answer + citations) |

### Out-of-corpus
- 33/35 correctly refused
- 2 hallucinated: Canada Basin depth (fabricated "13,000 ft"), Danish supermarket (picked "Kiwi" from irrelevant contexts)

### Key patterns
- 43/44 correct answers have citation precision < 0.5 — citation noise is universal
- 88/90 in-corpus cases are multi-hop; failure rate 98.9%
- Population questions fail across all categories
- Model often "thinks out loud" instead of stating answer directly (e.g. "To answer this, we need to identify...")

### Score distribution (743 chunks, floors at 0.0)
- Expected chunks: rerank median 0.678, vector median 0.522
- Unexpected chunks: rerank median 0.513, vector median 0.444
- OR semantics: best floor combo filters only 11% of noise
- Conclusion: floors cannot fix citation noise. Answer-overlap pruning is the correct approach.

## 2026-08-24 — Iteration 1 changes (commit da922ea)

### Track A: Answer prompt (generation/prompts.py)
- Added 4-step reasoning process: identify question → check every excerpt → synthesize across excerpts → decide
- Stronger anti-refusal: "Refusing is the last resort, not the first"
- Explicit multi-hop synthesis guidance
- NO_ANSWER sentinel contract preserved
- Expected impact: reduce over-refusal (16→?), improve wrong-answer rate (8→?)

### Track B: Citation pruning (generation/answerer.py)
- Answer-overlap filtering: only cite chunks with ≥2 content words overlapping answer
- Stopword-filtered tokenization (English only — Polish stopwords NOT included, potential gap)
- Safe default: short answers skip pruning entirely
- Applied in answer(), answer_async(), ground_async(), eval harness
- Expected impact: citation precision 0.26→? (should improve significantly), citation_accuracy must not drop

### Track C: Floor calibration
- Floors at 0.0/0.42 with OR semantics only filter 11% of noise
- Reranker neutral scores (0.5 for 44% of needed chunks) make threshold-based pruning ineffective
- No code change — conclusion documented

### Pending verification
- ~~Baseline eval with real floors (0.55/0.42) running — ~50 min ETA~~ **DIED** — partial JSONL (9/125 lines) found by watchdog
- Re-eval with new prompt + pruning needed after baseline completes
- RAGAS on full 125-case set (previous was only 20 cases)
- ~~Code review subagent reviewing diffs~~ **DONE** (commit 49234ad)
- ~~RAG consultant subagent proposing next improvements~~ **DONE** (recommendations logged above)

## 2026-08-24 — Watchdog recovery (15:41 CEST)

**Diagnosis**: Session STALLED. No background processes, no subagents, no file changes for ~10 min. Two partial eval JSONL files found:
- `20260824-131803.jsonl` (6/125 lines) — first eval attempt, died early
- `20260824-132844.jsonl` (9/125 lines) — second eval attempt, also died

Both were eval re-runs with all Iter 1+2 changes on the rebuilt Docker image. Neither completed.

**Git state**: `iteration_ai` branch, clean working tree, last commit `126904c` (15:31). All code changes committed.

**Recovery action**: Restarted eval as background process `proc_a3ab8201e728`:
`docker compose exec app python eval/run_eval.py --collection hybridqa --golden eval/golden_hybridqa_draft.yaml --agentic`
Using config.yaml floors (0.55/0.42). Model weights loaded, processing started. ETA ~55 min.

**Next steps after eval completes**:
1. Parse results JSON, compute metrics
2. Compare against baseline (answer_coverage 0.49, citation_precision 0.26, etc.)
3. Update decisions-log.md with results
4. If scores improved, consider running RAGAS on full 125-case set
5. If not improved, dispatch subagents for next iteration tracks

### Open questions
- Will the 4-step reasoning in SYSTEM_PROMPT cause the model to output reasoning steps in its answer? (qwen2.5:7b may not follow "work through steps internally")
- Will citation pruning hurt multi_hop_citation_accuracy? (If answer doesn't contain words from a needed source's chunk, that source gets pruned)
- Should Polish stopwords be added to the pruning filter?
- Is the 20-case RAGAS run too small to be meaningful? (Yes — need full 125-case run)

## 2026-08-24 — RAG consultant recommendations (subagent sa-0-ac6fde77)

5 improvements ranked by expected impact:

1. **Table-to-text serialization** (generation/prompts.py, format_excerpts) — convert markdown tables to "Entity | Property | Value" sentences before sending to LLM. 7B models read NL far better than markdown tables. Zero extra LLM calls. Expected: answer_accuracy 0.71→0.80+, answer_correctness 0.33→0.45+
2. **Reduce top_k to 3 + score-gap pruning** (config.yaml, retrieval/agentic.py _fuse_results) — fewer chunks = less confusion for 7B model. Expected: citation_precision 0.26→0.40+, context_precision 0.58→0.70+
3. **Two-stage extract-then-synthesize** (generation/prompts.py + answerer.py) — separate fact extraction from reasoning. +1.5s latency. Expected: answer_accuracy 0.71→0.82+, faithfulness 0.70→0.82+
4. **Retrieval confidence signal** (prompts.py build_user_prompt) — tell model "retrieval found relevant content" to reduce over-refusal. Expected: answer_coverage 0.49→0.56+, refusal_accuracy 0.76→0.82+
5. **Expand RAGAS to 60+ cases + multi-hop metrics** (eval/run_ragas.py, eval/metrics.py) — 20 cases = 5 pts each, statistically inadequate

Recommended order: #5 → #1 → #2 → #4 → #3

## 2026-08-24 — Code review fixes (commit 49234ad)

Addressed 4 MEDIUM findings from reviewer (sa-0-c30598f1):
1. **Reasoning step leakage** — added "do not show them in your output" + "Output ONLY the final answer" to SYSTEM_PROMPT
2. **Polish stopwords missing** — added 30+ Polish function words to _STOPWORDS
3. **Polish diacritics mangled** — switched regex from `[a-z0-9]+` to `\w+` with re.UNICODE
4. **Docstring inaccuracy** — fixed "minus" to "plus stopword filtering"

Tests: 290 passed, 19 skipped.

## 2026-08-24 — Iteration 2 changes (commits 758b083, af5e64a)

### Table-to-text serialization (generation/prompts.py)
- _table_to_sentences() converts markdown table rows to "Header: Value" sentences at prompt-build time
- Only converts lines starting with |, skips separator rows
- Non-table text passes through unchanged
- Expected: answer_accuracy 0.71→0.80+, answer_correctness 0.33→0.45+

### Score-gap pruning (retrieval/agentic.py)
- _fuse_results() now truncates to top-3 when >4 results AND gap >0.05 between 3rd and 4th
- Small result sets never cut
- 4 new tests added
- Expected: citation_precision 0.26→0.40+, context_precision 0.58→0.70+

### Eval re-run started
- Running with all Iter 1+2 changes on rebuilt Docker image
- proc_4d8d878b3aa9, ~55 min ETA

## 2026-08-24 — Cavecrew-reviewer Iter 2 findings + fixes (commit 2b45fb7)

3 critical, 3 medium, 1 minor found. All fixed:

**_table_to_sentences (3 critical):**
1. No-header tables lost first data row → fixed: require header+separator pattern
2. Multi-table excerpts corrupted (headers never reset) → fixed: reset headers on non-table lines
3. Pipe-prefixed non-table text (math notation) silently dropped → fixed: require separator row to activate table mode

**Score-gap pruning (1 medium):**
4. `len > 4` too aggressive for 5-result multi-hop sets → raised to `len >= 6`

Tests: 290 passed, 19 skipped.

## 2026-08-24 — Iteration 3 changes (commits pending)

### Iter 3A — Retrieval confidence signal (in progress)
- build_user_prompt() gets optional retrieval_confidence param
- "high" if top score > 0.6, "medium" if > 0.55, else None
- Tells model to check excerpts carefully before refusing
- Expected: answer_coverage 0.49→0.56+, refusal_accuracy 0.76→0.82+

### Iter 3B — Task-specific multi-hop prompt (commit 09e8add)
- MULTI_HOP_SYSTEM_PROMPT replaced with table→passage→answer guidance
- Identifies entity in table row, generates entity-based follow-up query
- Output format preserved (Sufficient/Missing/FollowUp)
- Expected: multi_hop_citation_accuracy 0.52→0.62+

### Iter 3C — Hyperparameter tuning (commit 771cc85)

Consultant found critical issue: Ollama `num_ctx` not set → defaults to 4096 → silently truncates SYSTEM_PROMPT. Likely root cause of 16/90 over-refusal.

Changes applied:
| Parameter | Old | New | Reason |
|---|---|---|---|
| num_ctx (llm.py) | 4096 (default) | 32768 | System prompt was being truncated |
| target_tokens | 500 | 350 | Tighter chunks, better reranker precision |
| table_rows_per_group | 20 | 10 | Less dilution per table chunk |
| top_k | 5 | 3 | Align base search with gap-pruned path |
| max_hops | 3 | 2 | Reduces spurious follow-up noise |
| multi_query_count | 3 | 5 | More retrieval coverage (22 RETRIEVAL_MISS) |
| FAST_PATH_MIN_RESULTS | 3 | 4 | Self-correction gets more chances |

Kept unchanged (already optimal):
- Answer temperature: 0.0 (deterministic)
- query_temperature: 0.7 (lexical variety for multi-query)
- candidates: 25 (appropriate for reranker)

Expected impact: faithfulness ↑↑ (num_ctx fix), context_precision ↑↑ (tighter chunks + top_k=3), answer_coverage ↑ (more multi-query coverage), refusal_accuracy ↑↑ (system prompt visible)

### Eval comparison tracking

| Metric | Baseline (floors 0.0) | 20-case quick | Full 125-case |
|---|---|---|---|
| answer_coverage | 0.49 | running | pending |
| answer_accuracy | 0.71 | running | pending |
| refusal_accuracy | 0.76 | running | pending |
| citation_accuracy | 0.67 | running | pending |
| citation_precision | 0.26 | running | pending |
| multi_hop_citation_accuracy | 0.52 | running | pending |
| RAGAS faithfulness | 0.70 | — | pending |
| RAGAS answer_correctness | 0.33 | — | pending |
| RAGAS context_precision | 0.58 | — | pending |

## 2026-08-24 — Cavecrew-reviewer Iter 3 (clean)

290 passed, 19 skipped. No critical, no medium. 2 INFO:
- num_ctx hardcoded (works for qwen2.5:7b, could be configurable later)
- FAST_PATH_MIN_RESULTS=4 + max_hops=2 is a design tradeoff, not a bug

All Iter 3 code approved. Ready for eval.