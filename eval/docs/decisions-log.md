# RAGnar Findings Log

Running log of discoveries, patterns, and decisions. Updated each iteration.

## 2026-08-26 — Latency target is max ≤35s, a hard ceiling — corrected same day

**Correcting the entry below, which recorded this as "p90 ~35s".** The owner
meant **max ≤35s: no case may exceed it.** That is a materially stricter
constraint and it reverses the conclusion drawn from the looser reading.
The entry below is kept as written, with its own correction note, because the
error is instructive — the numbers it cites are right and its conclusion is
wrong, purely on which statistic the target names.

**The pipeline does not meet this target.**

| Run | over 35s | max |
|---|---|---|
| `20260825-165051`, 125 cases, pre-move | **30/125 (24%)** | 182.7s — 5.2x over |
| `20260826-110031`, 20 cases, post-move | **1/20 (5%)** | 84.2s — 2.4x over |

Every violator in both runs is a high-fan-out case (7-13 generated queries),
and most are out-of-corpus: the search keeps generating queries hunting for
material that does not exist, paying a full rerank per query. **The tail is
the binding metric, not the middle.**

**Three consequences:**

1. **Rank on `max_s`, not `latency_p90`.** Every report already records it
   (`summary.latency.max_s`); this file's comparisons have been ranking on
   mean and p90, which say nothing about a ceiling. A config with a better
   p90 and a worse max is a regression under this target.
2. **`mq` 5 and 7 stay dead, and the pruning is re-justified on new
   grounds.** The +47s/case figure remains stale (measured on the container
   CPU reranker), but more generated queries lengthen exactly the tail this
   target binds on. The conclusion survives its original argument's death.
3. **Phase 2's inverted framing is correct after all** — "how far *down* can
   these parameters go before coverage breaks." The entry below un-inverted
   it; that was wrong and is reverted.

**The sweep cannot deliver this target, and should not be expected to.** A
parameter sweep shifts a distribution; it does not bound a tail.
`retrieval/agentic.py` has no deadline, timeout, or time budget of any kind —
verified, nothing in the code can enforce a ceiling. The best any grid cell
can do is make violations rarer. A guaranteed max needs a wall-clock check in
the hop (`_multi_hop`, line 488) and fan-out (`_retrieve_multi_query`, line
422) loops that stops and answers with what has been retrieved so far. That
is an implementation task, not a tuning one, and it is **not** currently in
the sweep plan.

**Pattern, fourth instance this week.** A number was recorded in a form that
was not what was meant (p90 vs max), and a chain of reasoning was built on it
within one turn. Cheap to catch here because the owner restated it
immediately. Same family as the three below: check what a foundational number
actually says before deriving from it.

## 2026-08-26 — Latency target relaxed to p90 ~35s; pruned grid cells stay pruned by choice

**Superseded within hours — see the entry above.** The target is max ≤35s,
not p90 ~35s, and under that reading the central claim of this entry is
false: latency is still binding, on the tail. The measurements cited below
are accurate; only the conclusion drawn from them is wrong. Kept for the
record.

Owner's call, superseding the p50 6-8s / p90 15-20s target agreed 2026-08-24.
The post-move p90 of 34.24s clears ~35s, so **latency stops being the binding
constraint on tuning** and becomes a reported number rather than a veto.

**This retroactively invalidated the reasoning behind 10 of 16 pruned Phase 2
cells — the conclusions were kept, the justifications were not.** Both
disqualifying arguments had expired:

- `multi_query_count` 5 and 7 were killed on a **+47s/case** measurement
  (commit `256c051`). That was measured with the reranker on container CPU,
  and `mq` is a direct multiplier on rerank count — a rerank went 10.34s →
  1.48s the same week. The penalty is stale by construction.
- `max_hops=4` was killed for missing a p90 target that no longer exists.

Asked explicitly and answered: **keep the six-combo grid, keep `top_k=12`
dropped.** The headroom goes to finishing the sweep, not re-deriving pruned
cells. Reopening `mq=5` across `max_hops` 1-3 is 9 combos / ~6h; the full
16-combo grid is ~10h for Phase 2 alone. `top_k=12` stays out on Branch D's
coverage-flattening past `top_k=10`, which was always the load-bearing reason
— latency was only ever the secondary one.

**Phase 2's framing reverts.** While the baseline sat 3x over budget the
phase had been reframed as "how far *down* can these parameters go before
coverage breaks", with latency as its objective. That inversion is off: the
six open combos are a genuine coverage-vs-latency tradeoff again, and a
config costing seconds for real coverage is allowed to win.

**Two caveats recorded alongside the new target, neither resolved by it.**
p90 is not a ceiling — the slowest case in the measured run was **84.2s**, an
out-of-corpus question where the fan-out keeps hunting for material that does
not exist. And the pipeline still runs inside a static `st.spinner()`
([ui/app.py:303](ui/app.py#L303)) with zero visible output, so ~35s p90 in
practice means occasional minute-plus stalls with nothing on screen. The UX
problem is untouched by the target change.

**Pattern worth naming, third instance this week.** A conclusion outlived the
measurement it rested on, and would have been carried into the sweep unexamined
had the target not been questioned — same shape as the thread-oversubscription
retraction and the projected-vs-measured budget row. When a foundational number
moves (a 7x speedup, a relaxed target), the decisions derived from it need
re-checking, not just the number itself.

## 2026-08-26 — Smoke test passes; the sweep budget was wrong twice and is now measured

First completed run with the host reranker actually serving. 20 cases,
agentic, `candidates=30/top_k=10`, `golden_subset20.yaml`, report
`20260826-110031`. Wiring check only — subset20 numbers never inform a
tuning decision (see the 2026-08-25 subset false-signal entry).

**Harness paths confirmed live, all four:**
- provenance stub written *before* case 1 (verified mid-run, not after)
- full report written with 20 cases and a `latency` block
- `report_table.py` picks the run up as a ledger row
- `run_ragas.py --report` scores a saved report — 80 judgements
  (10 scorable cases × 8 metrics), exit 0, judge reachable

**Budget: ~10-11h, not ~28h.** Measured 18.0s/case, 4.05 queries/case →
~37min per 125-case run × ~15 runs. The runbook has now carried three
different figures and this is the first with a completed run behind it:

| Claim | Per run | Total | Basis |
|---|---|---|---|
| original | ~55min | ~15h | never measured |
| 2026-08-26 revision | ~1.8h | ~28h | in-container half measured; **host half projected** |
| now | ~37min | ~10-11h | measured, host reranker serving |

The ~28h revision was the worst of the three, and it was the one that
replaced a figure for being unmeasured. Its in-container row (~2.9h) was
real; its host row (~1.8h) was an unlabelled projection off that row. The
original ~55min was closer to the truth than the correction applied to it.

**Why the 7x reranker win is only ~1.5x on wall-clock.** Reranks inside a
case's multi-query fan-out run *concurrently*, so a per-component ratio does
not multiply through. Proof from the pre-move 125-case run: 503 reranks ×
10.34s = 86.7min, which exceeds that run's entire 54.9min wall-clock. The
defensible comparison is like-for-like — 26.4s/case (pre-move, 4.02 q/case,
25 candidates) → 18.0s/case (post-move, 4.05 q/case, 30 candidates): ~1.5x
while doing *more* work per query.

**Lesson, and it is the same one as the reranker thread-oversubscription
retraction above:** a projection and a measurement must never be written into
the same table without labelling which is which. Both errors this week were
an estimate inheriting the authority of the measurement next to it. The
runbook's run-time table now marks its estimated row `(est.)` explicitly.

**Bug: every in-container report stamped `git: "unknown"`.** The app image
ships no `git` binary, so `_git_revision()`'s subprocess call returned
nothing and fell to its `except` — meaning the runbook's own pre-launch check
("confirm no `-dirty` suffix in provenance") could never fire, and no report
produced in Docker was traceable to a commit. Fixed by reading `.git/HEAD`,
loose refs, and `packed-refs` directly in Python; in-container runs now stamp
`<sha>-nogit`. Dirty detection cannot survive that fallback (it needs the
index hashing only git does), so `-nogit` marks dirtiness as *unknown* rather
than falsely clean, and the runbook now states plainly that the host-side
`git status --short` is the only real guard. 5 tests added
(`tests/test_git_revision.py`).

**RAGAS `n=3 → n=1` warning, resolved as a non-issue.** Ollama Cloud's
OpenAI-compatible endpoint ignores `n`, so `answer_relevancy` (the only
metric requesting 3, `strictness=3`) got 1. At `temperature=0` all three
generations would be identical anyway — the self-consistency averaging is
inert under a deterministic judge, and no extra call was being paid for.
Pinned `answer_relevancy.strictness = 1` to state what already happens.
Rejected the alternative (raise temperature, loop 3 separate calls): a
tuning sweep compares runs against each other, so a reproducible judge is
worth more than a variance-smoothed one.

**RAGAS scores a filtered subset — do not read it as run quality.**
`build_samples` drops out-of-corpus and refused cases, so this run's 20
became 10. The 5 in-corpus refusals — the actual coverage failures — are
invisible to every RAGAS number. `answer_coverage` from the house metrics is
the one that counts them.

## 2026-08-26 — Reranker moved to the host GPU: 7x, and a wrong diagnosis corrected on the way

A smoke run before Phase 1 came in at **~84s/case**, against this runbook's claimed
~10s/case. Chasing that produced the largest single speedup so far, and two
retractions worth recording.

**The bottleneck is the reranker, not the LLM.** During a run the app container sat at
~1390% CPU while Ollama sat at 0.1% and Qdrant at 0.18%. Ollama runs on the host and
reports `100% GPU`; the cross-encoder was doing all its work on container CPU. Per-case
time tracks `queries` almost linearly (1 query ≈ 17s, 7 queries ≈ 160s) because the
agentic path pays one **full** rerank per generated query. `reranker.py`'s own comment
claimed "1-3s for 25 candidates"; measured cost was **10.34s for 30**, 3-5x optimistic.
Comment corrected.

**Retracted: thread oversubscription.** A first benchmark suggested torch was
oversubscribing threads (4 concurrent reranks x 16 threads on 16 cores) and that
capping `torch.set_num_threads` would help. It was an artifact of test order — whichever
setting ran first absorbed all the warmup cost. Re-running 16 threads *last* as a
control gave 10.34s where running it *first* gave 31.08s, same setting. No thread
change was made. Recorded because the hypothesis was plausible, the first numbers
supported it, and only the control killed it.

**Retracted: ONNX as the fix.** Exported the cross-encoder to ONNX Runtime
(`eval/export_reranker_onnx.py`, parity-checked: max score delta 3.58e-07, identical
ranking). Real gain **1.21x** (10.34s -> 8.57s), not the 2-4x expected — ONNX Runtime
logs `Unknown CPU vendor` under Docker's VM and falls back to generic kernels. Kept,
because it is free at runtime and verified safe, but it is a trim, not the fix. It also
answered the wrong question: it optimised CPU inference *inside* a container when the
real constraint was that the reranker was in a container at all.

**The actual fix: run the reranker on the host, like Ollama already does.** Docker on
macOS has no access to Metal or the Neural Engine, so MPS/MLX/Core ML are all
unreachable from inside. Measured per 30-candidate rerank on an M5 Pro:

| Where | Per rerank |
|---|---|
| Container CPU (torch) | 10.34s |
| Container CPU (ONNX) | 8.57s |
| Host CPU (torch) | 6.58s |
| **Host GPU (MPS/Metal)** | **1.48s** |

The container costs ~1.6x on CPU work by itself; the GPU is the other ~4.4x.

**Shipped:** `retrieval/rerank_server.py`, a stdlib-`http.server` service run natively
on the host with the model on `mps`, warmed at startup, inference serialised behind a
lock (multi-query fan-out sends up to 4 concurrent requests; concurrent Metal forward
passes are not reliably safe and GPU work serialises anyway). Bound to `127.0.0.1` so
it stays off the LAN while Docker reaches it via `host.docker.internal`.
`BGEReranker` gains an HTTP path behind `RERANKER_URL`; the seam is one line — the
server returns **raw logits** and the sigmoid/sort/`top_k` stay client-side, so both
paths run identical code from model output onward.

**Verified end-to-end through the real `BGEReranker`, container to host:** `top_k`
ordering identical, max score delta **4.17e-07**, **5.95x** (8.93s -> 1.50s per rerank;
6.9x against the original torch baseline). Parity matters here specifically because
these scores feed `score_floor` — a shift near 0.55 changes which chunks survive and
would move every eval number silently.

**Deliberate design choice:** if `RERANKER_URL` is set and the server is down,
reranking **raises** rather than falling back to the in-container model. A silent
fallback would leave an unattended sweep running at 7x its budgeted latency, with
nothing in the report to explain it. Failing loudly costs one run; failing quietly
costs the experiment.

**Runbook budget corrected in both directions.** The documented "~55 min/run" was
measured at `candidates=25/top_k=5` and was already wrong before today: the real figure
is **~2.9h/run** in-container, **~1.8h/run** with the host service. Full four-phase
total is ~28h with the service, ~45h without — not the ~15h previously written down.
Added as pre-launch check 5.

**Not yet measured:** the effect on a full 125-case run. All figures above are
rerank-level or projected from mean queries/case. The next real run will show whether
~52s/case holds.

## 2026-08-26 — Phase 3 (floors) answered from existing reports; Phase 4 harness fixed; crash-resilience gap closed

Reviewed `eval/docs/tuning-runbook.md` against the code it drives before launching
any of the four phases. Findings:

**Phase 3 needed no new run.** Three uncensored (`score_floor=0.0`,
`vector_floor=0.0`) 125-case reports already existed on the same golden hash
(`228db7ac68da`): `20260824-132940`, `20260825-152139`, `20260825-165051`.
Replaying `replay_floor.py --grid` against all three (see
experiment-results.md Phase 3 for the full tables): the shipped `0.55/0.42`
wipes 0 of 125 cases at any swept cell — the floors do zero refusal work, all
refusal is the model's own `NO_ANSWER`. Score distributions do not separate
answerable from unanswerable questions (probe cosine median 0.642 vs.
in-corpus 0.558 — probes score *higher*, since a probe like "average annual
rainfall in Multan" retrieves the correct, relevant, fact-less table). The
floor's one live effect is context pruning: sweeping `vector_floor` at
`score_floor=0.55`, `0.50` wipes +4 probes / 0 in-corpus for 42% context vs.
74% at the shipped `0.42`. **Recommendation: `vector_floor: 0.42 → 0.50`,
`score_floor: 0.55` unchanged** — not yet shipped, gated on one confirm run
(replay can't see whether cutting context this much breaks generated
answers). `replay_floor.py`'s `VECTOR_GRID` widened to `[0.42, 0.46, 0.48,
0.50, 0.52, 0.55]` (was topping out at 0.45 — 0.50 was an edge winner the old
range only bracketed). Phase 3 collapses from 2 pipeline runs to 1.

**Phase 4 could not have run as written.** `eval/ingest_hybridqa.py` never
read `config.yaml` and had no `StructuralChunker` involvement at all — module
constants hardcoded 500 tokens / 20 rows, and `split_prose` was a fixed
character window with **no overlap implementation**. The three proposed
configs (`250/75/5`, `350/50/10`, `500/30/20`) would have built three
identical collections; the overlap axis had nothing to exercise. Fixed:
`--target-tokens`/`--overlap-tokens`/`--rows-per-group` flags added
(defaulting to the old hardcoded values, so no existing report is
invalidated), `split_prose` gained an `overlap_chars` parameter mirroring
`StructuralChunker._split`'s `step = max(max_chars - overlap_chars, 1)`. One
test added (`tests/test_ingest_hybridqa.py` — nothing covered `split_prose`
before). Noted for whoever runs this phase: 81% of retrieved contexts in
`20260825-152139` are prose, only 19% table rows, so `target_tokens`/overlap
are the levers that matter and `table_rows_per_group` is the least important
of the three; zero-overlap mid-word splitting is a live suspect for low
`answer_coverage`.

**A killed run was unrecoverable.** `run_eval.py` streamed every case to
`<stamp>.jsonl` (flushed per case) but wrote `<stamp>.json` — the only place
provenance lives — only after the *last* case. A run dying at case 120/125
left good cases with no config attached anywhere on disk, invisible to
`report_table.py` (which globs `2*.json`). Fixed: `run_eval.py` now writes
`<stamp>.json` with real provenance and empty `cases` *before* calling
`run_cases`, overwritten with the full report at the end. `report_table.load()`
now falls back to the sibling `.jsonl` when `cases` is empty, so a partial run
shows up as a correctly-attributed row with recomputed metrics instead of
being invisible. Verified against a simulated stub report + `.jsonl`.

**Runbook corrections, not yet applied to the file itself:** Phase 2's
`hops=3, mq=3` baseline reuse (`20260824-132940`) is invalid once Phase 1
changes `candidates`/`top_k` — that report is `candidates=25/top_k=5`, three
git shas back — so Phase 2 needs 6 new runs, not 5. No repeat-run variance
has ever been measured anywhere in this repo; `answer_coverage` at n=90
in-corpus cases has a ~±10pt 95% interval, and Branch D's headline `+9pts`
sits inside it — Phase 1/2's "take the winner" rules need a stated tie
threshold (proposed: <10pt gap = tie, keep incumbent).

**Not run:** no new pipeline evals were executed in this pass — this was a
harness-and-runbook fix, verified with `pytest`, a `replay_floor.py --grid`
rerun against all three existing reports (cross-checked, agreed to within one
case per cell — confirms these are retrieval-side numbers, not LLM noise),
and a simulated crash-recovery scenario, not a new measurement.

**Next:** commit `eval/run_ragas.py`'s pending `RunConfig(max_workers=4)`
change (else every report this week is stamped `-dirty`), fold the Phase
2/3/4 corrections into `tuning-runbook.md` itself, then run Phase 1.

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