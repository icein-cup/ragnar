# Hyperparameter Tuning Runbook

Run the phases in this order — **retrieval → agentic → floors → chunking**.
They are NOT independent: `score_floor`/`vector_floor` filter the rerank and
vector scores that retrieval produces, and `candidates`/`top_k` change that
score population. Tuning floors before retrieval invalidates the floor result
the moment retrieval changes — floors go last precisely because they are the
cheapest phase to redo (one run + arithmetic). Record every result in
`eval/docs/experiment-results.md` before moving on, and a dated entry in
`eval/docs/decisions-log.md` per phase.

**Total budget, all four phases (revised 2026-08-26 — see decisions-log.md):**

| Phase | Pipeline runs | Approx |
|---|---|---|
| 1 retrieval | 6-7 sweep runs (the finalist is one of them, no separate confirm) | ~6-6.5h |
| 2 agentic | 6 new runs — baseline is NOT reusable (see Phase 2), finalist is one of the 6 | ~5.5h |
| 3 floors | **Answered from existing reports** (see Phase 3) — 1 confirm run only | ~55min |
| 4 chunking | 2 new configs + Phase 3's confirm run as the baseline row, each new config requiring re-ingestion | ~2-3h |

~15-16h serial total. Each phase's RAGAS finalist check (see below) adds
external judge-call time, not another pipeline run — `run_ragas.py --report`
scores a report already on disk.

**Tie threshold — read before ranking any grid.** No repeat-run variance has
ever been measured in this repo. `answer_coverage` at n=90 in-corpus cases has
a binomial 1σ of ~5pts and a 95% interval of ~±10pts; Branch D's own headline
`+9pts` (`candidates=30/top_k=10` over the old default) sits inside that band.
Treat any gap under 10pts of `answer_coverage` between two configs as a tie
and keep the incumbent — this applies to Phase 1's "note the winning top_k"
and Phase 2's ranking both.

## Before you launch

Four checks, in order, before Phase 1 starts. All were verified live against
this repo's actual environment (services, models, RAM) on 2026-08-25 — the
numbers below are measured, not estimated.

**1. Clean git tree — required for provenance.** `run_eval.py`'s
`_git_revision()` stamps every report `<sha>-dirty` whenever
`git status --porcelain` is non-empty, and that file's own docstring is blunt
about why: *"A report that cannot be traced to code is a number without a
cause."* Commit any pending harness/doc changes before running anything, then
confirm:

```bash
git status --short   # must be empty
```

`eval/reports/` is gitignored (`.gitignore:24`), so the sweep's own output
never re-dirties the tree mid-run — this is a one-time check before Phase 1,
not something to repeat between phases.

**2. Smoke test — required, ~4 min, not a measurement.** `tune_params.py` and
`run_ragas.py` have only run against mocked subprocesses in unit tests; their
live paths (actually reaching Qdrant and Ollama) have not executed even once.
Discovering a wiring problem at hour 6 of an unattended sweep is expensive;
catching it here costs 4 minutes:

```bash
docker compose exec app python eval/run_eval.py --agentic \
  --collection hybridqa --golden eval/golden_subset20.yaml
```

Confirm: the run completes and writes `eval/reports/<stamp>.json`;
`summary.provenance.git` has **no** `-dirty` suffix (proves check 1 actually
took effect); `summary.provenance.golden_cases` is 20; the `latency` block is
present. Then confirm the RAGAS post-process path on that same report:

```bash
docker compose exec app python eval/run_ragas.py --report eval/reports/<stamp>.json
```

This makes real (cheap) calls to the `glm-5.2:cloud` judge — an external paid
API, so it's worth knowing that before running it.

**`golden_subset20.yaml` is a wiring check only.** `decisions-log.md`
(commit `35a07de`) already documents a subset-vs-full-set false-signal lesson
— nothing from these 20 cases belongs in `experiment-results.md` or should
influence a tuning decision. It only proves the pipes connect.

**3. RAM — comfortable for Phases 1-3, tight for Phase 4.** Measured:
48 GB physical, 33.9 GB free+inactive, swap lightly used (normal for macOS).
Serial working set for the agentic pipeline (Phases 1-3, no ingestion):

| Component | Approx |
|---|---|
| `qwen2.5:7b` (4.7 GB weights + ~1.9 GB KV at num_ctx 32768) | ~6.6 GB |
| `bge-m3` embedder | 1.2 GB |
| BGE reranker (app container) | ~2.3 GB |
| Qdrant | 1.1 GB |
| **Total** | **~11 GB against 33.9 GB free — ~3x headroom** |

No action needed for Phases 1-3. Phase 4's ingestion has a separate,
tighter constraint — see its section below.

**4. Services and data — already verified, re-check if time has passed.**
`docker compose ps` should show `app` and `qdrant` up; the `hybridqa`
collection should hold 2846 points at 1024 dims (`bge-m3`'s output size).
`qwen2.5:7b`, `bge-m3`, and the RAGAS judge model must be present in Ollama,
and `RAGAS_JUDGE_BASE_URL`/`RAGAS_JUDGE_API_KEY`/`RAGAS_JUDGE_MODEL` must be
set in the app container's environment.

## Setup

Tune against HybridQA (the only corpus with a full golden set):

```bash
docker compose up -d
# collection + golden used throughout:
--collection hybridqa --golden eval/golden_hybridqa_draft.yaml
```

One `--agentic` run over 125 cases ≈ 55 min (~10s/case) — the agentic path
makes ~5 LLM calls per case. Retrieval and agentic have no offline shortcut
(each combo changes what is retrieved or how the pipeline queries), so every
row in those two phases costs a full run.

`--golden` must always be passed alongside `--collection` — nothing checks
they agree, and a mismatched pair silently scores the wrong questions against
the wrong corpus.

## Phase 1 — Retrieval (candidates × top_k, ~6-7 runs / ~6h)

Does more candidate/top_k headroom buy recall without noise? Branch D already
found `top_k=10/candidates=30` = +9pts over the old `candidates=25/top_k=5`
default (see `config.yaml` retrieval comments) — this phase is about
confirming/refining around that point, not re-deriving it from scratch.

`top_k=12` is dropped from the grid: it pushes more context into an
already-over-latency-budget generation call (see Phase 2 below) for headroom
Branch D's own numbers show flattening past `top_k=10`.

Coordinate descent, not the full grid — hold one axis at default, sweep the
other, take the winner, sweep the second axis against it:

```bash
# Step 1: top_k axis, candidates held at the config default (30)
for tk in 5 8 10; do
  docker compose exec app python eval/run_eval.py --agentic \
    --collection hybridqa --golden eval/golden_hybridqa_draft.yaml \
    --candidates 30 --top-k $tk
done
# note the winning top_k as <TK*>

# Step 2: candidates axis, top_k held at <TK*>
for c in 20 25 40; do
  docker compose exec app python eval/run_eval.py --agentic \
    --collection hybridqa --golden eval/golden_hybridqa_draft.yaml \
    --candidates $c --top-k <TK*>
done
```

(If `<TK*>` comes out to 10, the `candidates=30` cell is already covered by
Step 1 and Step 2 only needs 3 runs, not 4 — 6 runs total, not 7.)

Record all rows in experiment-results.md. A confirm of `top_k=10/candidates=30`
is "confirms Branch D", not a new finding.

**Finalist check:** once a winner is picked, its own sweep run already wrote
an `eval/reports/<stamp>.json` — score that file with RAGAS (see "RAGAS
finalist check" below) before it's considered done. No extra pipeline run.

## Phase 2 — Agentic (max_hops × multi_query_count, 6 runs / ~5.5h)

**Latency, not coverage, is the open question here** — read this before
running anything. The current default `max_hops=3, multi_query_count=3`
already measures p90 **53.5s** against an agreed target of p90 15-20s
(`eval/reports/20260824-132940.json`; see experiment-results.md "Latency
reality"). Commit `256c051` separately measured `multi_query_count=5` at
**+47s/case** and reverted it. Together those disqualify 12 of the 16 combos
in the originally-planned grid before a single new run: `multi_query_count`
5/7 on the +47s measurement, `max_hops=4` because `max_hops=3` already misses
the p90 target 3x. See experiment-results.md Phase 2 for the full grid table.

The question this phase actually answers is **how far down these parameters
can go before coverage breaks** — `max_hops=2` or `multi_query_count=2` is a
latency *win* against a budget already being missed, not a tradeoff to weigh
against a coverage gain.

Run the full remaining grid (small enough that coordinate descent isn't
needed):

```bash
for hops in 1 2 3; do
  for mq in 2 3; do
    docker compose exec app python eval/run_eval.py --agentic \
      --collection hybridqa --golden eval/golden_hybridqa_draft.yaml \
      --candidates <C*> --top-k <TK*> \
      --max-hops $hops --multi-query-count $mq
  done
done
```

**`hops=3, mq=3` is NOT reusable as the baseline — re-run it, 6 runs not 5.**
The recorded baseline (`eval/reports/20260824-132940.json`) is
`candidates=25/top_k=5`; by the time this phase starts, Phase 1 has already
changed those to `<C*>/<TK*>`. A wider retrieved set changes what multi-hop
and self-correction see, so the old report is not a like-for-like comparison
— corrected 2026-08-26, see decisions-log.md. Note also that report's
`summary.latency` field is `null` (the 23.2s/15.9s/53.5s/144.9s figures
circulating for it were computed ad hoc from per-case `stages.seconds`, not
read from that field) — don't expect tooling reading `summary.latency` to see
them. Watch `latency_p90`, not just `latency_mean` — that is the point of
this phase.

**This phase cannot be run in parallel.** Sharding combos across one Ollama
makes per-request latency a function of contention, and contention scales
with how many LLM calls a combo makes — so it distorts the *ranking*, not just
the scale of every number by a constant. Moot at 5 runs; noted so parallelism
isn't re-proposed here later.

**Finalist check:** same as Phase 1 — the winning combo's own run already
wrote a report; score that file with RAGAS, no extra run.

## Phase 3 — Floors (score_floor × vector_floor, 1 pipeline run, ~55min)

**Already answered as of 2026-08-26 — this phase needs a confirm run, not a
fresh sweep.** Three uncensored (`score_floor=0.0`/`vector_floor=0.0`)
125-case reports already exist on the same golden hash (`228db7ac68da`):
`20260824-132940`, `20260825-152139`, `20260825-165051`. Replaying all three
agreed to within one case per grid cell (that agreement is itself the
cross-check that this is a retrieval-side signal, not LLM noise). Full
writeup in experiment-results.md Phase 3; headline:

- The shipped `0.55/0.42` wipes 0 of 125 cases at any cell — the floors do no
  refusal work on this corpus. Refusal is carried entirely by the model's
  `NO_ANSWER`.
- Probe and in-corpus score distributions overlap and do not separate
  cleanly — probes score *higher* on cosine (median 0.642 vs 0.558), because
  a probe like "average annual rainfall in Multan" retrieves the correct,
  relevant, fact-less table. No retrieval threshold sees that; only the
  model reading the content can.
- The floor's one live effect is context pruning. `vector_floor: 0.42 → 0.50`
  (keep `score_floor: 0.55`) wipes +4 probes / 0 in-corpus cases, cuts
  context from 74%→42% of retrieved chunks.

Skip step 1 below — no new uncensored run is needed, it already exists three
times over. Go straight to the confirm:

```bash
# confirm the recommended combo — on top of whatever Phase 1 settles on
docker compose exec app python eval/run_eval.py --agentic \
  --score-floor 0.55 --vector-floor 0.50 \
  --candidates <C*> --top-k <TK*> \
  --collection hybridqa --golden eval/golden_hybridqa_draft.yaml
```

If it holds `answer_coverage` within the tie band (see Recording) and raises
`refusal_accuracy`, update `config.yaml`. Watch `citation_precision` and
`latency_p90` too — the 58% context cut should move both favourably, and if
it doesn't, that's itself a finding.

If a re-derivation is ever actually needed (a corpus change, a new golden
set), the original procedure was:

```bash
# 1. one uncensored run (floors 0.0 so every score is saved) — ~55 min
docker compose exec app python eval/run_eval.py --agentic \
  --score-floor 0 --vector-floor 0 \
  --collection hybridqa --golden eval/golden_hybridqa_draft.yaml
# note the <stamp>, then:
# 2. replay floors offline (seconds, no LLM)
docker compose exec app python eval/replay_floor.py \
  --report eval/reports/<stamp>.json --grid
```

**Read before trusting the grid.** `replay_floor.py` re-derives `refused`
purely from whether a chunk clears the candidate floor
(`replay_floor.py:apply_floor`), overwriting whatever the model itself
decided, and it never re-runs generation. Two consequences:

- `refusal_accuracy` in the grid is a **lower bound** — a probe the model
  correctly refused gets re-scored as answered whenever any chunk clears the
  floor.
- `answer_coverage` in the grid is an **upper bound** — a case stays credited
  as correct even when the chunk that carried the answer got filtered out.
- `citation_accuracy` / `citation_precision` columns are **not meaningful** —
  citations are never re-derived, so treat those two columns as
  informational only, not decision inputs.

Both biases push toward floors that look better than they are.

**The grid is a fixed set of candidates, not a continuous search.**
`SCORE_GRID`/`VECTOR_GRID` are hardcoded lists (`replay_floor.py:37-41`) —
currently `[0.45, 0.50, 0.55, 0.60, 0.65]` × `[0.42, 0.46, 0.48, 0.50, 0.52,
0.55]` (the shipped default `0.42` is included, so `0.55/0.42` is a directly
comparable row, not something the grid only brackets; widened past the old
0.35-0.45 range on 2026-08-26 because 0.50 turned out to be the useful edge).
If a winner lands on an edge again, the true optimum may sit outside the
swept range; widen the lists and re-run the replay. It's free (no LLM calls),
so there's no reason to ship an edge winner unexamined.

Record the grid in the experiment-results.md Phase 3 table + a "Finding —"
line, but **do not edit `config.yaml` from the grid alone.** Confirm the
winning combo with one real run first:

```bash
docker compose exec app python eval/run_eval.py --agentic \
  --score-floor <winning_score> --vector-floor <winning_vector> \
  --collection hybridqa --golden eval/golden_hybridqa_draft.yaml
```

If that confirms a combo beating 0.55/0.42, update `config.yaml`. Stop here if
floors were the goal.

**Finalist check:** the confirm run above writes its own
`eval/reports/<stamp>.json` — score that file with RAGAS (see below) before
editing `config.yaml`. No third pipeline run.

## RAGAS finalist check (all three phases above)

`eval/run_ragas.py` scores the judged RAG triad — `faithfulness`,
`answer_relevancy`, `context_precision`/`recall` — plus `answer_correctness`
and the house `no_invented_numbers` critic. None of the deterministic metrics
in `eval/metrics.py` can catch a drop in these: `answer_coverage` only checks
whether the expected words appear in the answer, not whether the rest of it
is invented.

Running RAGAS on every grid combo is not worth it — it's an external, paid
judge call per case (needs `RAGAS_JUDGE_BASE_URL` / `RAGAS_JUDGE_API_KEY`
set). Run it once per phase, on the finalist only.

**Use `--report`, not `--agentic`.** Every finalist above already has a saved
`eval/reports/<stamp>.json` from its own run — the sweep run for Phases 1-2,
the confirm run for Phase 3. Pointing `run_ragas.py` at that file scores it as
a post-process, no pipeline re-run:

```bash
docker compose exec app python eval/run_ragas.py \
  --report eval/reports/<stamp>.json
```

Passing `--agentic --collection ... --golden ...` instead (the pre-`--report`
form) still works, but re-runs retrieval and generation from scratch —
another full ~55-min pipeline run per phase, on top of the run that already
produced the finalist's deterministic metrics. Always prefer `--report`
here; `--agentic` on this command is for scoring a fresh unsaved run, not for
the finalist check.

Record the three means (`faithfulness`, `answer_correctness`,
`context_precision`) in that phase's experiment-results.md table. Treat a
`faithfulness` drop against the current baseline (0.70, per
decisions-log.md's tracking table) as a reason to keep the previous default
even if `answer_coverage` improved — `answer_coverage` alone cannot see
hallucination, `faithfulness` is the gate for that.

## Phase 4 — Chunking (structural only, 2 new runs / ~2-3h)

**This phase could not have run as written before 2026-08-26 — read this
before editing `config.yaml`.** `eval/ingest_hybridqa.py` does not read
`config.yaml` and does not use `StructuralChunker` — editing
`config.yaml`'s chunking section and re-ingesting via that script would have
built the *same* collection every time, silently. It has its own,
now-parameterized, path: `--target-tokens` / `--overlap-tokens` /
`--rows-per-group` flags (defaults 500/0/20, matching every existing
report). Use those flags, not `config.yaml`, to vary this phase.

Strategy is settled (semantic vs structural was a 1-point wash, Branch G);
only the token budget is open. 81% of retrieved contexts on this corpus are
prose passages, only 19% are table rows (measured on
`20260825-152139`) — so `--target-tokens`/`--overlap-tokens` are the levers
that matter here and `--rows-per-group` is the least important of the three.
The old default has **zero overlap** and splits prose on a raw character
window (mid-word, no sentence awareness) — a plausible direct cause of low
`answer_coverage` if an answer sentence lands split across two chunks.

```bash
# baseline row: reuse Phase 3's confirm run on the existing hybridqa
# collection (500/0/20, today's default) — no re-ingest needed for this row.

# overlap only — isolates the mid-sentence-split hypothesis
docker compose exec app python eval/ingest_hybridqa.py \
  --collection hybridqa_500_75 --golden eval/golden_hybridqa_draft.yaml \
  --target-tokens 500 --overlap-tokens 75 --rows-per-group 20
docker compose exec app python eval/run_eval.py --agentic \
  --collection hybridqa_500_75 --golden eval/golden_hybridqa_draft.yaml \
  --candidates <C*> --top-k <TK*> --score-floor <winning_score> --vector-floor <winning_vector>

# config.yaml's actual production values
docker compose exec app python eval/ingest_hybridqa.py \
  --collection hybridqa_350_50 --golden eval/golden_hybridqa_draft.yaml \
  --target-tokens 350 --overlap-tokens 50 --rows-per-group 10
docker compose exec app python eval/run_eval.py --agentic \
  --collection hybridqa_350_50 --golden eval/golden_hybridqa_draft.yaml \
  --candidates <C*> --top-k <TK*> --score-floor <winning_score> --vector-floor <winning_vector>
```

Naming each config its own collection (`hybridqa_500_75`, not overwriting
`hybridqa`) means nothing needs re-ingesting twice and there's no
"don't query concurrently" hazard to manage — the old collection stays
queryable throughout. If no config beats the baseline, keep it and drop the
two scratch collections.

**Honest framing:** this measures the *eval* ingester's chunking, which is a
separate code path from the app's `StructuralChunker` (see above). A win
here is a hypothesis about production chunking, not a direct measurement of
it — `StructuralChunker`'s own `target_tokens`/`overlap_tokens`/
`table_rows_per_group` would need a dedicated pass to confirm.

**Ingestion worker count.** `ingestion.workers: 4` (as the previous version
of this section warned) applies to the *app's* document ingestion pipeline,
which runs a Docling converter per worker at ~1-2 GB each. `eval/ingest_hybridqa.py`
does not go through that pipeline — it fetches pre-parsed JSON directly from
GitHub (`TABLE_URL`/`REQUEST_URL`) and does no Docling conversion, so the
8-worker/17.2 GB OOM risk described there does not apply here. No worker
setting is needed for this phase.

## Optional: full-grid sweep via tune_params.py

`eval/tune_params.py --phase retrieval` and `--phase agentic` run the full
4×4 grids (`RETRIEVAL_GRID` / `AGENTIC_GRID`) instead of the reduced set
above — 16 runs each, ~15h serial per phase. Only reach for this if the
reduced grids above leave a real question open (e.g. an axis winner
contradicts Branch D's `top_k=10/candidates=30`); the interaction effects a
full grid buys are not worth ~30h against a coordinate-descent result that
already agrees with prior findings.

```bash
docker compose exec app python eval/tune_params.py --phase retrieval \
  --collection hybridqa --golden eval/golden_hybridqa_draft.yaml
```

Sharding this across parallel workers is possible for the **retrieval** phase
only (metric is `answer_coverage`, which contention doesn't move) — **not**
for agentic, whose metric is latency and whose whole point contention would
corrupt. Two things need fixing first if this is attempted:

- `run_cases`' `finally` calls `llm.unload()` (`run_eval.py:187-193`), which
  is a **global** Ollama eviction (`keep_alive=0` on `/api/generate`,
  `generation/llm.py:82-97`), not per-client — one shard finishing a combo
  evicts the model out from under the others. Guard the call off for sweep
  runs.
- The report stamp is second-resolution (`run_eval.py:350`); shards launched
  together collide on the same `<stamp>.jsonl`/`.json`. Stagger launches by a
  few seconds, or append the PID.

`tune_params.py` has no built-in shard flag; splitting the grid means editing
`RETRIEVAL_GRID` per shard or writing explicit `run_eval.py` command batches.
Given the fixes required and that coordinate descent already answers the
question in ~6h serial, this is a fallback, not a recommendation.

## Recording

- Every result → `eval/docs/experiment-results.md` before the next phase.
  Generate the numbers with `python eval/report_table.py` (`--full` for every
  provenance field) rather than transcribing from terminal output by hand —
  it recomputes any metric a run predates from its saved cases, so it stays
  correct even against older reports.
- **Tie threshold: a gap under 10pts of `answer_coverage` is noise, not a
  finding.** No repeat-run variance has been measured anywhere in this repo;
  n=90 in-corpus cases puts a 95% interval at roughly ±10pts, and Branch D's
  own `+9pts` headline sits inside it. Below that gap, keep the incumbent
  and say so — don't pick a "winner" the data can't actually support.
- Each "Finding —" line states what won, by how much, and whether it beats
  the default.
- "Default is already optimal" is a valid finding — write it and freeze.
- Dated entry in `eval/docs/decisions-log.md` per phase.
- **A run that dies mid-sweep is not lost.** `run_eval.py` stamps
  `<stamp>.json` with real provenance before the run starts and streams every
  case to `<stamp>.jsonl` as it goes; `report_table.py` reads the `.jsonl`
  fallback when `cases` is empty. Killed at case 120 of 125 still shows up as
  a correctly-attributed 120-case row — check `report_table.py --full` before
  re-running a combo from scratch.
