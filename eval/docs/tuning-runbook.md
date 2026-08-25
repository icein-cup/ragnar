# Hyperparameter Tuning Runbook

Run the phases in this order — **retrieval → agentic → floors → chunking**.
They are NOT independent: `score_floor`/`vector_floor` filter the rerank and
vector scores that retrieval produces, and `candidates`/`top_k` change that
score population. Tuning floors before retrieval invalidates the floor result
the moment retrieval changes — floors go last precisely because they are the
cheapest phase to redo (one run + arithmetic). Record every result in
`eval/docs/experiment-results.md` before moving on, and a dated entry in
`eval/docs/decisions-log.md` per phase.

**Total budget, all four phases:**

| Phase | Pipeline runs | Approx |
|---|---|---|
| 1 retrieval | 6-7 sweep runs (the finalist is one of them, no separate confirm) | ~6-6.5h |
| 2 agentic | 5 new runs (baseline reused; finalist is one of them) | ~4.5h |
| 3 floors | 1 uncensored + 1 confirm run (finalist is the confirm run) | ~1.8h |
| 4 chunking | 3 configs, each requiring re-ingestion | ~4-5h |

~17-19h serial total. Each phase's RAGAS finalist check (see below) adds
external judge-call time, not another pipeline run — `run_ragas.py --report`
scores a report already on disk.

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

## Phase 2 — Agentic (max_hops × multi_query_count, ~5 runs / ~4.5h)

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
      --max-hops $hops --multi-query-count $mq
  done
done
```

`hops=3, mq=3` is the recorded baseline (`eval/reports/20260824-132940.json`)
— reuse that number instead of re-running it if the config hasn't changed
since. Watch `latency_p90`, not just `latency_mean` — that is the point of
this phase.

**This phase cannot be run in parallel.** Sharding combos across one Ollama
makes per-request latency a function of contention, and contention scales
with how many LLM calls a combo makes — so it distorts the *ranking*, not just
the scale of every number by a constant. Moot at 5 runs; noted so parallelism
isn't re-proposed here later.

**Finalist check:** same as Phase 1 — the winning combo's own run already
wrote a report; score that file with RAGAS, no extra run.

## Phase 3 — Floors (score_floor × vector_floor, 2 pipeline runs, ~1.8h)

Not ~55 min total — that's step 1 alone. Step 2 (the confirm run below) is a
second full ~55-min agentic run; skipping it and shipping straight off the
grid is exactly what the bias warning below says not to do.

Find the `score_floor` × `vector_floor` combo that maximizes
`answer_coverage` while keeping `refusal_accuracy >= 0.85`.

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
`SCORE_GRID`/`VECTOR_GRID` are hardcoded lists (`replay_floor.py:37-40`) —
currently `[0.45, 0.50, 0.55, 0.60, 0.65]` × `[0.35, 0.40, 0.42, 0.45, 0.50]`
(the shipped default `0.42` is included, so `0.55/0.42` is a directly
comparable row, not something the grid only brackets). If a winner lands on
an edge — `score_floor=0.45` or `vector_floor=0.50` — the true optimum may sit
outside the swept range; widen the lists and re-run the replay. It's free (no
LLM calls), so there's no reason to ship an edge winner unexamined.

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

## Phase 4 — Chunking (structural only, ~4-5h)

Strategy is settled (semantic vs structural was a 1-point wash, Branch G);
only the token budget is open. For each of 250/75/5, 350/50/10, 500/30/20:
edit `config.yaml` chunking, re-ingest, run one full eval, record. If no
config beats 350/50/10, keep it.

**This phase mutates the `hybridqa` collection** (re-ingestion overwrites it)
— unlike Phases 1-3, which only read it. Don't run this concurrently with
anything else that queries `hybridqa`.

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
- Each "Finding —" line states what won, by how much, and whether it beats
  the default.
- "Default is already optimal" is a valid finding — write it and freeze.
- Dated entry in `eval/docs/decisions-log.md` per phase.
