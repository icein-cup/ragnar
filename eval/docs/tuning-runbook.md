# Hyperparameter Tuning Runbook

Run the four tuning phases in order. Each is independent — stop after any
phase and the results so far are valid. Record every result in
`eval/docs/experiment-results.md` before moving on.

## Setup

Tune against HybridQA (the only corpus with a full golden set):

```bash
docker compose up -d
# collection + golden used throughout:
--collection hybridqa --golden eval/golden_hybridqa_draft.yaml
```

One `--agentic` run over 125 cases ≈ 55 min (~10s/case) — the agentic path
makes ~5 LLM calls per case. Phases 2-3 have no offline shortcut (each combo
changes what is retrieved), so they are ~15h serial. See "Running Phases 2-3
in parallel" below to cut that to ~7-8h.

## Phase 1 — Floors (~55 min, do first)

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

Record the grid in the experiment-results.md Phase 1 table + a "Finding —" line. If
a combo beats 0.55/0.42, update `config.yaml`. Stop here if floors were the
goal.

## Phase 2 — Retrieval (candidates × top_k)

Does more candidate/top_k headroom buy recall without noise?

```bash
docker compose exec app python eval/tune_params.py --phase retrieval --collection hybridqa
```

Record in experiment-results.md. Branch D already found `top_k=10/candidates=30` =
+9pts; a confirm is "confirms Branch D", not new.

## Phase 3 — Agentic (max_hops × multi_query_count)

Marginal gain per hop / query variant vs latency.

```bash
docker compose exec app python eval/tune_params.py --phase agentic --collection hybridqa
```

Record in experiment-results.md. Watch `latency_mean` — that is the point.

## Phase 4 — Chunking (structural only, ~4-5h)

Strategy is settled (semantic vs structural was a 1-point wash, Branch G);
only the token budget is open. For each of 250/75/5, 350/50/10, 500/30/20:
edit `config.yaml` chunking, re-ingest, run one full eval, record. If no
config beats 350/50/10, keep it.

## Running Phases 2-3 in parallel

The bottleneck is Ollama, not the 48 GB M5: a single Ollama instance
serializes concurrent requests to the same model (`qwen2.5:7b`, ~6.6 GB).

**Lever:** `OLLAMA_NUM_PARALLEL` (supported) controls how many requests
Ollama runs simultaneously per model. Set it to 4 on the host, restart
Ollama. It is not free parallelism — GPU compute is shared, so 4× concurrent
requests give ~1.5-2.5×, not 4×.

**Then shard the grid.** `tune_params.py` runs all 16 combos serially with no
subset option. Two ways:

- **Shard flag** (reusable): add `--shard N/M` to `tune_params.py` to run
  slice N of M combos, then launch 4 background jobs:
  ```bash
  docker compose exec -d app python eval/tune_params.py --phase retrieval --shard 1/4 &
  docker compose exec -d app python eval/tune_params.py --phase retrieval --shard 2/4 &
  docker compose exec -d app python eval/tune_params.py --phase retrieval --shard 3/4 &
  docker compose exec -d app python eval/tune_params.py --phase retrieval --shard 4/4 &
  ```
  (If `exec -d` doesn't detach, use `nohup ... &` inside the container.)
  Merge the 4 shard CSVs (dedupe header), then summarize.

- **Pure shell** (no code change, one-time): write 16 explicit `run_eval.py`
  commands, split across 4 background shells.

**Caveats:** 4 concurrent 7b requests ≈ 2× speedup (drop to 2 shards if
worse). Each combo is a separate `run_eval.py` with `seed=42`, so sharding
does not change per-combo results. All shards read `hybridqa` read-only —
safe.

| Approach | Wall-clock per phase |
|----------|----------------------|
| Serial | ~15h |
| 4 shards + NUM_PARALLEL=4 | ~7-8h |
| 2 shards (fallback) | ~10-12h |

## Recording

- Every result → `eval/docs/experiment-results.md` before the next phase.
- Each "Finding —" line states what won, by how much, and whether it beats
  the default.
- "Default is already optimal" is a valid finding — write it and freeze.
- Dated entry in `eval/docs/decisions-log.md` per phase.
