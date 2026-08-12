# Evaluation

Development-time only. Never runs in the app, never runs in CI automatically.

## Isolation

`eval/` imports app modules; the app never imports `eval/`. This boundary
matters: if an external LLM judge (e.g. Ragas) is added later, its API key
would live only in this environment. Production documents have no code
path to an external service through the app itself.

## Running

    docker compose exec app python eval/run_eval.py             # deterministic metrics
    docker compose exec app python eval/run_eval.py --calibrate # sweep the similarity floor

## Metrics

| Metric | Judge | Measures |
|---|---|---|
| refusal_accuracy | No | Refused exactly on out-of-corpus questions |
| citation_accuracy | No | Cited the expected source |

Both are deterministic and free to run — no LLM judge, no API key, no cost.
They cover the two failure modes that matter most in a business context:
confidently answering something the corpus doesn't contain, and citing the
wrong source.

## Current state

The golden set has 5 cases (3 in-corpus, 2 out-of-corpus) against the
`sample.pdf` fixture only — enough to prove the harness works end-to-end,
not enough to calibrate the similarity floor with real confidence. The
design calls for 30-50 hand-written cases against the real corpus before
this calibration should be trusted for production use.

The current `retrieval.score_floor` (0.55, in `config.yaml`) was chosen by
sweeping candidate floors with `--calibrate` and picking the lowest floor
that reached the best observed refusal_accuracy without lowering
citation_accuracy. With only 5 cases this is a crude signal — it separates
the two out-of-corpus questions from the three in-corpus ones cleanly at
this floor, but a single mis-scored case would shift the whole picture.

A second floor, `retrieval.vector_floor` (0.42), was added later: the
reranker scores table-row chunks as near-neutral regardless of relevance,
so a chunk is now refused only when *both* the rerank score and the raw
vector-similarity score fall below their floors. `--calibrate` currently
sweeps `score_floor` only — `vector_floor` was set from a handful of live
measurements against a table-heavy corpus, not from the golden set, and
needs its own calibration pass once the golden set is large enough to
cover tabular content.
Treat it as a starting point, not a validated production threshold.

Ragas integration (judged metrics requiring an external LLM judge) is not
yet implemented — the isolation boundary above is designed to support it
when it's added, but no external API calls happen today.

## Comparing configurations

    # edit config.yaml: chunking.strategy: semantic (or back to structural)
    docker compose restart app
    # re-ingest the corpus via the UI or a script
    docker compose exec app python eval/run_eval.py
    # diff against the previous report in eval/reports/
