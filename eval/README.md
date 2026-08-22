# Evaluation

Development-time only. Never runs in the app, never runs in CI automatically.

## Quickstart

### 1. Set up the judge

Copy `.env.example` to `.env` and fill the three Ragas vars. The key lives
in the eval environment only; the app never reads it. `docker-compose.yml`
forwards the vars into the container.

```
RAGAS_JUDGE_BASE_URL=https://ollama.com/v1   # OpenAI-compatible base (note /v1)
RAGAS_JUDGE_API_KEY=<key>                    # eval-only key
RAGAS_JUDGE_MODEL=glm-5.2:cloud
```

Start the stack if it isn't already running:

    docker compose up -d

### 2. Ingest your documents

Copy evaluation documents into `data/inbox/` (or upload via the UI). The
worker picks them up, chunks, and indexes them automatically. Real
evaluation documents live in `corpus/` (gitignored — never committed);
that folder is staging only — the app reads from `data/inbox`.

Wait until all documents show as ✅ done in the UI documents panel (expand
"Documents" in the sidebar).

### 3. Look up doc_ids

The UI documents panel shows filenames, not ids. Query the registry:

    docker compose exec app python -c "from core.config import Config; from ingestion.registry_db import Registry; r = Registry(Config().data_dir / 'registry.db'); [print(d.doc_id, d.filename, d.status.value) for d in r.all()]"

Only `done` documents have converted markdown. Note the `doc_id` of each
document you want to generate golden entries for.

### 4. Draft golden examples

    docker compose exec app python eval/gen_golden.py --docs <doc_id> --n 10 > eval/golden_draft.yaml

`gen_golden.py` reads the ingested document's converted markdown and emits
candidate question/answer/reference entries to stdout — never
`golden_set.yaml`.

### 5. Review the draft (mandatory)

The judge can echo its own mistakes back into ground truth, so this gate is
not optional. For each entry:
- Is the question answerable ONLY from the excerpt?
- Is the answer a short, factual statement from the excerpt?
- Is the reference genuinely verbatim?

The `reference` field is a source excerpt for your review — it is not fed
to Ragas.

### 6. Merge into the golden set

Copy accepted entries into `eval/golden_set.yaml` in this format:

```yaml
- question: "What is the service contract number?"
  expected_answer: "SC-4471"
  expected_sources: ["your-doc.pdf"]
  out_of_corpus: false
```

### 7. Add out-of-corpus probes

Roughly one quarter of the set should be `out_of_corpus: true` — questions
the corpus cannot answer (e.g. "What is the capital of France?").
`gen_golden.py` only produces in-corpus entries, so add these by hand.

### 8. Run the eval

    docker compose exec app python eval/run_ragas.py

Report goes to `eval/reports/ragas-<timestamp>.json`.

---

## Deterministic metrics (no judge, free)

For a quick check without the LLM judge:

    docker compose exec app python eval/run_eval.py             # deterministic metrics
    docker compose exec app python eval/run_eval.py --calibrate # sweep the similarity floor
    docker compose exec app python eval/run_eval.py --agentic   # measure the pipeline the UI runs

`--agentic` routes through `AgenticSearch` (query rewrite, multi-query,
multi-hop, self-correction) instead of the bare `Search`. The UI always takes
the agentic path, so without this flag the numbers describe a pipeline no
user hits. It costs several LLM calls per case.

`--vector-floor` overrides `retrieval.vector_floor` for the run. During
`--calibrate` that floor is *pinned*, not swept, and printed in the header —
sweep the other axis by re-running with a different value.

| Metric | Judge | Measures |
|---|---|---|
| refusal_accuracy | No | Refused exactly on out-of-corpus questions |
| citation_accuracy | No | Cited the expected source |

These cover the two failure modes that matter most in a business context:
confidently answering something the corpus doesn't contain, and citing the
wrong source.

---

## Ragas judged metrics

`eval/run_ragas.py` layers Ragas on top of the deterministic metrics, using
an external LLM judge (OpenAI-compatible, e.g. Ollama Cloud — model
`glm-5.2:cloud`) plus the same local embedding model (Ollama `bge-m3`). It
reuses `run_cases()`, so the exact same retrieval + answer pipeline is
exercised.

| Metric | Judge | Measures |
|---|---|---|
| faithfulness | LLM | Answer supported by retrieved chunks (hallucination) |
| answer_relevancy | LLM + embeddings | Answer on-question, no fluff |
| answer_correctness | LLM + embeddings | Factual match vs `expected_answer` |
| answer_similarity | embeddings | Semantic closeness to expected |
| context_precision | LLM | Relevant chunks ranked top |
| context_recall | LLM | All answer-relevant chunks surfaced |
| context_entity_recall | LLM | Golden entities recovered |
| no_invented_numbers | LLM | Answer states no figure absent from contexts |

Golden-set field map (no schema change needed):

    question        -> user_input
    answer          -> response
    expected_answer -> reference
    contexts        -> retrieved_contexts

Out-of-corpus cases are refused by design and have no answer/contexts to
score — they are skipped here and covered by `refusal_accuracy`.

---

## Isolation

`eval/` imports app modules; the app never imports `eval/`. This boundary
matters: an external LLM judge's API key lives only in this environment.
Production documents have no code path to an external service through the
app itself.

---

## Current state

The golden set has 5 cases (3 in-corpus, 2 out-of-corpus) against the
`sample.pdf` fixture only — enough to prove the harness works end-to-end,
not enough to calibrate the similarity floor with real confidence. The
design calls for 30-50 hand-written cases against the real corpus before
this calibration should be trusted for production use.

**The current `retrieval.score_floor` (0.55, in `config.yaml`) has not been
validly calibrated.** It was chosen by sweeping candidate floors with
`--calibrate`, but that sweep was measuring nothing: `run_cases` built its
`Search` with positional arguments that stopped at `score_floor`, leaving
`vector_floor` at its `0.0` default. The two floors are OR'd, and a reranked
result's `vector_score` is a cosine that is >= 0 in practice — so every
result cleared the gate at every floor in the sweep, and refusal only ever
fired when retrieval returned zero candidates. The printed column could not
vary.

That is fixed (both floors are passed now, and the pinned `vector_floor`
is printed in the header), but the number it produced has not been
re-derived. Re-run `--calibrate` against a real corpus before treating 0.55
as anything but a placeholder. With only 5 golden cases the signal would be
crude even once the mechanism works.

A second floor, `retrieval.vector_floor` (0.42), was added later: the
reranker scores table-row chunks as near-neutral regardless of relevance,
so a chunk is now refused only when *both* the rerank score and the raw
vector-similarity score fall below their floors. `--calibrate` currently
sweeps `score_floor` only — `vector_floor` was set from a handful of live
measurements against a table-heavy corpus, not from the golden set, and
needs its own calibration pass once the golden set is large enough to
cover tabular content.
Treat it as a starting point, not a validated production threshold.

---

## Comparing configurations

    # edit config.yaml: chunking.strategy: semantic (or back to structural)
    docker compose restart app
    # re-ingest the corpus via the UI or a script
    docker compose exec app python eval/run_eval.py
    # diff against the previous report in eval/reports/

---

## Troubleshooting

**"RAGAS_JUDGE_BASE_URL and RAGAS_JUDGE_API_KEY must be set"**
Credentials missing. Fill `.env` and re-run via docker compose so they're
forwarded.

**"No in-corpus answered cases to score"**
Golden set has no in-corpus entries that produced an answer, or the corpus
was never ingested. Ingest docs, then grow `golden_set.yaml`.

**Out-of-corpus cases skipped silently**
By design — a refused case has no answer or contexts to score.
`refusal_accuracy` covers those.

**Judge returns non-JSON in `gen_golden.py`**
The model didn't follow the JSON instruction. Re-run, or lower `--n`. The
failed document is skipped and reported on stderr.