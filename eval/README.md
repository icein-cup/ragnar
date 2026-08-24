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
| answer_accuracy | No | The answer states the expected answer |
| refusal_accuracy | No | Refused exactly on out-of-corpus questions |
| citation_accuracy | No | Cited the expected source |
| citation_precision | No | Of the sources cited, how many were needed |

These cover the failure modes that matter most in a business context:
confidently answering something the corpus doesn't contain, citing the wrong
source, and — the one a citation check cannot see — retrieving the right
document and then stating a fact that is not in it.

Read `citation_accuracy` and `answer_accuracy` together, never apart. On the
HybridQA benchmark they read 0.83 and 0.50: the retriever is doing its job and
the answerer is not.

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

`golden_set.yaml` holds 208 cases (51 out-of-corpus) across 17 real
documents. `golden_hybridqa_draft.yaml` holds a separate 125 (35
out-of-corpus) for the cross-document multi-hop benchmark.

The headline result, measured over the HybridQA benchmark on `qwen2.5:3b`:
**96% of answered questions cite a source the question needed, and 50%
actually state the right answer.** Retrieval finds the document; the model
then misreads it about half the time. `answer_accuracy` is the metric that
shows this — it was added after a review found that `citation_accuracy`
alone made a system answering wrongly look healthy.

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
as anything but a placeholder.

Measured on the HybridQA corpus, the floors are worse than uncalibrated —
they are **inert**. `bge-reranker-v2-m3` returns the neutral 0.500 (a zero
logit, "no opinion") for most chunks, including 44% of the chunks a question
genuinely needs, and tops out at 0.731. A 0.55 floor under that distribution
rejects almost nothing: **zero** cases in the last 125-case run were refused
by retrieval. Every refusal came from the model saying so in words. Treat
`classify()`'s NO_RESULTS path as untested on table-heavy corpora.

Reports now record the rerank and vector score of every retrieved chunk, so
the next calibration is arithmetic over a saved report instead of nine full
pipeline runs.

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

## Cross-document multi-hop benchmark (HybridQA)

The hand-written golden set above is single-hop: every question answers from
one document. Cross-document multi-hop — the case where an answer needs the
anchor table **and** one or more linked passages — is measured separately,
against the public **HybridQA** dataset (`wenhuchen/HybridQA` +
`wenhuchen/WikiTables-WithLinks`), so the benchmark is independent of the
random documents otherwise uploaded. HybridQA is table-plus-text, so it also
exercises the `is_table` / `table_summary` retrieval path.

### 1. Draft the golden set

    .venv/bin/python eval/load_hybridqa.py --n 20 --seed 42 --out eval/golden_hybridqa_draft.yaml

Fetches `dev.traced.json`, samples N multi-hop questions (anchor table + ≥1
linked passage), and emits `eval/golden_hybridqa_draft.yaml`. Each entry:

```yaml
- question: "..."
  expected_answer: "..."
  expected_sources: ["anchor table title", "linked passage title", ...]
  out_of_corpus: false
  multihop: true
  table_id: "..."
```

HybridQA's `answer-node` field is distant supervision — it marks every cell
and linked passage where the answer *string* occurs, not the evidence a
reader would use. Taken raw it produces sources that are not evidence at all
(answer "Gothic Revival" traced to the passage "Renaissance Revival
architecture", which mentions Gothic Revival only to say it is something
else). The loader therefore keeps a question only when its nodes all point at
one passage, and that passage is the only one in the table containing the
answer — a sibling passage carrying the same string would make a defensible
citation score as a miss. Expect roughly a third of candidates to be dropped.

**Review before running** — the traces are filtered, the questions are not.
HybridQA phrases them against a table already on screen, so some identify
nothing on their own ("A 2009 title came out in what month ?", "the older
player between number 9 and number 10"), and a few carry a false premise.
Those are unanswerable by retrieval and belong in no golden set: drop them.
The committed draft has been through this pass — 100 sampled, 16 dropped.

The draft also carries 28 hand-written `out_of_corpus: true` probes (a quarter
of 112). Without them `refusal_accuracy` measures nothing on this collection:
every remaining case is answerable, so the score is just "never refused". Most
probes are near misses — an entity the corpus *does* hold, asked for a fact its
passage never states ("What is the average annual rainfall in Multan ?") —
because the ingested passages are Wikipedia lead sections and stop there. Each
was checked against the ingested text. `load_hybridqa.py` cannot regenerate
them, so keep them when you re-draft.

Two hand-written blocks raise the ceiling, because the sampled entries are all
the same shape — one table row, one passage, answer copied out:

- **Hard cases** (6, in-corpus). Two rows of one table compared *before* any
  passage is read, a filter over a column, arithmetic across rows, one
  three-source chain, and one question whose two documents hang off different
  tables — nothing links them but the question.
- **Adversarial refusals** (7). The corpus holds a near miss for each: the
  Zürich agglomeration's municipality count is absent but Bern's "36" sits in
  the neighbouring passage; Calgary's 2018 passenger figure is absent but 2017
  is right there; the Canada Basin's average depth is absent but the sentence
  naming it gives the Amerasia Basin's. Retrieval succeeds and the answer still
  has to be a refusal, which is the failure users actually notice.

### 2. Ingest the reachable subgraph

    .venv/bin/python eval/ingest_hybridqa.py --golden eval/golden_hybridqa_draft.yaml

Fetches only the tables + passages named in the draft, chunks them
(`filename` = source title), embeds, and upserts into a **separate** Qdrant
collection `hybridqa` — the app's `documents` collection is untouched.

### 3. Run

    .venv/bin/python eval/run_eval.py --collection hybridqa --golden eval/golden_hybridqa_draft.yaml --agentic

The report adds `multi_hop_citation_accuracy`: over `multihop` entries, the
fraction whose answer cited **every** `expected_sources` title (vs
`citation_accuracy`, which only requires any one). This is the deterministic
signal for the multi-hop capability.

Citation matching is canonicalized so a sub-article citation counts as its
parent: Wikipedia tables link both a parent page and its specific sub-pages
(e.g. `Alpine skiing at the 1988 Winter Olympics` and `… – Men's super-G`),
and retrieval may rank either. Both are the correct evidence, so
`multi_hop_citation_accuracy` accepts a word-boundary prefix match.

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