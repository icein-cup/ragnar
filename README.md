<div align="center">

# RAGnar

**A local RAG chat tool for a small team.**

It answers from your documents and nothing else — with citations, and a refusal
when the answer isn't in there.

</div>

---

## The name

**RAGnar** — RAG + Ragnar

| | |
|---|---|
| **RAG** | retrieval-augmented generation — the answer is built from retrieved text, not from what a model happens to remember |
| **nar** | it reads like a Norse name, and it stuck |

*Only what the documents say.* That is the entire product. A question goes in, and
what comes back is either something the corpus actually supports — with the file and
page to check it against — or an admission that it isn't there.

---

## What it does

You drag a PDF into the sidebar and ask a question. In between, RAGnar:

1. **Parses the file** with Docling, keeping page and sheet provenance. OCR stays off
   by default and only kicks in when a page yields fewer than 50 characters — a
   scanned document still works, without making every native-text PDF pay for it.
2. **Chunks it structurally** — grouped by heading, never split across a page
   boundary. Tables are chunked by row group with the header row repeated in each
   one, so no chunk is a headerless fragment nobody can read. A semantic strategy is
   also selectable, cutting on meaning instead of headings — better suited to
   scanned or heading-poor documents, worth comparing on your own corpus.
3. **Embeds and stores** — BGE-M3 vectors in Qdrant.
4. **Retrieves and narrows** — 30 candidates, reranked by a cross-encoder down to 10,
   then measured against a similarity floor.
5. **Works the question, when one pass isn't enough** — the query is rewritten for
   retrieval, fanned out into variants, and followed up on where excerpts leave a gap.
   Strong first-pass results skip all of it. Each stage is a checkbox in Settings.
6. **Answers, or refuses** — the model sees only the retrieved excerpts, and the
   citations are assembled from those chunks rather than from anything the model wrote.

Every model in that chain runs on your own machine.

---

## What it looks like

A browser tab at `localhost:8501`. One chat, three collapsed panels, nothing else:

```
┌───────────────────┬────────────────────────────────────────────────┐
│  RAGnar           │                                                │
│  local RAG chat   │   what is the notice period?                   │
│                   │                                                │
│  ▸ Settings       │   Three months, counted from the end of the    │
│  ▸ Documents      │   calendar month in which notice is given.     │
│  ▸ Chats          │                                                │
│                   │   ▾ Sources                                    │
│                   │       contract.pdf, p. 12                      │
│                   │       contract.pdf, p. 13                      │
│                   │                                                │
│                   │   [ Ask about your documents             ]     │
└───────────────────┴────────────────────────────────────────────────┘
```

**Documents** takes the upload and shows what has been ingested. **Chats** keeps past
conversations. **Settings** exposes the knobs that are worth turning — model,
temperature, reranker, similarity floor, chunking strategy, chunk size.

Ingestion runs in the background with an estimated time remaining, so a slow scan of a
200-page PDF doesn't block the chat you're already having.

---

## Why refusing is the feature

Most document chatbots will answer anything. Ask about a contract clause that isn't in
the corpus and you get a fluent paragraph assembled from the model's general knowledge —
indistinguishable, in tone, from the answers that are actually grounded.

RAGnar decides whether it *can* answer before the model is involved:

```
question
   └─▶ embed ─▶ retrieve 30 ─▶ rerank ─▶ top 10
                                          │
   rerank score below 0.55 AND ──yes──▶ refuse, and list related documents
   vector score below 0.42?               │
                                          │
     aggregation words + mostly ──yes──▶ refuse, and name the file and sheet
     table chunks?                        │
                                          no
                                          ▼
                            answer, cited from the chunks actually used
```

A chunk only needs to clear *one* of the two floors. The reranker scores
markdown table rows as near-neutral (~0.50) against natural-language
questions regardless of relevance, so the raw embedding-similarity score is
a second signal that rescues genuinely relevant tabular content the
reranker has no opinion on.

Three things follow:

- **The refusal is deterministic.** Below the floor the LLM is never called at all, so
  there is no prompt to talk it out of and no temperature setting that makes it guess.
- **Citations cannot be invented.** They are built from the metadata of retrieved
  chunks, never parsed out of the model's prose — the model cannot cite a document that
  was not retrieved.
- **The one question RAG structurally cannot answer is refused by name.** "What's the
  total?" against a spreadsheet asks for arithmetic across a whole table when retrieval
  only ever sees a handful of rows. Two signals have to agree — aggregation wording
  *and* a majority of table chunks — before it fires, so ordinary questions about
  tables still get answered. The wording list covers English and Polish, including
  Polish superlatives in their inflected forms (`największa`, `najwyższy`).

What you get instead of a wrong number:

> This looks like a question that requires calculating across a whole table. I can only
> read individual rows, so any total I gave you could be wrong.
>
> The relevant data is in: budget.xlsx (sheet 2025)

---

## Privacy

Not a policy — a property of how it is built.

- The LLM runs on your machine, through Ollama.
- Embeddings and reranking run on your machine.
- The vector store is a container on your machine.
- Documents are written to `./data` and never leave it.

Two containers are defined, `app` and `qdrant`, and `docker compose config` is the
whole story. Answering a question uses no account, no API key, and no per-token bill.

The one exception is opt-in and eval-only: if you fill `RAGAS_JUDGE_API_KEY` in `.env`,
compose forwards it into the `app` container for `eval/run_ragas.py`. No code path in
the app itself reads it — only the eval harness does, and it runs against a hand-written
golden set, never your corpus, unless you point it there yourself. Leave the variable
empty and no key exists anywhere.

The one network access in the app itself is HuggingFace downloading the reranker
weights, once. `HF_TOKEN` is optional and only raises the rate limit while that happens.

---

## Where things land

Plain files on disk, next to the code:

```
data/
├── inbox/            ← uploads land here first, stamped `name.<hash8>.ext`
├── originals/        ← the file as you gave it
├── converted/        ← Docling output, cached by content hash
├── registry.db       ← what has been ingested, and how it went
└── chats.db          ← past conversations

qdrant_storage        ← a Docker volume holding the vectors
```

Converted documents are cached by content hash, so re-uploading the same file costs
nothing and a chunking change can be replayed without re-parsing — the parsed blocks
are cached alongside the markdown (`converted/{doc_id}.blocks.json`) and reused. The
vectors are the one thing that would have to be rebuilt from scratch — see Known gaps.

Identity is the content hash, never the filename. Both `inbox/` and `originals/` stamp
the short hash into the name, so two unrelated documents that happen to both be called
`report.pdf` stay two documents. Citations still show the name you uploaded.

---

## Install

Ollama has to run natively on the host rather than in a container, because Docker on
macOS cannot reach the GPU:

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3

cp .env.example .env      # optional: HF_TOKEN silences a rate-limit warning

docker compose up -d --build
```

Open `http://localhost:8501`, upload a document, ask a question.

Both containers use `restart: unless-stopped`. Docker Desktop under memory pressure will
SIGKILL the app, and without that line it stays dead until you notice — but a deliberate
`docker compose stop` is still respected.

### The reranker wants the host too

The same GPU constraint that keeps Ollama on the host applies to the cross-encoder
reranker, and it is the most expensive thing in the pipeline. Measured on an M5 Pro,
30 candidates per rerank:

| Where it runs | Per rerank |
|---|---|
| In the container, CPU | 10.34s |
| In the container, CPU + ONNX export | 8.57s |
| On the host, CPU | 6.58s |
| **On the host, Metal GPU** | **1.48s** |

The agentic path pays that once per generated query, so a question that fans out to
seven phrasings spends over a minute reranking alone. Starting the host service is
optional but worth roughly **7x**:

Those figures assume ordinary chunks. A cross-encoder pads every pair in a batch to
the longest sequence in it, so **one oversized chunk makes all 30 candidates cost as
if every one were that long** — the same batch takes 1.74s with a 1750-char longest
chunk and 9.00s with a 6648-char one. `retrieval/reranker.py` caps input at 512
tokens (`RERANKER_MAX_LENGTH`) to keep that bounded; on this corpus that truncates
5 chunks out of 2846 and leaves the rest scored identically.

```bash
.venv/bin/python retrieval/rerank_server.py     # leave running; loads on Metal
```

`docker-compose.yml` already points `RERANKER_URL` at it. With the service down, set
`RERANKER_URL=` (empty) to score in-container instead — otherwise reranking fails
loudly rather than silently reverting to the slow path, which on an unattended eval
sweep is the difference between one failed run and a whole experiment quietly running
at seven times its budgeted latency.

---

## Settings

The panel exposes what is worth changing per question; `config.yaml` holds the defaults.

| Setting | Default | What it changes |
|---|---|---|
| Model | `qwen2.5:7b` | Which local model writes the answer |
| Temperature | low | Higher wanders further from the excerpts |
| Reranker | on | Off is faster and noticeably less precise |
| Similarity floor | `0.55` | Rerank score below this AND vector floor below its own → refuse |
| Vector floor | `0.42` | Second, more lenient check on raw embedding similarity |
| Chunk size | 350 tokens | Target size per chunk, 50-token overlap |
| Table rows per group | 10 | Rows per table chunk, header repeated in each |
| Query rewriting | on | Rewrites the question for retrieval before searching |
| Multi-query retrieval | on | Searches several phrasings in parallel, fuses the hits |
| Multi-hop reasoning | on | Follows up when the excerpts leave a gap, up to 3 hops |
| Self-correction | on | Grades a draft answer and re-retrieves if it falls short |

The four agentic toggles each cost at least one extra LLM call per question, and they
compound — a question that misses the fast path can spend six or more round-trips
before a word is streamed. On a 3B local model that is the difference between a
snappy answer and a slow one. Turn them off to feel the floor of the pipeline.

`agentic.latency_budget_s` (25s) is what stops that compounding from running away.
Past it no *new* expansion starts — no fan-out, no further hop, no self-correction —
and the pipeline answers with what it already retrieved. Work already done is kept
and a hit is never turned into a refusal. It exists because no parameter value can
give you a ceiling: tuning shifts a distribution, only a clock bounds a tail. 25
rather than 35 because answer generation still runs after it; that ~10s difference is
a measured margin, not a guarantee, since nothing yet bounds the generation call
itself. Set `0` to disable.

The floors are the ones to understand before touching. Both are **stopgaps, not a
calibration**: on a real corpus, out-of-corpus questions scored 0.50–0.503 on the
reranker and relevant prose scored 0.578 and up, so 0.55 sits in that gap with margin
either side. That is five data points, not a golden set. A chunk is refused only when
*both* floors miss — lower either one and refusals turn into confident guesses.

Worse than hand-tuned, in fact. `0.55` was originally read off an `eval/run_eval.py
--calibrate` sweep that could not have produced a signal: the harness left the vector
floor at `0.0`, and because the two floors are OR'd, every result cleared the gate at
every swept value. That bug is fixed, but the number predates the fix and has not been
re-derived. Treat both floors as placeholders until you re-run the sweep on your own
corpus — see `eval/EVAL_README.md`.

---

## What's in an answer

- **The answer**, streamed, written only from the retrieved excerpts and in the language
  of the question — a Polish question against an English contract is answered in Polish.
- **Sources**, an expander listing every chunk that fed the answer:
  `contract.pdf, p. 12` · `budget.xlsx, sheet 2025` · `notes.docx`
- **Or a refusal.** "I could not find anything relevant in the indexed documents,"
  followed by *Related documents you might check* — the near-misses, so a bad question
  still points somewhere useful.

---

## Supported formats

`.pdf` · `.xlsx` · `.docx`

---

## Configuration

`config.yaml` holds the pipeline defaults:

| Key | Default | Notes |
|---|---|---|
| `models.llm` | `qwen2.5:7b` | Answer generation, via Ollama |
| `models.embedding` | `bge-m3` | 1024-dimensional vectors |
| `models.reranker` | `BAAI/bge-reranker-v2-m3` | Cross-encoder, downloaded once |
| `chunking.target_tokens` | `350` | 50-token overlap |
| `chunking.table_rows_per_group` | `10` | Header repeated per group |
| `retrieval.candidates` | `30` | Fetched before reranking |
| `retrieval.top_k` | `10` | Kept after reranking |
| `retrieval.score_floor` | `0.55` | Rerank-score floor |
| `retrieval.vector_floor` | `0.42` | Vector-similarity floor — either clearing its own floor keeps a chunk |

Environment (`.env`):

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | Yes | `http://host.docker.internal:11434` — Ollama on the host |
| `QDRANT_URL` | Yes | `http://qdrant:6333` — the sibling container |
| `RERANKER_URL` | No | `http://host.docker.internal:8007` — the host reranker service (see Install). Set empty to score in-container on CPU instead |
| `RERANKER_MAX_LENGTH` | No | `512`. Token cap per (query, chunk) pair. Bounds rerank cost, which a batch's longest chunk otherwise sets for every candidate in it |
| `HF_TOKEN` | No | Raises the HuggingFace rate limit while the reranker downloads |
| `FILE_SERVER_HOST` | No | Bind address for the archive file server. `0.0.0.0` by default, which is required under Docker — set `127.0.0.1` when running the app natively |
| `RAGAS_JUDGE_BASE_URL` | No | Eval only. OpenAI-compatible judge endpoint |
| `RAGAS_JUDGE_API_KEY` | No | Eval only. Leave empty and no key exists anywhere |
| `RAGAS_JUDGE_MODEL` | No | Eval only. Judge model name |

Citations open the archived original in a browser tab, served over HTTP on port 8510.
Compose publishes that as `127.0.0.1:8510`, so only the host reaches it — **that
publish is the only thing keeping the archive off your network.** The server has no
authentication. Run the app outside Docker and you must set `FILE_SERVER_HOST`
yourself.

---

## Known gaps

Tracked rather than glossed over:

- **Excel coverage does not run by default.** `tests/fixtures/sample.xlsx` and
  `tests/test_parser_xlsx.py` exist, but they are marked `integration` and need real
  Docling — so a normal `pytest` run proves nothing about `.xlsx`.
- **Neither floor has been validly calibrated.** The sweep that produced `0.55` was
  broken (see Settings); the harness is fixed but the number has not been re-derived,
  and the golden set is 5 cases against a fixture. See `eval/EVAL_README.md`.
- **The eval harness measures less than the app does.** `run_eval.py` defaults to the
  bare retrieval path; the UI always runs the agentic one. Pass `--agentic` to compare
  like for like — it costs several LLM calls per case.
- **No recovery path if the vector store is lost.** Re-embedding from the converted
  document cache (the parsed blocks, not just the markdown) would need a "rebuild all"
  action wired up in the UI; the cache itself exists but nothing drives it end to end.

The test suite is two halves. `pytest` runs ~200 unit tests against fakes in a couple
of seconds. The tests that need real Ollama, Qdrant or Docling are marked `integration`
and **skipped unless you ask for them**:

```bash
docker compose exec app python -m pytest              # fast, fakes only
docker compose exec app python -m pytest --run-integration   # the real stack
```

A green default run is not evidence the integration path works.

---

## Requirements

Docker Desktop, and [Ollama](https://ollama.com/) running natively on the host with
`qwen2.5:7b` and `bge-m3` pulled. Apple Silicon is the tested configuration; the
reranker adds a one-time download on first run.

Optionally, `retrieval/rerank_server.py` running on the host as well — the same GPU
constraint that keeps Ollama out of the container applies to the reranker, and it is
worth ~7x on Apple Silicon (see Install). The pipeline works without it, on CPU.

`eval/` is deliberately isolated from the running app and never imported by it — the
evaluation harness cannot change the behaviour it is measuring.
