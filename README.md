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
4. **Retrieves and narrows** — 25 candidates, reranked by a cross-encoder down to 5,
   then measured against a similarity floor.
5. **Answers, or refuses** — the model sees only the retrieved excerpts, and the
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
   └─▶ embed ─▶ retrieve 25 ─▶ rerank ─▶ top 5
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
  tables still get answered.

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

Two containers are defined, `app` and `qdrant`, and neither is given a key to anything
external — `docker compose config` is the whole story. There is no account, no API key,
and no per-token bill.

The one network access in the whole project is HuggingFace downloading the reranker
weights, once. `HF_TOKEN` is optional and only raises the rate limit while that happens.

---

## Where things land

Plain files on disk, next to the code:

```
data/
├── inbox/            ← uploads land here first
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

---

## Install

Ollama has to run natively on the host rather than in a container, because Docker on
macOS cannot reach the GPU:

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3

cp .env.example .env      # optional: HF_TOKEN silences a rate-limit warning

docker compose up -d --build
```

Open `http://localhost:8501`, upload a document, ask a question.

Both containers use `restart: unless-stopped`. Docker Desktop under memory pressure will
SIGKILL the app, and without that line it stays dead until you notice — but a deliberate
`docker compose stop` is still respected.

---

## Settings

The panel exposes what is worth changing per question; `config.yaml` holds the defaults.

| Setting | Default | What it changes |
|---|---|---|
| Model | `qwen2.5:3b` | Which local model writes the answer |
| Temperature | low | Higher wanders further from the excerpts |
| Reranker | on | Off is faster and noticeably less precise |
| Similarity floor | `0.55` | Rerank score below this AND vector floor below its own → refuse |
| Vector floor | `0.42` | Second, more lenient check on raw embedding similarity |
| Chunk size | 500 tokens | Target size per chunk, 50-token overlap |
| Table rows per group | 20 | Rows per table chunk, header repeated in each |

The floors are the ones to understand before touching. Both are **stopgaps, not a
calibration**: on a real corpus, out-of-corpus questions scored 0.50–0.503 on the
reranker and relevant prose scored 0.578 and up, so 0.55 sits in that gap with margin
either side. That is five data points, not a golden set. A chunk is refused only when
*both* floors miss — lower either one and refusals turn into confident guesses.

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
| `models.llm` | `qwen2.5:3b` | Answer generation, via Ollama |
| `models.embedding` | `bge-m3` | 1024-dimensional vectors |
| `models.reranker` | `BAAI/bge-reranker-v2-m3` | Cross-encoder, downloaded once |
| `chunking.target_tokens` | `500` | 50-token overlap |
| `chunking.table_rows_per_group` | `20` | Header repeated per group |
| `retrieval.candidates` | `25` | Fetched before reranking |
| `retrieval.top_k` | `5` | Kept after reranking |
| `retrieval.score_floor` | `0.55` | Rerank-score floor |
| `retrieval.vector_floor` | `0.42` | Vector-similarity floor — either clearing its own floor keeps a chunk |

Environment (`.env`):

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | Yes | `http://host.docker.internal:11434` — Ollama on the host |
| `QDRANT_URL` | Yes | `http://qdrant:6333` — the sibling container |
| `HF_TOKEN` | No | Raises the HuggingFace rate limit while the reranker downloads |

---

## Known gaps

Tracked rather than glossed over:

- **Excel is designed for but under-tested.** Citations carry a sheet field and tables
  get their own chunking, but no `.xlsx` fixture exists in the test suite yet.
- **Both floors are hand-tuned**, not calibrated — see `eval/README.md`. They
  need a much larger golden set before the numbers deserve trust.
- **No recovery path if the vector store is lost.** Re-embedding from the converted
  document cache (the parsed blocks, not just the markdown) would need a "rebuild all"
  action wired up in the UI; the cache itself exists but nothing drives it end to end.

The test suite runs against real Ollama, Qdrant and Docling rather than mocks, which is
why it is slow and why it catches integration breakage that mocks would hide.

---

## Requirements

Docker Desktop, and [Ollama](https://ollama.com/) running natively on the host with
`qwen2.5:3b` and `bge-m3` pulled. Apple Silicon is the tested configuration; the
reranker adds a one-time download on first run.

`eval/` is deliberately isolated from the running app and never imported by it — the
evaluation harness cannot change the behaviour it is measuring.
