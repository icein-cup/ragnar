# Citation Pruning Design Note

## Problem

Citation precision is 0.26 — 74% of cited sources are irrelevant noise.
The cause: `AgenticSearch` fuses results from multiple queries and hops,
returning 10–16 chunks. `citation_labels(results)` cites every one of them.
The answer typically draws from 1–3 chunks but cites 5–16.

## Constraints

- **citation_accuracy (0.67)** must not drop: at least one expected source
  must appear in citations for each answerable question.
- **multi_hop_citation_accuracy (0.52)** must not drop: ALL expected sources
  must appear for multi-hop questions.
- **Reranker-based pruning already failed** (answerer.py lines 157–164):
  bge-reranker-v2-m3 gives 44% of needed chunks the same neutral 0.500
  score as irrelevant ones. Every threshold that lifted precision from
  0.26 to ~0.5 dropped source recall from 0.74 to ~0.42, halving
  multi_hop_citation_accuracy.

## Approaches Considered

### A) LLM-based citation filtering (post-generation)
Ask the model which excerpts it actually used. Adds a second LLM call per
answer — the same cost problem that kept the grounding check off the
critical path. Also requires prompt engineering and is hard to verify.

### B) Reranker score-based filtering
Already tried and measured. Failed because the reranker gives needed chunks
neutral scores identical to irrelevant ones. Rejected.

### C) Reduce top_k from 5 to 3
Reduces the context set sent to the LLM, which could hurt answer quality.
Multi-hop questions need multiple sources — capping at 3 risks dropping
a required source before the model ever sees it. Too risky for
multi_hop_citation_accuracy.

### D) Fusion deduplication improvements
`_fuse_results` already deduplicates by `doc_id:chunk_index`. Additional
dedup (e.g. by text similarity) would help marginally but doesn't address
the core issue: the model draws from 1–3 chunks regardless of how many
are retrieved.

### E) Answer-overlap filtering (CHOSEN)
After the answer is generated, filter citations to only those chunks whose
text has meaningful word overlap with the answer. This is a post-hoc filter
on the citation list only — it does NOT touch the result set sent to the
LLM, so answer quality and multi-hop source retrieval are preserved.

## Chosen Approach: Answer-Overlap Filtering (E)

### How it works

1. The answer is generated normally from all retrieved chunks.
2. `prune_citations(answer_text, results)` extracts content words
   (alphanumeric tokens minus stopwords) from both the answer and each
   chunk's text.
3. A chunk is cited only if at least **2 distinct content words** from its
   text appear in the answer.
4. Results are deduplicated by citation label in first-seen order.

### Why this is safe

- **Does not reduce context**: The LLM sees all retrieved chunks. Answer
  quality is unchanged. Multi-hop questions still get all their sources in
  the prompt.
- **Does not rely on reranker scores**: sidesteps the failure mode that
  killed approach B.
- **Conservative threshold**: 2 content words is a low bar. The goal is to
  drop chunks with ZERO content overlap (pure fusion noise from multi-query/
  multi-hop), not to aggressively trim borderline cases.
- **Safe default for short answers**: If the answer has fewer than 2 content
  words, no filtering is applied — all results are kept. This preserves
  citation_accuracy for very short answers.
- **No extra LLM calls**: pure string matching, O(n) in chunks.

### Implementation

- `prune_citations(answer_text, results)` in `generation/answerer.py`
- `citation_labels(results, answer_text=None)` — pruning is opt-in via the
  optional `answer_text` parameter. When `None` (backward-compatible), no
  pruning is applied.
- `build_citations(results, answer_text=None)` — same opt-in pattern for
  the UI's rich citations.
- `Answerer.answer()` passes `text` to `citation_labels`.
- `Answerer.answer_async()` passes `text` to `citation_labels`.
- `Answerer.ground_async()` passes `text` to `citation_labels`.
- `eval/run_eval.py:resolve_answer()` passes the stripped draft to
  `citation_labels`.
- `ui/app.py` passes `text` to `build_citations`.

### Tokenization

Uses the same regex as `eval/metrics.py._words`:
`re.findall(r"[a-z0-9]+", text.lower())`, then removes a stopword list.
Stopwords are common function words (the, is, a, of, ...) that appear in
every chunk and every answer — matching on them would keep every citation.

### Threshold choice

`_MIN_OVERLAP_WORDS = 2` — a single shared word could be coincidence
(a common noun, a number), but two distinct content words appearing in both
the chunk and the answer is strong evidence the model drew from that chunk.
This is deliberately conservative: the goal is to drop chunks with zero
content overlap, not to aggressively trim borderline ones.

### Expected impact

- **Precision**: should improve significantly. Chunks retrieved by
  multi-query/multi-hop that the model never used will have zero content
  overlap with the answer and be dropped from citations.
- **citation_accuracy**: should be preserved. If the model used a chunk to
  write the answer, the answer will contain words from that chunk. The
  conservative threshold and safe-default-for-short-answers guard against
  over-pruning.
- **multi_hop_citation_accuracy**: should be preserved. Each required
  source that contributed to the answer will have word overlap with it.
  The context set is not reduced, so multi-hop retrieval still finds all
  sources.

### What could go wrong

1. **Paraphrasing**: If the model heavily paraphrases a chunk (uses synonyms
   but none of the original words), the citation could be dropped. This is
   unlikely with the conservative 2-word threshold — even paraphrased
   answers typically share proper nouns, numbers, and key terms.
2. **Table chunks**: Table data (numbers in cells) might have low word
   overlap if the answer describes the data in prose. Mitigated by the fact
   that numbers are kept in tokenization (`[a-z0-9]+`).
3. **Very short answers**: "Yes." or "No." have 0–1 content words. The safe
   default returns all results in this case.