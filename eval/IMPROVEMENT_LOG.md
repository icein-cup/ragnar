# RAGnar RAGAS Improvement Log

Tracking iterative improvements to RAGnar's RAGAS and deterministic eval scores.
All changes on branch `iteration_ai`. Model: `qwen2.5:7b`. Benchmark: HybridQA (125 cases).

## Baseline (20260824-132940.json — floors at 0.0, agentic, qwen2.5:7b)

| Metric | Score | Notes |
|---|---|---|
| answer_coverage | 0.49 | #1 problem — half of answerable questions wrong |
| answer_accuracy | 0.71 | When it answers, right 71% of time |
| refusal_accuracy | 0.76 | 24% of refuse/answer decisions wrong |
| citation_accuracy | 0.67 | Cites right source 2/3 of time |
| citation_precision | 0.26 | 74% of citations are noise |
| multi_hop_citation_accuracy | 0.52 | Half of multi-hop answers miss a source |

RAGAS (20-case subset only — needs full run):

| Metric | Score |
|---|---|
| faithfulness | 0.70 |
| answer_relevancy | 0.76 |
| answer_correctness | 0.33 |
| answer_similarity | 0.51 |
| context_precision | 0.58 |
| context_recall | 0.80 |
| no_invented_numbers | 0.95 |

### Failure modes (from 125-case analysis)
- 28/90 in-corpus questions refused (over-refusal)
- 18/62 answered questions got wrong answers (misreading)
- 2/35 out-of-corpus questions NOT refused (hallucination)
- Citation noise: avg 5-16 chunks cited, ~1-3 actually used

---

## Iteration Log

### Iteration 1 — Baseline re-run + parallel improvements (in progress)

**Date**: 2026-08-24

**Changes in flight**:
- Track A: Improved SYSTEM_PROMPT (reduce over-refusal, add multi-hop synthesis guidance)
- Track B: Citation pruning (reduce noise from 0.26 precision without losing source recall)
- Track C: Floor calibration (find optimal score_floor/vector_floor from score distributions)
- Fresh baseline eval running with real floors (0.55/0.42) on full 125-case set

**Status**: Code committed (da922ea), awaiting eval re-run

**Results**: Pending — baseline eval with real floors still running (~55 min)

### Floor calibration findings (Track C)
- Analyzed 743 chunks from the floors-at-0.0 report (140 expected, 603 unexpected)
- Rerank scores: expected median 0.678, unexpected median 0.513
- Vector scores: expected median 0.522, unexpected median 0.444
- OR semantics (current): best floor combo (0.58/0.38) filters only 66/603 noise chunks (11%)
- AND semantics: barely better, loses expected chunks at higher thresholds
- Conclusion: floor calibration alone cannot fix citation noise. Answer-overlap pruning (Track B) is the right approach.