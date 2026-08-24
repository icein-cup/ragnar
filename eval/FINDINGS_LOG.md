# RAGnar Findings Log

Running log of discoveries, patterns, and decisions. Updated each iteration.

## 2026-08-24 — Baseline analysis (125-case HybridQA, qwen2.5:7b, floors 0.0)

### Failure category breakdown (90 in-corpus cases)

| Category | Count | % | Root cause |
|---|---|---|---|
| CITATION_NOISE | 39 | 43% | Multi-query/multi-hop fusion piles 5-16 chunks, all cited |
| RETRIEVAL_MISS | 22 | 24% | Expected source not retrieved at all |
| OVER_REFUSAL | 16 | 18% | Model says NO_ANSWER when answer IS in contexts |
| WRONG_ANSWER | 8 | 9% | Model answers but misreads retrieved text |
| MULTI_HOP_MISS | 4 | 4% | Answer correct but not all sources cited |
| CORRECT | 1 | 1% | Fully correct (answer + citations) |

### Out-of-corpus
- 33/35 correctly refused
- 2 hallucinated: Canada Basin depth (fabricated "13,000 ft"), Danish supermarket (picked "Kiwi" from irrelevant contexts)

### Key patterns
- 43/44 correct answers have citation precision < 0.5 — citation noise is universal
- 88/90 in-corpus cases are multi-hop; failure rate 98.9%
- Population questions fail across all categories
- Model often "thinks out loud" instead of stating answer directly (e.g. "To answer this, we need to identify...")

### Score distribution (743 chunks, floors at 0.0)
- Expected chunks: rerank median 0.678, vector median 0.522
- Unexpected chunks: rerank median 0.513, vector median 0.444
- OR semantics: best floor combo filters only 11% of noise
- Conclusion: floors cannot fix citation noise. Answer-overlap pruning is the correct approach.

## 2026-08-24 — Iteration 1 changes (commit da922ea)

### Track A: Answer prompt (generation/prompts.py)
- Added 4-step reasoning process: identify question → check every excerpt → synthesize across excerpts → decide
- Stronger anti-refusal: "Refusing is the last resort, not the first"
- Explicit multi-hop synthesis guidance
- NO_ANSWER sentinel contract preserved
- Expected impact: reduce over-refusal (16→?), improve wrong-answer rate (8→?)

### Track B: Citation pruning (generation/answerer.py)
- Answer-overlap filtering: only cite chunks with ≥2 content words overlapping answer
- Stopword-filtered tokenization (English only — Polish stopwords NOT included, potential gap)
- Safe default: short answers skip pruning entirely
- Applied in answer(), answer_async(), ground_async(), eval harness
- Expected impact: citation precision 0.26→? (should improve significantly), citation_accuracy must not drop

### Track C: Floor calibration
- Floors at 0.0/0.42 with OR semantics only filter 11% of noise
- Reranker neutral scores (0.5 for 44% of needed chunks) make threshold-based pruning ineffective
- No code change — conclusion documented

### Pending verification
- Baseline eval with real floors (0.55/0.42) running — ~50 min ETA
- Re-eval with new prompt + pruning needed after baseline completes
- RAGAS on full 125-case set (previous was only 20 cases)
- Code review subagent reviewing diffs
- RAG consultant subagent proposing next improvements

### Open questions
- Will the 4-step reasoning in SYSTEM_PROMPT cause the model to output reasoning steps in its answer? (qwen2.5:7b may not follow "work through steps internally")
- Will citation pruning hurt multi_hop_citation_accuracy? (If answer doesn't contain words from a needed source's chunk, that source gets pruned)
- Should Polish stopwords be added to the pruning filter?
- Is the 20-case RAGAS run too small to be meaningful? (Yes — need full 125-case run)