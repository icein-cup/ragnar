# Model & prompt comparisons

Head-to-head tests run via `eval/replay_answer.py` / `eval/replay_gate.py` — one LLM call per case
over a saved report's already-retrieved contexts, not a full pipeline run. Cheap (minutes, not the
~30-80 min a full `run_eval.py --agentic` costs) because retrieval isn't repeated, only the stage
under test. Complements [report_table.py](report_table.py), which tracks full end-to-end runs; this
file tracks the cheaper A/B tests that never produce a `reports/*.json` of their own.

Every entry names its exact command so it can be re-run. When a change ships, log the result here
*before* moving on — this file exists because the alternative is these numbers living only in a chat
transcript.

---

## Model comparison (answering stage)

Same 20 in-corpus + 10 probes, same saved report, seed 42 — directly comparable across rows.

```
eval/replay_answer.py --report eval/reports/20260823-165853.json --prompt baseline \
  --model <model> --sample 20 --probes 10 --think <off|on|default>
```

| model | think | coverage | probe refusal | sec/call | resident |
|---|---|---|---|---|---|
| qwen2.5:3b | default | 0.42 *(40-case sample, not this 20-case one — not directly comparable)* | 0.55 | ~1s | 1.9 GB |
| qwen2.5:7b | default | **0.60** | 0.90 | ~3s | 6.6 GB |
| qwen2.5:14b | default | **0.60** | 0.90 | ~5s | 9.0 GB |
| Qwen3.8-27 (thinking) | default (native) | **0.70** | 0.90 | ~20s | 32 GB resident |
| Qwen3.8-27 | **false** | 0.05 | — | ~8s | 32 GB resident |

**Finding — forcing a thinking model not to think destroys it.** Qwen3.8-27 with `think: false`
refused 38 of 40 in-corpus questions with a bare `NO_ANSWER`; the 2 it did answer were both correct,
so it isn't incapable, it declines instead of reasoning. `config.yaml`'s `models.think` comment
carries this in full. Never set `think: false` on a model whose native mode is thinking.

**Finding — the 27B's real cost is 32 GB resident, not its 18.2 GB on-disk size.** Won't load on a
32 GB machine at all. `qwen2.5:7b` is the fallback for smaller hosts.

**Finding — 27B latency (~20s/call) already exceeds the production latency budget** established
2026-08-24 (see below) on its own, before any agentic multi-call overhead. Parking the 27B question
until the cheaper branches are exhausted.

**Finding — doubling parameters (7b→14b) bought nothing.** Identical coverage (0.60) and probe
refusal (0.90) to `qwen2.5:7b`, for ~1.7x the latency (~5s vs ~3s/call) and ~1.4x the resident memory
(9.0 GB vs 6.6 GB). Whatever the bottleneck is in this range, it isn't raw model capacity — the
sentinel prompt rewrite alone moved coverage more (0.60→0.62 on the 40-case sample) than doubling
parameters did. Reinforces the plan's prompt/retrieval-first ordering over a bigger-model default.

---

## Prompt comparison — the refusal sentinel (`SYSTEM_PROMPT`)

40 in-corpus + 20 probes, qwen2.5:7b, `eval/reports/20260823-165853.json`.

| prompt | answer_coverage | refusal_rate | note |
|---|---|---|---|
| original (regex-only refusal detection, no sentinel) | 0.60 | 0.85 | pre-sentinel baseline |
| sentinel v1 ("explain briefly why" on refusal) | 0.50 | 0.95 | **regression** — see below |
| **sentinel v2 (shipped)** — "answer whenever the excerpts contain the answer... only when they genuinely do not, say so plainly" | **0.62** | 0.95 | shipped in `SYSTEM_PROMPT` |

**Finding — giving a model a token for declining makes declining easier to reach for.** v1 asked the
model to justify a refusal ("explain briefly why"), which reads as an invitation to find a reason.
v2's explicit "answer whenever you can" bias is what pays for the sentinel — it isn't padding, it's
load-bearing. Full rationale is a comment above `SYSTEM_PROMPT` in
[generation/prompts.py](generation/prompts.py).

---

## Prompt comparison — bridge-question resolution (`hops` variant)

**Status: MEASURED (2026-08-24), post-fix.** Both blocking bugs (trailing-sentinel miss, the
refusal-counted-as-correct numerator bug) are fixed; report is the current post-Branch-C baseline
(`eval/reports/20260824-132940.json`). Same seed (42) so `baseline` and `hops` score the identical
40-case + 20-probe sample.

```
eval/replay_answer.py --prompt baseline --sample 40 --probes 20
eval/replay_answer.py --prompt hops     --sample 40 --probes 20
```

| prompt | answer_accuracy (40 in-corpus) | refusal_rate (20 probes) |
|---|---|---|
| baseline (shipped `SYSTEM_PROMPT`) | 0.42 | 0.85 |
| **hops** (bridge-resolution rule, no sentinel) | **0.65** | 0.80 |

**Deictic-only slice** — the population this prompt targets (regex: `\b(this\|that\|these\|those)\b`
or `the <1-5 words> (that\|which\|who\|whose)`; 40/89 in-corpus cases hit it, close to the 45/90
counted in the original finding). Same report, same two prompts, this slice only:

```
# ad hoc script, not checked in — reuses eval/replay_answer.py's PROMPTS/answer_with directly:
# filter report cases to the deictic regex, run "baseline" then "hops" over just those
```

| prompt | accuracy on 40 deictic cases |
|---|---|
| baseline | 0.33 |
| **hops** | **0.55** |

**Finding — a plain prompt swap recovers most of the deictic gap, at zero added latency.** 0.33→0.55
closes 22 of the 31-point gap to plain-phrasing questions (64% from the original finding), using the
exact same single generation call the pipeline already makes — no extra LLM call, no re-retrieval.
This makes Branch A's originally-designed conditional retry (detect deictic, fire a second generation
call) unnecessary: the win comes from the instruction itself, not from asking twice. **Simpler
design available:** swap the prompt outright rather than gating it by question type.

**Cost — refusal_rate slipped 0.85→0.80 (one more probe wrongly answered instead of refused).** n=20
probes, so that is one case; noisy, but real, and `hops` **drops the NO_ANSWER sentinel entirely** —
it falls back to the regex-only refusal check that worked here only because this model's phrasing
happened to match `REFUSAL_PATTERNS`.

**Merge attempt — the sentinel itself, not just the CoT scaffolding, eats most of the gain.** Tried
folding the bridge-resolution rule into the sentinel-carrying prompt two ways, same report/sample/
seed throughout:

| prompt | shape | accuracy (40 in-corpus) | deictic-only (40) | refusal (20 probes) |
|---|---|---|---|---|
| baseline (shipped) | sentinel + 4-step CoT | 0.42 | 0.33 | 0.85 |
| **hops** | short, no sentinel, no CoT | **0.65** | **0.55** | 0.80 |
| hops_sentinel | hops content + sentinel + 4-step CoT | 0.40 | 0.30 | 0.85 |
| hops_sentinel_minimal | hops content + sentinel, no CoT | 0.47 | — | 0.85 |

Adding the 4-step CoT block back costs ~7 points on its own (0.47→0.40) — consistent with the
`SYSTEM_PROMPT` comment's existing finding that a refusal token invites refusal, now showing the same
effect from the *reasoning scaffold* independently of the token. But even the **minimal** sentinel
addition — one line, no CoT — gives up 18 points versus `hops` outright (0.65→0.47) for a refusal
gain that is 1 case out of 20. The sentinel mechanism's cost on this model/task is larger than
previously measured on the older sentinel v1-vs-v2 test (experiment-results.md above), which never compared
against a no-sentinel baseline this strong.

**Follow-up — rewording (not just relocating) the sentinel instruction recovers most of the loss.**
`hops_sentinel_minimal` stated the refusal instruction as a flat, co-equal rule. `hops_sentinel_v2`
instead borrows two things the ORIGINAL sentinel v1-vs-v2 test already proved work
(`generation/prompts.py`'s `SYSTEM_PROMPT` comment): an explicit "default to answering, refusing is
the last resort" rule, and stating the NO_ANSWER instruction LAST, gated on "only if you have
genuinely checked" rather than offered as one option among several near the top.

| prompt | accuracy (40 in-corpus) | deictic-only (40) | refusal (20 probes) |
|---|---|---|---|
| baseline (shipped) | 0.42 | 0.33 | 0.85 |
| hops (no sentinel) | 0.65 | 0.55 | 0.80 |
| hops_sentinel (+ CoT) | 0.40 | 0.30 | 0.85 |
| hops_sentinel_minimal | 0.47 | — | 0.85 |
| **hops_sentinel_v2** | **0.55** | **0.42** | **0.85** |

`hops_sentinel_v2` dominates the shipped baseline on every measured axis — +13 points full accuracy,
+9 points deictic, refusal rate unchanged (still 17/20, sentinel contract fully intact). It recovers
9 of `hops`'s 13-point full-sample lead and 9 of its 22-point deictic lead over baseline, while
keeping the model-agnostic NO_ANSWER contract `hops` gives up.

**Two more rewording attempts, isolating just the refusal-instruction wording (position and
"default to answering" framing held constant at v2's):**

| variant | change from v2 | accuracy (40 in-corpus) | refusal (20 probes) |
|---|---|---|---|
| v3 | drops "briefly state what is missing" (hypothesis: asking to justify invites refusal, per the original v1 finding) | 0.45 | 0.85 |
| v4 | reframes NO_ANSWER as a parsing/formatting requirement rather than a content choice | 0.50 | 0.85 |

Both **underperform v2's 0.55** — the explanation request v3 removed turns out to help here (the
opposite of the original v1 finding, which tested a different instruction — "explain briefly why" —
in a different prompt shape; not the same lever). Two tries moving away from v2 both lost ground, so
**v2 stands as the best sentinel-preserving candidate found.**

**Further rounds, per explicit instruction to keep searching and also amend the general
(bridge-resolution) instruction, not just the sentinel wording:**

- `hops_sentinel_v5` (v2 + one light "read every excerpt carefully" line, no full CoT): **0.55/0.85**
  — identical to v2, the added line changed nothing measurable.
- `hops_example` (adds a worked example resolving a description to an entity; **no sentinel** — tests
  the general-prompt axis in isolation): **0.62/0.85** — nearly matches `hops`'s accuracy (0.65) while
  *beating* its refusal rate (0.85 vs 0.80), on the regex-only fallback alone.
- `hops_sentinel_v6` (hops_example's worked example + v2's sentinel wording, testing whether the two
  axes stack): **0.55/0.85** — identical to v2 and v5. **The example's gain evaporates once the
  sentinel is added back.**

**Three sentinel-preserving variants (v2, v5, v6) now land on the exact same 0.55/0.85** despite
different content changes — strong evidence of a real plateau for "hops-shaped content + this
sentinel-instruction shape" on this model, not sampling noise. Whatever the sentinel mechanism
suppresses, it caps accuracy around 0.55 regardless of what else is added. `hops_example` (no
sentinel) is now the best-performing candidate overall on this sample: higher accuracy than every
sentinel variant (0.62 vs 0.55) at the identical 0.85 refusal rate — the regex fallback matched the
sentinel's own rate exactly on this run.

**Every variant tried stays as a named, exact-text entry in
[eval/replay_answer.py](eval/replay_answer.py)'s `PROMPTS` dict — nothing is deleted, so any of them
can be re-run or promoted to `generation/prompts.py` at any time.** Full ranking:

| variant | accuracy | deictic | refusal | sentinel? | status |
|---|---|---|---|---|---|
| baseline (shipped) | 0.42 | 0.33 | 0.85 | yes | current production |
| hops | 0.65 | 0.55 | 0.80 | no | highest accuracy, no contract |
| hops_sentinel (+CoT) | 0.40 | 0.30 | 0.85 | yes | worst — CoT scaffold actively hurts |
| hops_sentinel_minimal | 0.47 | — | 0.85 | yes | baseline sentinel wording, no CoT |
| hops_sentinel_v2 | 0.55 | 0.42 | 0.85 | yes | plateau candidate 1 |
| hops_sentinel_v3 | 0.45 | — | 0.85 | yes | worse — dropping the explanation ask hurt |
| hops_sentinel_v4 | 0.50 | — | 0.85 | yes | worse — "parsing requirement" framing hurt |
| hops_sentinel_v5 | 0.55 | — | 0.85 | yes | plateau candidate 2 — ties v2 exactly |
| **hops_example** | **0.62** | — | **0.85** | **no** | **best overall — no sentinel** |
| hops_sentinel_v6 | 0.55 | — | 0.85 | yes | plateau candidate 3 — example + v2 combined, still ties |

**SHIPPED, then CONFIRMED to diverge sharply from the replay prediction (2026-08-25).**
`hops_example` was promoted to `SYSTEM_PROMPT` and run through the full 125-case agentic pipeline
(`eval/reports/20260825-152139.json`, same floors/collection/seed as the canonical baseline):

| metric | old baseline | new (full pipeline) | delta |
|---|---|---|---|
| answer_accuracy | 0.759 | 0.580 | **-0.178** |
| answer_coverage | 0.489 | 0.522 | +0.033 |
| refusal_accuracy | 0.736 | 0.888 | +0.152 |
| citation_accuracy | 0.622 | 0.811 | +0.189 |
| p90 latency | 53.5s | 43.6s | better |

**The 40-case replay's clean win (0.42→0.62 accuracy, no refusal cost) did not hold up on the full
pipeline.** `answer_coverage` — the honest headline metric, chosen specifically because it can't be
gamed by refusing more — moved only +3.3 points, inside the scorer's own documented 2-5 point slack
(see the plan file's "Do we have proper benchmarks?" section). That could be noise. Meanwhile
`answer_accuracy` (correct ÷ answered) dropped hard: the model answers more often and cites the right
source far more often (+18.9 points), but a much bigger share of those extra answers are wrong.

**Likely cause: the replay never touched the self-correction stage.** `eval/replay_answer.py` scores
one isolated generation call over a frozen, already-correct saved context. The full agentic pipeline
also uses `SYSTEM_PROMPT` to generate `_self_correct`'s draft answer
([retrieval/agentic.py](../retrieval/agentic.py)), which can get reused as the final answer
(`outcome.draft_answer`) without ever going through the replay's code path. The "resolve the bridge,
default to answering" instruction may be pushing confident-but-wrong bridge resolutions when
retrieval didn't actually surface the right entity — a failure mode a single-call replay over
guaranteed-correct contexts structurally cannot surface.

**Diagnosed (2026-08-25): NOT a self-correction/draft-reuse artifact.** `draft_reused` fired on
essentially 0 cases in the new run (1 in the old) — the hypothesis that `_self_correct`'s draft
generation was the mechanism is ruled out empirically.

The real mechanism, from the raw correct/wrong/refused counts (90 in-corpus cases each):

| | refused | correct | wrong |
|---|---|---|---|
| old | 32 | 44 | 14 |
| new | 9 | 47 | 34 |

Of the 23 cases that stopped being refused, only 3 became correct — 20 became wrong (a 13%/87%
conversion, not the reverse). Not concentrated on the slow/multi-hop path either — wrong answers land
mostly on fast-path in both runs (13/14 old, 30/34 new), so this isn't specifically about multi-hop
retrieval noise.

**Conclusion: the old prompt's caution wasn't just over-refusing easy bridge questions.** A
substantial share of its 32 refusals were on genuinely hard cases where declining was the safer,
more honest call than guessing. `hops_example` has no comparable caution mechanism (no sentinel, no
"only decide after checking every excerpt" discipline) — it now attempts those cases too, and gets
most of the attempts wrong. This is not a narrowly fixable bug in one code path; it is the prompt's
fundamental risk posture. Combined with `answer_coverage`'s barely-there +3.3 point move (inside the
scorer's known noise band) against a real, large `answer_accuracy` drop, **the honest read is that
this prompt is a net loss for a system where a wrong answer costs more than an honest refusal.**

**REVERTED (2026-08-25).** `generation/prompts.py`'s `SYSTEM_PROMPT` restored to the sentinel + 4-step
CoT version that produced the 0.759/0.489/0.736 baseline. `hops_example` and all other tested variants
remain in `eval/replay_answer.py`'s `PROMPTS` dict for any future attempt. **Lesson: a replay-only test
cannot see a prompt's effect on refusal calibration under real retrieval variance — test the full
pipeline before declaring a prompt change ready, not only at the final confirmation step.**

---

## Round 2 — bridge-resolution folded ADDITIVELY into the caution-preserving prompt

Applying the lesson: instead of replacing the reverted prompt, added the bridge-resolution instruction
(step 1) and two entity-precision rules from `hops`/`hops_example` directly into it — nothing removed,
sentinel/CoT/"default to answering" all intact. Tested on a full-pipeline run (not replay) over a
seeded 55-case subset (40 in-corpus + 15 probes, `eval/golden_hybridqa_draft.yaml` sampled with the
same seed as the replay harness) before touching the shipped state, per the lesson above.

```
# subset built with: rng.shuffle(in_corpus)[:40] + rng.shuffle(probes)[:15], seed 42
eval/reports/20260825-161533.json   (baseline, reverted prompt, on the subset)
eval/reports/20260825-163353.json   (candidate, bridge-resolution added, same subset)
```

| metric | baseline | candidate | delta |
|---|---|---|---|
| answer_accuracy | 0.565 | 0.565 | 0.000 |
| answer_coverage | 0.325 | 0.325 | 0.000 |
| refusal_accuracy | 0.655 | 0.655 | 0.000 |
| citation_accuracy | 0.550 | 0.550 | 0.000 |
| citation_precision | 0.500 | **0.602** | **+0.101** |
| mean latency | 19.5s | 16.5s | better |
| max latency | 199.4s | 106.3s | better |

**Not a bug — verified.** Only 13/55 answers are byte-identical text between the two runs (42/55
genuinely differ), so the prompt change is doing real work; it just landed on the same coarse
correct/wrong/refused classification for most individual cases on this sample. The candidate's answers
read as more direct and bridge-resolved on inspection (e.g. one case: baseline rambles through a
paraphrase, candidate answers "Robert Louis Stevenson" outright).

**No regression on any accuracy axis, a real citation-precision gain, and meaningfully faster** (likely
fewer/shorter follow-up hops when the model resolves the bridge in one pass instead of needing
self-correction to catch it). Full 125-case confirmation launched to verify this holds at scale before
calling it shipped.

**REJECTED — the full 125-case run contradicted the subset entirely** (`eval/reports/20260825-165051.json`
vs the canonical baseline `eval/reports/20260824-132940.json`):

| metric | old baseline | Round 2 (full 125) | delta |
|---|---|---|---|
| answer_accuracy | 0.759 | 0.560 | **-0.199** |
| answer_coverage | 0.489 | 0.311 | **-0.178** |
| refusal_accuracy | 0.736 | 0.656 | -0.080 |
| citation_accuracy | 0.622 | 0.522 | -0.100 |
| citation_precision | 0.265 | 0.503 | +0.238 |
| p90 / max latency | 53.5s / 144.9s | 58.6s / 182.7s | worse |

Worse than the shipped baseline on nearly every axis — worse even than Round 1's rejected
`hops_example` swap (which at least held coverage at 0.522). Only citation_precision improved, same as
the subset predicted.

**Root cause of the false signal: the 55-case subset was not representative.** Re-checking its own
numbers — the REVERTED (known-good) baseline scored only 0.325 coverage on that specific subset,
versus 0.489 on the full 125-case set. The subset had, by chance, sampled a harder-than-average slice.
A tied score between baseline and candidate *on a hard subset* said nothing about relative performance
on the full, more representative population — both prompts were depressed by the same hard sample, and
the subset was too small (only 40 of 90 in-corpus cases) to catch a regression concentrated in the
other 50.

**SECOND LESSON, layered on the first:** "test the full pipeline, not just replay" was necessary but
not sufficient — the pipeline also needs the FULL benchmark, not an arbitrary same-size-class sample.
A full-pipeline test on an unrepresentative subset can produce the same false confidence a replay test
did, just one level more expensive to discover. There is no cheap substitute for the complete 125-case
run before a prompt change touching refusal calibration is trusted. **Reverted (2026-08-25)** — see
`generation/prompts.py`'s history comment for the full record. Bridge-resolution content from both
rejected rounds is preserved verbatim in `eval/replay_answer.py`'s `PROMPTS` dict for any future
attempt, but the next one should go straight to a full 125-case run rather than any intermediate
shortcut.

---

## Retrieval width — `top_k` / `candidates` (Branch D)

Two cheap steps before committing to a full agentic run, per the lesson learned above (never trust
anything short of the full 125-case run for a final call, but no reason to pay for one before a
cheaper signal justifies it).

**Step 1 — retrieval-only recall** (embed + rerank, no generation, 89 in-corpus cases, `hybridqa`):

| config | answer recall | source recall |
|---|---|---|
| top_k=5 / candidates=25 (shipped) | 67/89 (0.75) | 78/89 (0.88) |
| top_k=8 / candidates=25 | 69/89 (0.78) | 79/89 (0.89) |
| top_k=10 / candidates=30 | 71/89 (0.80) | 80/89 (0.90) |

Monotonic gain, no cases lost at any tier.

**Step 2 — one generation call per case** (base `Search`, no agentic overhead, all 89 in-corpus + 35
probes, current `SYSTEM_PROMPT`):

| config | correct | wrong | probe refusal |
|---|---|---|---|
| top_k=5 | 24/89 (0.270) | 23/89 | 34/35 (0.971) |
| top_k=10 | 32/89 (0.360) | 22/89 | 31/35 (0.886) |

+9 points correct, wrong count flat-to-better, a modest probe-refusal cost (3 more of 35 wrongly
answered). Unlike the two rejected prompt rounds above, this doesn't touch refusal wording or
self-correction — it's a pure retrieval-width change, mechanically simpler and less prone to the
failure mode that burned Round 1/2.

**Staged in `config.yaml`** (`top_k: 5→10`, `candidates: 25→30`) but **not yet confirmed on the full
125-case agentic benchmark** — that run was started and stopped before completion. Given today's two
prompt-round lessons, this should not be treated as shipped until that confirmation actually runs.

---

## Gate comparison — `GROUNDING_PROMPT` accept rule

```
eval/replay_gate.py --report eval/reports/20260824-132940.json --model qwen2.5:7b --sample 25
```

| gate | bad answers caught | correct answers lost |
|---|---|---|
| qwen2.5:3b, no accept rule | 10/10 | 23/25 |
| qwen2.5:7b, no accept rule | 9/10 | 16/25 |
| **qwen2.5:7b, with accept rule (shipped in `GROUNDING_PROMPT`)** | 9/20 | **7/25** |

**Finding — stating the accept rule more than halved false rejections**, but not enough to gate on.
Losing 28% of correct answers to catch 45% of bad ones is a bad trade while over-refusal is already
the pipeline's largest failure. `_grounded` stays advisory (`answer_async`/`ground_async` only, off
the critical path) — see its docstring in [generation/answerer.py](generation/answerer.py).

---

## Findings that aren't A/B tests but inform the above

**Latency reality (2026-08-24, full 125-case run, `eval/reports/20260824-132940.json`):**

| | mean | median | p90 | max |
|---|---|---|---|---|
| all cases | 23.2s | 15.9s | 53.5s | 144.9s |
| fast path (82%) | 19.0s | — | — | — |
| slow path (18%, multi-hop) | 42.8s | — | — | 127.1s |

And it's fully blocking — the whole agentic pipeline runs inside a static `st.spinner()`
([ui/app.py:303](ui/app.py#L303)) with zero visible output until it returns. Production target
agreed 2026-08-24: p50 under ~6-8s, p90 under ~15-20s, nothing silent for more than a few seconds.
**Current p90 already fails that, before any accuracy change.** Any future comparison in this file
must report latency alongside accuracy — a coverage win that pushes p90 past target is not a win.

**Target revised 2026-08-26: max ≤35s — a hard ceiling on every case, not a percentile.**
Owner's call, superseding the p50 6-8s / p90 15-20s figures above. This is a stricter
constraint than a p90, and **the pipeline does not currently meet it**:

| Run | over 35s | max |
|---|---|---|
| `20260825-165051`, 125 cases, pre-move | **30/125 (24%)** | 182.7s — 5.2x over |
| `20260826-110031`, 20 cases, post-move | **1/20 (5%)** | 84.2s — 2.4x over |

Every violator is a high-fan-out case (7-13 generated queries), and most are out-of-corpus:
the search keeps generating queries hunting for material that does not exist, and each query
costs a full rerank. That is the tail, and the tail is now the binding metric.

**`max_s`, not `latency_p90`, is the number that decides a config.** Every report already
records it (`summary.latency.max_s`); comparisons in this file have been ranking on mean and
p90, which say nothing about a ceiling. A config with a better p90 and a worse max is a
regression under this target.

**Tuning alone cannot deliver this, and the sweep should not be expected to.** A parameter
sweep shifts a distribution; it does not bound a tail. `retrieval/agentic.py` has no
deadline, timeout, or time budget of any kind — nothing in the code can enforce a ceiling,
so the best any config can do is make violations rarer. A guaranteed max needs a wall-clock
check in the hop / fan-out loops that stops and answers with what has been retrieved so far.
Until that exists, treat `max_s` as a measurement of how far over the ceiling a config runs,
not as something a grid cell can fix.

**Where that latency actually goes (2026-08-26).** Profiled after a smoke run came in at
~84s/case under the current `candidates=30/top_k=10`. The reranker, not the LLM, is the
dominant cost: during a run the app container sat at ~1390% CPU while Ollama sat at 0.1%
(it runs on the host, on the GPU) and Qdrant at 0.18%. Per-case time tracks the number of
generated queries almost linearly, because the agentic path pays one **full** rerank per
query:

| queries in a case | observed case time |
|---|---|
| 1 | 15-24s |
| 4 | ~103s |
| 7 | 116-198s |

Measured cost of a single 30-candidate rerank, by where it runs (M5 Pro):

| Location | Per rerank | Notes |
|---|---|---|
| Container CPU (torch) | 10.34s | the shipped path until today |
| Container CPU (ONNX) | 8.57s | 1.21x; ONNX Runtime can't identify the CPU under Docker's VM |
| Host CPU (torch) | 6.58s | the container alone costs ~1.6x |
| **Host GPU (Metal)** | **1.48s** | **7x** — Docker on macOS cannot reach Metal at all |

`reranker.py`'s own comment claimed "1-3s for 25 candidates" — 3-5x optimistic, now
corrected. Verified end-to-end through `BGEReranker` container-to-host: identical `top_k`
ordering, max score delta 4.17e-07, 5.95x. See decisions-log.md for the two hypotheses
that were measured and discarded first (thread oversubscription, ONNX-as-the-fix), both
killed by controls rather than by theory.

**Consequences for every row in this file.** Latency numbers recorded before 2026-08-26
were measured on the in-container CPU reranker and are not comparable to anything measured
with the host service running — note which applies when adding a row. Accuracy numbers are
unaffected: the parity check confirms scores are identical to seven decimal places, so
nothing about what gets retrieved or refused changes. And `multi_query_count` is now known
to be a direct multiplier on the single most expensive stage, which sharpens Phase 2's
question considerably.

**What the move actually bought, measured end-to-end (2026-08-26, later).** The 7x on the
reranker is **~1.5x on wall-clock**, because reranks inside a case's multi-query fan-out run
concurrently — a per-component ratio does not multiply through. The arithmetic proves it:
the pre-move 125-case run issued 503 reranks, and 503 x 10.34s = 86.7min, more than that
run's entire 54.9min wall-clock.

| Run | Reranker | candidates / top_k | queries/case | **s/case** |
|---|---|---|---|---|
| `20260825-165051`, 125 cases | container CPU | 25 / 5 | 4.02 | 26.4 |
| `20260826-110031`, 20 cases | **host Metal** | 30 / 10 | 4.05 | **18.0** |

Near-identical fan-out per case makes these comparable on cost, and the post-move run is
doing *more* work per query (30 candidates to rerank, twice the chunks into the prompt).
No accuracy figure from `20260826-110031` appears here or anywhere in this file — it ran on
`golden_subset20.yaml`, which the 2026-08-25 subset false-signal entry rules out as a tuning
signal. Latency is recorded because cost per case is a property of the configuration, not of
which questions were asked; the caveat is that this set is 25% out-of-corpus and those cases
sit at both extremes (3.1s and 84.2s). On the production target: the pre-move 125-case p90
was 58.65s and the post-move 20-case p90 was 34.24s. Against the original ~15-20s both
missed; against the revised ~35s (owner, 2026-08-26, see above) the post-move figure
clears and the pre-move one does not.

**Deictic phrasing predicts difficulty better than the system's own confidence signal** (same
report): questions using "this X" / "the X that..." phrasing score 33% (15/45) vs 64% (29/45) for
plain phrasing — a bigger gap than fast-path-vs-slow-path (47% vs 62%). Detectable for free from the
question text, before retrieval. 17 of 25 reading-failure cases are deictic. Basis for the targeted
retry design in the main plan (cheap, one extra LLM call over existing chunks, vs. a full multi-hop
re-run for every deictic question, which would blow the latency budget above).

**Aggregation guard (`TABLE_MAJORITY`, `_has_aggregation_intent`):** measured against
[golden_aggregation.yaml](golden_aggregation.yaml) (8 real aggregations, 8 same-vocabulary lookups).
Vocabulary narrowed from "any aggregation word" (16/16 false-positive rate) to collective terms +
partitives (8/8 caught, 0/8 false positives). The guard cannot fire at all on the `hybridqa`
collection — every table there is exactly one chunk, so a 3-of-5 table majority is structurally
unreachable; it does fire on the real `documents` collection (11/14 table docs have 3+ chunks).
`golden_aggregation.yaml`'s header carries the full derivation.

**Semantic chunking — MEASURED, closed (2026-08-24).** First pass wrongly concluded "not worth
testing" from checking presence/absence in the existing structural chunks only, without ever running
semantic chunking — caught as an unsupported claim, corrected, and actually run.

```
# ingest: eval/ingest_hybridqa.py --collection hybridqa_semantic --semantic
# compare: retrieval-only (embed + rerank, no generation) over both collections,
#          89 in-corpus cases, top_k=5/candidates=25 (config.yaml defaults)
```

| | structural (`hybridqa`) | semantic (`hybridqa_semantic`) |
|---|---|---|
| answer recall | 67/89 (0.75) | 68/89 (0.76) |
| source recall | 78/89 (0.88) | 77/89 (0.87) |
| retrieval refused | 0/89 | 0/89 |

**Finding — a 1-point wash, not a win.** Semantic gained 3 cases and lost 2 relative to structural —
different individual questions, not a net improvement large enough to matter on n=89. Re-ingesting a
whole corpus into topic-boundary chunks to move one point (inside noise) is not worth the switch.
Confirms, with a real measurement this time, what the first (invalid) pass guessed: chunking strategy
is not where this benchmark's points are. **Branch G closed, no change shipped.**

Found and fixed along the way: `split_prose_semantic`'s `_split_sentences(text) or [text]` fallback
turned an empty passage into a single empty-string "sentence", which Ollama's embed endpoint 400s on
— fixed by dropping the fallback (an empty passage now yields zero chunks, correctly). Separately,
`OllamaEmbedder.embed` also hit a transient `400` from Ollama's own embed runner ("...tokenize: EOF")
that succeeded on an unmodified retry — added a 3-attempt retry in
[retrieval/embedder.py](retrieval/embedder.py), which benefits every caller, not just this script.

**Retrieval floors do not separate answerable from unanswerable — on HybridQA specifically.**
(uncensored scores, floors at 0.0, `eval/reports/20260824-132940.json`): in-corpus best-rerank-score
spans 0.512-0.731, out-of-corpus 0.501-0.731 — full overlap, `vector_floor` is actually
anti-correlated with answerability on this corpus (probe median cosine 0.618 vs in-corpus 0.558).
No floor setting beats `is_refusal` at this job — **on HybridQA.**

**CORRECTED (2026-08-24): this does not license deleting the floors.** `config.yaml`'s
`score_floor`/`vector_floor` comments say they were calibrated against the real `documents`
production collection, described there as having a *clean* separation (out-of-corpus 0.50-0.503 vs
in-corpus 0.578+; table rows at 0.4494-0.4718 cosine). HybridQA has no golden set covering
`documents`, so the finding above was never actually tested on the corpus the floors exist for — see
Branch E in the main plan file, reconsidered rather than executed. The floors stay.

---

## Hyperparameter tuning — floors, retrieval volume, agentic params

**Status: IN PROGRESS (2026-08-25).** Systematic sweeps replacing the stopgap defaults that
`config.yaml` itself flags as "NOT VALIDLY CALIBRATED". Execution instructions and runtime budget
live in [`eval/docs/tuning-runbook.md`](tuning-runbook.md); this section records the results. Ranking is
by `answer_coverage` subject to `refusal_accuracy >= 0.85` — the codebase's documented primary
metric and constraint, never a composite score.

Phases run in dependency order — **retrieval → agentic → floors → chunking**
— not the numeric order the phases were first drafted in, because floors
filter the score population that retrieval changes (see tuning-runbook.md).
Numbering below follows execution order.

Each phase's finalist (not every grid row) also gets a RAGAS judged pass —
`faithfulness` / `answer_correctness` / `context_precision` — since
`answer_coverage` alone cannot see a hallucinated answer that happens to
contain the right words (see tuning-runbook.md "RAGAS finalist check").

### Phase 1: Retrieval volume (`candidates` × `top_k`)

| candidates | top_k | answer_coverage | citation_precision | latency_max | latency_p90 | ragas_faithfulness | ragas_answer_correctness | ragas_context_precision | note |
|---|---|---|---|---|---|---|---|---|
| 30 | 10 | TBD | TBD | TBD | — | — | — | current default (Branch D, pending full confirmation) |
| ... | ... | ... | ... | ... | | | | |

`top_k=12` dropped from the swept range — see Finding below. `ragas_*`
columns filled only for the finalist row, per the RAGAS finalist check.

**Finding —** `top_k=12` excluded from this sweep rather than measured: it
adds context on a generation call already over the latency budget the
agentic phase measures (see Phase 2 below), and Branch D's own recall numbers
already show the gain flattening past `top_k=10`. Coordinate descent used
instead of the full `candidates × top_k` grid — see tuning-runbook.md Phase 1.
[remainder pending measurement]

### Phase 2: Agentic parameters (`max_hops` × `multi_query_count`)

**Finding — 12 of the originally-planned 16 combos are disqualified on
latency before any new run, and the phase's real question is inverted.** Not
a new measurement — a conclusion that follows from two results already in
this file and in git history, but never stated together until now.

1. The current default `max_hops=3, multi_query_count=3` measures p90
   **53.5s** against the agreed target of p90 15-20s (see "Latency reality"
   below, `eval/reports/20260824-132940.json`). The baseline is already ~3x
   over budget before any tuning change.
2. `multi_query_count=5` was separately measured at **+47s/case** and
   reverted (commit `256c051`) against a "<25s per question" requirement.

Together those close most of the grid:

| | mq=2 | mq=3 | mq=5 | mq=7 |
|---|---|---|---|---|
| **max_hops=1** | open | open | dead (+47s) | dead |
| **max_hops=2** | open | open | dead (+47s) | dead |
| **max_hops=3** | open | baseline | dead (+47s) | dead |
| **max_hops=4** | dead | dead | dead | dead |

`max_hops=4` is dead because `max_hops=3` already misses the p90 target 3x;
`multi_query_count` 5 and 7 are dead on the +47s measurement. Six combos
remain, one of which (`3, 3`) is the already-measured baseline — **5 new
runs, ~4.5h**, not 16 runs and ~20h.

**The pruning stands; one of its two arguments died and was replaced.**
Read the table above with this attached:

- The **+47s/case** that killed `mq` 5 and 7 is **stale** — measured with the
  reranker on container CPU, and `multi_query_count` multiplies rerank count,
  which went 10.34s → 1.48s. The figure would be a fraction of +47s today.
  But `mq` 5 and 7 stay dead on a different argument: more generated queries
  lengthen the tail, and every case over 35s in both measured runs had 7-13
  queries. The conclusion outlived its original reason.
- `max_hops=4` was dead for missing a p90 target since replaced by a stricter
  one — max ≤35s, which the baseline violates at 84.2s (see "Latency reality"
  above). Still dead, now on the tail rather than the p90.

Owner's call 2026-08-26, asked and answered explicitly: **keep the six-combo
grid**, keep `top_k=12` dropped. Reopening `mq=5` across `max_hops` 1-3 (9
combos, ~6h) is the cheapest way back in if a later result makes the question
live again; the full 16-combo grid is ~10h for this phase alone.

**The question stays inverted, and the max target keeps it that way.** This
phase is "how far *down* can these parameters go before coverage breaks" —
`max_hops=2` or `multi_query_count=2` is a latency win against a ceiling
being missed, not a tradeoff against a coverage gain. Rank rows on
`latency_max`. (An earlier revision of this section un-inverted the framing
on a misreading of the target as p90 ~35s; see decisions-log.md 2026-08-26.)

**This phase cannot be run in parallel.** Sharding combos across one Ollama
makes per-request latency a function of contention, and contention scales
with how many LLM calls a combo makes — so it would distort the *ranking*,
not just rescale every number by a constant. Moot at 5 runs; recorded so
parallelism isn't re-proposed here later.

| max_hops | multi_query_count | answer_coverage | multi_hop_citation_accuracy | latency_mean | ragas_faithfulness | ragas_answer_correctness | ragas_context_precision | note |
|---|---|---|---|---|---|---|---|---|
| 3 | 3 | TBD | TBD | 23.2s (p90 53.5s) | — | — | — | current default; pre-move figures, over the old 15-20s target |
| 3 | 2 | TBD | TBD | TBD | | | | cheaper fan-out |
| 2 | 3 | TBD | TBD | TBD | | | | cheaper hops |
| 2 | 2 | TBD | TBD | TBD | | | | both cheaper |
| 1 | 3 | TBD | TBD | TBD | | | | single hop |
| 1 | 2 | TBD | TBD | TBD | | | | floor of the shippable region |

### Phase 3: Floor calibration (`score_floor` × `vector_floor`)

**Read before trusting this table.** `replay_floor.py`'s grid re-derives
`refused` purely from whether a chunk clears the candidate floor, discarding
whatever the model itself decided, and never re-runs generation. So
`refusal_accuracy` here is a lower bound (a correctly-refused probe can be
re-scored as answered), `answer_coverage` is an upper bound (a case can stay
credited even after the chunk carrying the answer is filtered out), and
`citation_accuracy`/`citation_precision` are not meaningful — citations are
never re-derived. Both biases push toward floors that look better than they
are. **The winning combo must be confirmed with one real
`run_eval.py --agentic` run before `config.yaml` changes** — see
tuning-runbook.md Phase 3.

**Replayed against three existing uncensored reports** (`score_floor=0.0`,
`vector_floor=0.0`, same golden hash `228db7ac68da`, 125 cases: 90 in-corpus /
35 probes): `20260824-132940`, `20260825-152139`, `20260825-165051`. No new
pipeline run was needed for this table — that is the entire point of
`replay_floor.py`.

**The shipped `0.55/0.42` grid is flat: 0 of 125 cases lose all their chunks
at any of the original grid's cells.** The floors do no refusal work at that
setting — refusal is carried entirely by the model's own `NO_ANSWER` (30 of
35 probes refused in `20260825-152139`, vs. `replay_floor.py`'s
floor-only-derived 0.72 refusal_accuracy on the same report).

Best-chunk score distributions do not separate answerable from unanswerable
questions — if anything probes score *higher* on cosine:

| | min | p25 | p50 | p75 | max |
|---|---|---|---|---|---|
| in-corpus rerank | 0.549 | 0.699 | 0.726 | 0.730 | 0.731 |
| probe rerank | 0.504 | 0.597 | 0.687 | 0.716 | 0.731 |
| in-corpus vector | 0.397 | 0.522 | 0.558 | 0.616 | 0.714 |
| probe vector | 0.458 | 0.546 | **0.642** | 0.660 | 0.726 |

This is structural, not a calibration failure: a HybridQA probe like "What is
the average annual rainfall in Multan?" retrieves the correct, genuinely
relevant Multan table — the corpus is *about* the entity, it simply lacks the
fact asked for. No retrieval-score threshold can see that distinction; only
the model reading the content can.

**The floor's one measurable effect is context pruning**, holding
`score_floor=0.55` (identical across all three reports):

| vector_floor | in-corpus wiped /90 | probes wiped /35 | chunks kept |
|---|---|---|---|
| 0.42 (shipped) | 0 | 0 | 74% |
| 0.46 | 0 | 2 | 53% |
| 0.48 | 0 | 3 | 47% |
| **0.50** | **0** | **4** | **42%** |
| 0.52 | 0 | 4 | 38% |
| 0.55 | 1 | 4 | 36% |
| 0.60 | 1 | 4 | 34% |

`score_floor` cross-check at `vector_floor=0.50`: 0.50 keeps 100% of chunks
(all rerank scores measured are ≥0.504 — anything at or below ~0.50 is
inert); 0.55 wipes 0 in-corpus / 4 probes; 0.60 is identical on probes; 0.65
costs 3 in-corpus cases for +2 probes. `0.55` sits at the knee.

These three reports predate Branch D (`candidates=25/top_k=5`, not the
current `30/10`). Under wider retrieval the retrieved set is a strict
superset, so per-case max score can only rise — the floors become *strictly*
less likely to wipe a case. The conclusion below strengthens under `30/10`,
it does not weaken.

| score_floor | vector_floor | answer_coverage | refusal_accuracy | citation_precision | ragas_faithfulness | ragas_answer_correctness | ragas_context_precision | note |
|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.42 | 0.53 (replay) | 0.72 (replay, lower bound) | 0.29 (not meaningful, see above) | — | — | — | current default; 0/125 cases wiped |
| 0.55 | 0.50 | 0.53 (replay) | 0.75 (replay, lower bound) | 0.29 (not meaningful) | TBD | TBD | TBD | **recommended** — 0 in-corpus / +4 probes wiped, 58% less context |

**Finding — the floors are not the refusal mechanism on this corpus; they are
a context-pruning knob, and the shipped `vector_floor` leaves it almost
entirely open.** Move `vector_floor: 0.42 → 0.50`, keep `score_floor: 0.55`
(already at the knee). Zero in-corpus cases lost, +4 correctly-refused
probes, and generation sees 42% of retrieved context instead of 74% — worth
watching `citation_precision` (currently 0.287) and `latency_max` for the
knock-on effect. **Not yet shipped to `config.yaml`** — the replay cannot see
whether cutting context this much breaks the *answers* it was generated
from, so one confirm run (`run_eval.py --agentic --score-floor 0.55
--vector-floor 0.50`, on top of whatever Phase 1 settles on) gates the
`config.yaml` change. Phase 3 therefore needs 1 pipeline run, not 2 —
the uncensored run the runbook calls for already exists three times over.
`replay_floor.py`'s `VECTOR_GRID` was widened to `[0.42, 0.46, 0.48, 0.50,
0.52, 0.55]` (was topping out at 0.45) since 0.50 is the useful point and the
old range only bracketed it.

### Phase 4: Chunking token budget (structural only)

Chunking *strategy* is settled — semantic vs structural was measured and closed as a 1-point wash
(see "Semantic chunking" above, Branch G). Only the token budget within structural chunking is
untested.

Unlike Phases 1-3, this phase re-ingests and so **mutates the `hybridqa`
collection** — don't run it concurrently with anything else querying that
collection.

| config | answer_coverage | citation_precision | multi_hop_citation_accuracy | ragas_faithfulness | ragas_answer_correctness | ragas_context_precision | note |
|---|---|---|---|---|---|---|---|
| 350/50/10 | TBD | TBD | TBD | — | — | — | current default |
| 250/75/5 | TBD | TBD | TBD | | | | tight |
| 500/30/20 | TBD | TBD | TBD | | | | loose |

**Finding —** [pending]
