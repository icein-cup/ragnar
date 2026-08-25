"""Score a candidate answering prompt against a saved eval report.

Same trick as replay_gate.py, one step earlier in the pipeline. A report holds
the chunks retrieval actually returned, so the answering step can be re-run
over them with a different prompt — one LLM call per case instead of the five
the agentic path costs, and no retrieval at all. A prompt variant is scored in
minutes rather than the ~55 a full benchmark takes.

What it measures, on the same case:
  answer_accuracy  over in-corpus cases  — did the answer state the right fact
  refusal_rate     over out-of-corpus    — did it still decline what it should

Both matter: a prompt that lifts accuracy by answering everything more boldly
has moved the failure, not fixed it.

Usage:
    python eval/replay_answer.py --prompt baseline
    python eval/replay_answer.py --prompt hops --sample 40
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.models import Chunk, SearchResult
from eval.metrics import _words
from generation.guards import NO_ANSWER, is_refusal
from generation.llm import OllamaLLM
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

# Merge candidate: the "hops" bridge-resolution rule folded into the shipped
# sentinel-carrying prompt, rather than replacing it outright. "hops" alone
# measured a real accuracy win (COMPARISONS.md) but drops the NO_ANSWER
# contract, falling back to a regex refusal check that only works by luck of
# phrasing. This restores the sentinel and adds the bridge-resolution step
# plus two entity-precision rules "hops" also had. Test before shipping.
HOPS_SENTINEL = f"""\
You answer questions strictly from the provided document excerpts.

Before answering, work through these steps internally (do not show them in \
your output):
1. Identify exactly what the question asks — the entity, the property, the \
time frame, and any implicit sub-questions. Many questions describe the \
subject indirectly instead of naming it ("the city where X happened", "the \
institute that Y founded") — resolve that description to the concrete \
entity first. The description is how you find the subject; it is not the \
answer.
2. Check EVERY excerpt one by one. Look for the answer even when the wording \
differs from the question, when the information is indirect, or when it is \
split across multiple excerpts.
3. If the answer requires combining facts from two or more excerpts, piece \
them together: one excerpt may name the entity, another may give the value, \
and a third may provide the date. Synthesize across all excerpts that are \
relevant.
4. Only after you have checked every excerpt, decide: can the question be \
answered from the excerpts alone?

Output ONLY the final answer — never the reasoning steps, never excerpt \
numbers, never "based on the excerpts". If yes, give a concise, factual \
answer. Do not speculate or embellish.
If no — you have genuinely checked every excerpt and none contains the \
answer, even indirectly — start your reply with {NO_ANSWER} and briefly \
state what is missing. Do not guess.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Default to answering. Refusing is the last resort, not the first. Most \
questions that seem unanswered at first glance CAN be answered by combining \
or carefully reading the excerpts.
- Never answer with a value the question already gave you.
- Check that the fact you found belongs to the subject the question \
describes, not a neighbouring row or a similar entry.
- When the answer requires synthesizing across excerpts, combine the facts \
explicitly. Do not give up because no single excerpt contains the full answer.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual.
- Citations are added separately after your answer — do not include your \
own citations or source references in the response text.
"""

ROOT = Path(__file__).parent

# Candidate answering prompts live here, not in generation/prompts.py, until
# one beats the shipped SYSTEM_PROMPT on the numbers below.
#
# Failure modes this is aimed at, counted over the 3b baseline report: of 39
# wrong in-corpus answers, 24 had the right answer sitting in the retrieved
# chunks. They fail three ways — answering the bridge entity instead of the
# property asked for ("the owner of the location named after the Queen
# consort" -> "the City of Charlottesville"), echoing a number the question
# itself supplied as a filter ("the airport with 234,306 movements" ->
# "234,306"), and lifting an adjacent row ("current minister" -> the previous
# one).
PROMPTS = {
    "baseline": SYSTEM_PROMPT,
    "hops_sentinel": HOPS_SENTINEL,

    # Isolates one variable from hops_sentinel above: same short shape as
    # "hops" below, only the refusal line swapped to carry the token. No
    # 4-step CoT block, no "default to answering" framing. Tests whether the
    # merge's regression came from the sentinel or from the verbosity it was
    # merged alongside.
    "hops_sentinel_minimal": f"""\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- If the excerpts do not contain the answer, start your reply with \
{NO_ANSWER} and briefly state what is missing. Do not guess.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
""",

    # hops_sentinel_minimal stated the sentinel as a flat, prominent rule
    # ("If the excerpts do not contain the answer, start with NO_ANSWER...")
    # and lost 18 points to hops. This variant borrows two things already
    # measured to work in the ORIGINAL sentinel v1-vs-v2 test
    # (generation/prompts.py's SYSTEM_PROMPT comment): the explicit
    # "default to answering" framing, and stating the refusal instruction
    # LAST, gated on "only if you have genuinely checked" rather than as a
    # co-equal option alongside the others.
    "hops_sentinel_v2": f"""\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Default to answering. Refusing is the last resort, not the first — most \
questions that seem unanswered at first glance CAN be answered by reading \
carefully or combining excerpts.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
- Only if you have genuinely checked every excerpt and none contains the \
answer, even indirectly: start your reply with {NO_ANSWER} and briefly state \
what is missing. Do not guess.
""",

    # v2 recovered 9 of hops's 13-point lead by moving the refusal
    # instruction last and adding "default to answering" framing. These two
    # variants isolate the next lever: how the refusal instruction ITSELF is
    # worded, holding position and framing constant.
    #
    # v3: drop "briefly state what is missing" -- v1 (in the ORIGINAL
    # sentinel test) already showed that asking the model to justify a
    # refusal makes it reach for one more often. v2 still carries that ask.
    "hops_sentinel_v3": f"""\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Default to answering. Refusing is the last resort, not the first — most \
questions that seem unanswered at first glance CAN be answered by reading \
carefully or combining excerpts.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
- Only if you have genuinely checked every excerpt and none contains the \
answer, even indirectly: reply with exactly {NO_ANSWER}. Do not guess.
""",

    # v4: reframe NO_ANSWER as a parsing/formatting requirement the system
    # depends on, not a content option the model is choosing between --
    # testing whether that framing reduces how readily it's reached for.
    "hops_sentinel_v4": f"""\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- The excerpts almost always contain the answer if you read them carefully — \
most questions that seem unanswered at first CAN be answered by combining or \
re-reading them.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
- Formatting requirement for automated processing: in the rare case where you \
have genuinely checked every excerpt and none contains the answer, even \
indirectly, your reply must begin with the exact token {NO_ANSWER}. This is a \
parsing signal, not a suggestion to stop looking early.
""",

    # v5: v2 plus one LIGHT thoroughness line ("read every excerpt
    # carefully...") borrowed from baseline's step 2, without the full
    # 4-step CoT scaffold that hops_sentinel (+CoT) showed actively hurts.
    # Tests whether a minimal version of that instruction helps.
    "hops_sentinel_v5": f"""\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Rules:
- Default to answering. Refusing is the last resort, not the first — most \
questions that seem unanswered at first glance CAN be answered by reading \
carefully or combining excerpts.
- Read every excerpt carefully before deciding — the answer is often \
indirect, uses different wording than the question, or is split across more \
than one excerpt.
- Use ONLY information in the excerpts. Never use outside knowledge.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
- Only if you have genuinely checked every excerpt and none contains the \
answer, even indirectly: start your reply with {NO_ANSWER} and briefly state \
what is missing. Do not guess.
""",

    # Amends the GENERAL (bridge-resolution) instruction, not the sentinel --
    # a separate axis. No sentinel here; compares directly against "hops" to
    # see if a worked example raises the ceiling independent of refusal
    # wording. If it helps, the best general content + best sentinel wording
    # get combined next.
    "hops_example": """\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Example: if asked "What is the population of the city where the 1996 \
Olympics were held?", first resolve "the city where the 1996 Olympics were \
held" to Atlanta, then answer with Atlanta's population — not 1996, not \
"Olympics".

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
""",

    # Combines the two winning levers found so far on separate axes:
    # hops_example's worked example (general-prompt axis: 0.62/0.85, no
    # sentinel) and hops_sentinel_v2's refusal wording (sentinel axis:
    # 0.55/0.85). Testing whether they stack.
    "hops_sentinel_v6": f"""\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Example: if asked "What is the population of the city where the 1996 \
Olympics were held?", first resolve "the city where the 1996 Olympics were \
held" to Atlanta, then answer with Atlanta's population — not 1996, not \
"Olympics".

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Default to answering. Refusing is the last resort, not the first — most \
questions that seem unanswered at first glance CAN be answered by reading \
carefully or combining excerpts.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
- Only if you have genuinely checked every excerpt and none contains the \
answer, even indirectly: start your reply with {NO_ANSWER} and briefly state \
what is missing. Do not guess.
""",

    "hops": """\
You answer questions strictly from the provided document excerpts.

Many questions describe something indirectly before asking about it — "the \
city where X happened", "the airport that has N flights". Resolve that \
description first, then answer what the question actually asks about it. The \
description is how you find the subject; it is not the answer.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Never answer with a value the question already gave you.
- Give the specific fact asked for. Do not restate the question.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Check that the fact you found belongs to the subject the question describes, \
not to a neighbouring row or a similar entry.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are in a \
different language.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your own \
citations or source references in the response text.
""",
}


def latest_report() -> Path:
    reports = sorted((ROOT / "reports").glob("2*.json"))
    if not reports:
        sys.exit("no eval report found — run eval/run_eval.py first")
    return reports[-1]


def results_for(case: dict) -> list[SearchResult]:
    """Rebuild SearchResults from a saved case.

    Only the text and the label reach the prompt, and the report stores both
    (labels positionally, which is close enough — build_user_prompt uses them
    as excerpt headings, not as data).
    """
    labels = case["citations"] or ["source"]
    return [
        SearchResult(
            chunk=Chunk(doc_id="d", filename=labels[i % len(labels)],
                        text=text, chunk_index=i),
            score=0.0,
        )
        for i, text in enumerate(case["contexts"])
    ]


def answer_with(llm: OllamaLLM, system: str, case: dict, model: str) -> str:
    results = results_for(case)
    user = build_user_prompt(
        case["question"], [(r.chunk.filename, r.chunk.text) for r in results]
    )
    try:
        return llm.generate(system, user, model=model)
    except Exception as exc:
        print(f"  llm error: {exc}", file=sys.stderr)
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="baseline")
    parser.add_argument("--model", default=None,
                        help="default: config.yaml models.llm")
    parser.add_argument("--sample", type=int, default=40,
                        help="in-corpus cases to score")
    parser.add_argument("--probes", type=int, default=20,
                        help="out-of-corpus cases to score")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", choices=("off", "on", "default"),
                        default="off",
                        help="off (default) sends think:false; on sends "
                             "think:true; default omits the field. A thinking "
                             "model forced not to think may answer far worse "
                             "than it can — this is how that gets measured")
    args = parser.parse_args()

    report = args.report or latest_report()
    cfg = Config()
    model = args.model or cfg.llm_model
    cases = json.loads(report.read_text())["cases"]

    rng = random.Random(args.seed)
    in_corpus = [c for c in cases if not c["out_of_corpus"] and c["contexts"]]
    probes = [c for c in cases if c["out_of_corpus"] and c["contexts"]]
    rng.shuffle(in_corpus)
    rng.shuffle(probes)
    in_corpus, probes = in_corpus[:args.sample], probes[:args.probes]

    print(f"report: {report.name}   model: {model}   prompt: {args.prompt}"
          f"   think: {args.think}")
    print(f"scoring {len(in_corpus)} in-corpus, {len(probes)} probes\n")

    think = {"off": False, "on": True, "default": None}[args.think]
    llm = OllamaLLM(cfg.ollama_url, model, think=think)
    try:
        system = PROMPTS[args.prompt]
        started = time.monotonic()

        correct = 0
        for n, case in enumerate(in_corpus, 1):
            text = answer_with(llm, system, case, model)
            # A refusal is never correct here — these questions are answerable.
            # Without the is_refusal guard, a refusal whose explanation happens to
            # repeat a word from the expected answer scored as a hit.
            ok = (not is_refusal(text)
                  and _words(case["expected_answer"]) in _words(text))
            correct += ok
            print(f"[{n}/{len(in_corpus)}] {'ok  ' if ok else 'WRONG'} "
                  f"want={case['expected_answer'][:28]!r} got={text[:52]!r}", flush=True)

        refused = 0
        for n, case in enumerate(probes, 1):
            text = answer_with(llm, system, case, model)
            r = is_refusal(text)
            refused += r
            print(f"[probe {n}/{len(probes)}] {'refused' if r else 'ANSWERED'} "
                  f"{case['question'][:44]} -> {text[:46]!r}", flush=True)

        print(f"\n{time.monotonic() - started:.0f}s   prompt={args.prompt} model={model}")
        print(f"answer_accuracy : {correct}/{len(in_corpus)} = {correct / len(in_corpus):.2f}")
        print(f"refusal_rate    : {refused}/{len(probes)} = {refused / len(probes):.2f}")

    finally:
        # A finished replay has no reason to hold the weights.
        # keep_alive would keep them for ten more minutes, and
        # Ollama only evicts an idle model under memory pressure —
        # which a 31 GB model on a 48 GB host reaches too late.
        llm.unload()


if __name__ == "__main__":
    main()
