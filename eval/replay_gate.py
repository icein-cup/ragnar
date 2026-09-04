"""Score a candidate answer-gate against a saved eval report.

A gate that refuses hallucinated answers has two numbers, and only the pair
means anything: how many fabrications it catches, and how many good answers it
destroys. A gate that always says "unsupported" scores a perfect catch rate —
that is exactly what ``qwen2.5:3b`` did (10/10 caught, 23/25 good answers lost).

Re-running the benchmark to learn that costs ~40 minutes. But a report already
records, per case, the question, the answer, the retrieved contexts and the
citations — everything the gate needs. Replaying only the gate over those saved
answers costs one LLM call per case, so a candidate model is scored in ~2
minutes with no retrieval and no re-answering.

Usage:
    python eval/replay_gate.py --model qwen2.5:7b
    python eval/replay_gate.py --model qwen2.5:7b --contexts used
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from eval.metrics import _words
from generation.llm import OllamaLLM
from generation.prompts import build_grounding_prompt

ROOT = Path(__file__).parent


def latest_report() -> Path:
    reports = sorted((ROOT / "reports").glob("2*.json"))
    if not reports:
        sys.exit("no eval report found — run eval/run_eval.py first")
    return reports[-1]


def is_correct(case: dict) -> bool:
    """Did the answer actually state the expected answer?

    This must be correctness, not "cited a plausible source". Scoring the 27B
    against a cited-source population made it look far worse than it was: 10 of
    25 "good answers" in that sample were wrong answers that cited the right
    document, so the gate got penalised for catching them.
    """
    return _words(case["expected_answer"]) in _words(case["answer"])


def populations(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """(should be refused, must survive) — the two groups a gate is judged on.

    Should be refused: the corpus cannot answer it, yet the system answered —
    plus in-corpus answers that are simply wrong. Both are answers a user
    should never see, and a gate catching either is doing its job.
    Must survive: answered, in-corpus, and factually right.
    """
    answered = [c for c in cases if not c["refused"]]
    bad = [c for c in answered
           if c["out_of_corpus"] or not is_correct(c)]
    good = [c for c in answered
            if not c["out_of_corpus"] and is_correct(c)]
    return bad, good


def used_contexts(case: dict) -> list[str]:
    """The excerpts the answer was actually built from, best-effort.

    The report stores contexts and citations separately, so the two can only be
    lined up by position — good enough to test whether a smaller context helps
    the judge, which is all this flag is for.
    """
    return case["contexts"][:max(len(case["citations"]), 1)]


# Experimental second gate, kept here rather than in generation/prompts.py
# until it beats the shipped one: extraction is an easier task for a small
# model than judgement, and the quote it returns can be checked deterministically
# instead of trusted.
EVIDENCE_PROMPT = """\
Find the evidence for an answer in the provided document excerpts.

Reply in exactly two lines:
QUOTE: the sentence from the excerpts that supports the answer, copied \
word for word, or NONE if no sentence supports it
VERDICT: SUPPORTED or UNSUPPORTED

Copy the quote exactly. Do not paraphrase it, and do not write a sentence \
that is not in the excerpts.
"""


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def evidence_gate(llm: OllamaLLM, case: dict, model: str, contexts: str) -> bool:
    """Ask for the supporting sentence, then verify the quote really exists.

    A hallucinated answer has no sentence to point at, so the model either says
    NONE or invents one — and an invented quote fails the string check against
    the excerpts, which needs no trust in the judge at all.
    """
    texts = used_contexts(case) if contexts == "used" else case["contexts"]
    body = (f"Excerpts:\n\n" + "\n\n".join(texts)
            + f"\n\nAnswer:\n{case['answer']}")
    try:
        reply = llm.generate(EVIDENCE_PROMPT, body, model=model)
    except Exception as exc:
        print(f"  judge error ({exc}) — counting as SUPPORTED", file=sys.stderr)
        return True

    quote, verdict = "", ""
    for line in reply.splitlines():
        low = line.strip().lower()
        if low.startswith("quote:"):
            quote = line.split(":", 1)[1].strip()
        elif low.startswith("verdict:"):
            verdict = line.split(":", 1)[1].strip().upper()
    if verdict.startswith("UNSUPPORTED") or quote.upper().startswith("NONE"):
        return False
    # The quote must actually appear in the excerpts. Compare a prefix: models
    # trail off or fix punctuation, but the opening words are copied verbatim
    # when the sentence is really there.
    hay = _squash(" ".join(texts))
    needle = _squash(quote)[:60]
    return bool(needle) and needle in hay


def grounded(llm: OllamaLLM, case: dict, model: str, contexts: str) -> bool:
    texts = used_contexts(case) if contexts == "used" else case["contexts"]
    system, user = build_grounding_prompt(
        case["answer"], [(f"source {i + 1}", t) for i, t in enumerate(texts)]
    )
    try:
        verdict = llm.generate(system, user, model=model).strip().upper()
    except Exception as exc:
        print(f"  judge error ({exc}) — counting as SUPPORTED", file=sys.stderr)
        return True
    return not verdict.startswith("UNSUPPORTED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=None,
                        help="eval report to replay (default: newest)")
    parser.add_argument("--model", default=None,
                        help="Ollama model to judge with (default: config.yaml)")
    parser.add_argument("--gate", choices=("grounding", "evidence"),
                        default="grounding",
                        help="grounding: one-word supported/unsupported "
                             "verdict. evidence: quote the supporting sentence, "
                             "then verify the quote is really in the excerpts")
    parser.add_argument("--contexts", choices=("all", "used"), default="all",
                        help="feed the judge every retrieved chunk, or only "
                             "the ones the answer cited")
    parser.add_argument("--sample", type=int, default=25,
                        help="how many good answers to score (all "
                             "fabrications are always scored)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = args.report or latest_report()
    cfg = Config()
    model = args.model or cfg.llm_model
    cases = json.loads(report.read_text())["cases"]
    fabrications, good = populations(cases)
    random.Random(args.seed).shuffle(good)
    sample = good[:args.sample]

    print(f"report:   {report.name}")
    print(f"model:    {model}   gate: {args.gate}   contexts: {args.contexts}")
    print(f"scoring:  {len(fabrications)} answers that should be refused, "
          f"{len(sample)} of {len(good)} correct answers that must survive\n")

    gate = evidence_gate if args.gate == "evidence" else grounded
    llm = OllamaLLM(cfg.ollama_url, model, think=False)
    try:
        started = time.monotonic()

        caught = []
        for case in fabrications:
            if not gate(llm, case, model, args.contexts):
                caught.append(case)
            print(f"bad answer   {'CAUGHT ' if case in caught else 'missed '} "
                  f"{case['question'][:58]}", flush=True)

        lost = []
        for case in sample:
            if not gate(llm, case, model, args.contexts):
                lost.append(case)
            print(f"correct      {'LOST   ' if case in lost else 'kept   '} "
                  f"{case['question'][:58]}", flush=True)

        print(f"\n{time.monotonic() - started:.0f}s")
        print(f"bad answers caught:     {len(caught)}/{len(fabrications)}")
        print(f"correct answers lost:   {len(lost)}/{len(sample)}")
        if len(lost) == len(sample):
            print("\nThis gate refuses everything — catching every fabrication "
                  "here means nothing.")
        for case in lost:
            print(f"  lost: {case['question'][:60]}\n"
                  f"        {case['answer'][:70]!r}")

    finally:
        # A finished replay has no reason to hold the weights.
        # keep_alive would keep them for ten more minutes, and
        # Ollama only evicts an idle model under memory pressure —
        # which a 31 GB model on a 48 GB host reaches too late.
        llm.unload()


if __name__ == "__main__":
    main()
