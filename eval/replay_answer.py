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
