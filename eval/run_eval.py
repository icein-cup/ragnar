"""Evaluation harness.

Imports app modules; the app never imports this. The external judge's API
key lives only here, and the harness runs against a hand-written test set —
production documents have no code path to an external service.

Usage:
    python eval/run_eval.py                 # deterministic metrics only
    python eval/run_eval.py --calibrate      # sweep the similarity floor
    python eval/run_eval.py --agentic        # measure the pipeline the UI runs
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow `python eval/run_eval.py` to find the app packages even though
# running a script (rather than `-m`) puts this file's directory, not the
# repo root, at the front of sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from core.config import Config
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.reranker import BGEReranker
from retrieval.search import Search
from retrieval.agentic import AgenticSearch
from generation.llm import OllamaLLM
from generation.answerer import (Answerer, AnswerMode, citation_labels,
                                 classify)
from generation.guards import is_refusal, refusal_text, strip_no_answer
from eval.metrics import (answer_accuracy, answer_coverage, refusal_accuracy,
                          citation_accuracy, citation_precision,
                          multi_hop_citation_accuracy)

ROOT = Path(__file__).parent


def resolve_answer(question: str, outcome, mode: AnswerMode,
                   answerer: Answerer) -> tuple[str, list[str], bool, bool]:
    """One case's (answer text, citations, refused, draft_reused).

    Extracted from the run loop so the draft-reuse branch below is testable
    without a live Qdrant and Ollama behind it.

    Self-correction already generates a full answer from these same excerpts,
    and the UI displays it directly (ui/app.py). Calling answerer.answer()
    when that draft exists would generate the whole thing a second time — the
    most expensive call in the pipeline — and would measure a path no user
    takes, which is the opposite of what --agentic is for.
    """
    if mode is not AnswerMode.ANSWER:
        return "", [], True, False

    draft = getattr(outcome, "draft_answer", None)
    if draft:
        # Same order as the UI: read the sentinel before stripping it, since
        # stripping is exactly what hides it from is_refusal.
        refused = is_refusal(draft)
        if refused:
            return refusal_text(draft), [], True, True
        stripped = strip_no_answer(draft)
        return stripped, citation_labels(outcome.results, stripped), False, True

    answer = answerer.answer(question, outcome.results)
    return answer.text, answer.citations, answer.refused, False


def run_cases(score_floor: float | None = None,
              vector_floor: float | None = None,
              agentic: bool = False,
              collection: str | None = None,
              golden: Path | None = None,
              sink: Path | None = None) -> list[dict]:
    """Run the golden set through retrieval + answering.

    Both floors must be passed: retrieval.search.clears_floor keeps a result
    when EITHER clears, so leaving vector_floor at its 0.0 default made every
    reranked result pass (a cosine vector_score is >= 0 in practice) and the
    floor sweep measured nothing at all.

    agentic=True routes through AgenticSearch, which is what the UI actually
    runs — the bare Search path below measures a pipeline no user hits.

    sink, when given, receives one JSON line per case as it completes. An
    agentic run over the HybridQA set is ~55 minutes, and without this a
    crash or a Ctrl-C at case 120 of 125 threw away the whole hour. The
    lines carry the same per-case shape the replay harnesses read, so a
    partial run is still scoreable.
    """
    cfg = Config()
    floor = cfg.score_floor if score_floor is None else score_floor
    vfloor = cfg.vector_floor if vector_floor is None else vector_floor
    collection = collection or cfg.collection
    golden = golden or (ROOT / "golden_set.yaml")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, collection, cfg.embedding_dim)
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model,
                    think=cfg.llm_think, seed=cfg.llm_seed)
    search = Search(embedder, store, reranker=BGEReranker(cfg.reranker_model),
                    candidates=cfg.candidates, top_k=cfg.top_k,
                    score_floor=floor, vector_floor=vfloor)
    if agentic:
        # Same wiring as ui/services.build_services, so the numbers describe
        # the pipeline the UI takes.
        search = AgenticSearch(search, llm, **cfg.agentic)
    answerer = Answerer(llm)

    golden_entries = yaml.safe_load(golden.read_text())
    cases = []
    sink_file = sink.open("w") if sink else None

    # A --agentic run is ~10s per case and prints nothing until the end, which
    # reads as a hang on a 100+ entry golden set. One line per case on stderr,
    # so the report on stdout stays pipeable.
    started = time.monotonic()
    try:
        for n, entry in enumerate(golden_entries, 1):
            case_started = time.monotonic()
            outcome = search.find(entry["question"])
            mode = classify(entry["question"], outcome.refused, outcome.results)

            answer_text, citations, refused, draft_reused = resolve_answer(
                entry["question"], outcome, mode, answerer)

            case = {
                **entry,
                "answer": answer_text,
                "citations": citations,
                "refused": refused,
                "contexts": [r.chunk.text for r in outcome.results],
                # Scores are what floor calibration needs. Without them, tuning
                # score_floor/vector_floor means re-running the whole pipeline once
                # per candidate value; with them it is arithmetic over this file.
                #
                # These are the chunks that SURVIVED the floors. Run with both
                # floors at 0.0 and they are the full reranked top_k instead —
                # the uncensored population a floor would be chosen from. The
                # summary records the floors so a reader can tell which they are
                # looking at.
                "scores": [{"label": r.chunk.citation_label(),
                            "rerank": r.score, "vector": r.vector_score}
                           for r in outcome.results],
                # Where the run spent its LLM calls. All of these are already on
                # AgenticSearchOutcome and were simply never written down, so a
                # slow benchmark gave no clue which stage to attack: a fast-path
                # case costs about two calls, a self-corrected one about six.
                # With these saved, every run profiles itself.
                "stages": {
                    "seconds": round(time.monotonic() - case_started, 2),
                    "fast_path": getattr(outcome, "fast_path", False),
                    "hops": getattr(outcome, "hops_performed", 0),
                    "self_corrected": getattr(outcome, "self_corrected", False),
                    "draft_reused": draft_reused,
                    "queries": len(getattr(outcome, "queries_executed", [])),
                },
            }
            cases.append(case)
            if sink_file:
                sink_file.write(json.dumps(case, ensure_ascii=False) + "\n")
                sink_file.flush()

            elapsed = time.monotonic() - started
            eta = elapsed / n * (len(golden_entries) - n)
            want = "refuse" if entry["out_of_corpus"] else "answer"
            got = "refuse" if refused else "answer"
            print(f"[{n}/{len(golden_entries)}] {'ok ' if want == got else 'MISS'} "
                  f"want={want} got={got} eta={eta / 60:.1f}m "
                  f"| {entry['question'][:60]}", file=sys.stderr, flush=True)

    finally:
        if sink_file:
            sink_file.close()
        # Release the weights however this ended — normal exit, exception, or
        # Ctrl-C. keep_alive would otherwise hold them for ten more minutes,
        # and Ollama does not evict an idle model until memory runs out.
        llm.unload()
    return cases


def _git_revision() -> str:
    """Short SHA, suffixed "-dirty" when the tree has uncommitted changes.

    A report that cannot be traced to code is a number without a cause.
    """
    def _git(*args: str) -> str:
        return subprocess.run(("git",) + args, cwd=ROOT.parent,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        sha = _git("rev-parse", "--short", "HEAD")
        if not sha:
            return "unknown"
        return f"{sha}-dirty" if _git("status", "--porcelain") else sha
    except Exception:
        return "unknown"


def provenance(cfg: Config, golden: Path, *, agentic: bool,
               collection: str, score_floor: float,
               vector_floor: float) -> dict:
    """Everything needed to say what produced a set of numbers.

    Without this a report is five metrics and a timestamp, and two runs that
    disagree give no way to tell whether the model changed, the floors moved,
    or the golden set itself was edited underneath. The golden hash is the
    load-bearing part: this benchmark went 100 -> 84 -> 112 -> 125 cases
    during development, and scores from different sets are not comparable at
    all.
    """
    body = golden.read_bytes()
    return {
        "git": _git_revision(),
        "model": cfg.llm_model,
        "think": cfg.llm_think,
        "seed": cfg.llm_seed,
        "embedding_model": cfg.embedding_model,
        "reranker_model": cfg.reranker_model,
        "collection": collection,
        "agentic": agentic,
        "agentic_config": cfg.agentic if agentic else None,
        "score_floor": score_floor,
        "vector_floor": vector_floor,
        "candidates": cfg.candidates,
        "top_k": cfg.top_k,
        "golden": golden.name,
        "golden_sha256": hashlib.sha256(body).hexdigest()[:12],
        "golden_cases": len(yaml.safe_load(body)),
    }


SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def calibrate_floor(score_floor: float | None = None,
                    vector_floor: float | None = None,
                    agentic: bool = False) -> None:
    """Sweep score_floor and report which value separates the two groups best.

    The floor cannot be chosen in advance - it depends on the corpus. This
    is what the out-of-corpus golden entries exist for.

    vector_floor is PINNED for the sweep, not swept, and printed in the
    header. It has to be: the two floors are OR'd, so a vector_floor left at
    0.0 rescues every result and flattens this table into a constant. Sweep
    the other axis by re-running with a different --vector-floor.

    score_floor is ignored here — this function's whole job is to sweep it —
    but it is accepted so main() can pass its arguments through uniformly.

    Prefer the offline route: one run with both floors at 0.0 saves every
    score, and any candidate floor can then be evaluated arithmetically over
    that report. This sweep re-runs the entire pipeline once per value, so at
    ~55 minutes a run it costs the better part of a day.
    """
    cfg = Config()
    vfloor = cfg.vector_floor if vector_floor is None else vector_floor
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"vector_floor pinned at {vfloor:.2f} "
          f"(agentic={'on' if agentic else 'off'})")
    print(f"{'floor':>7} {'refusal_acc':>12} {'citation_acc':>13}")
    for floor in SWEEP:
        # Each sweep point gets its own sink: nine sequential runs with
        # nothing on disk until the last one finishes is a whole day at risk.
        cases = run_cases(score_floor=floor, vector_floor=vfloor,
                          agentic=agentic,
                          sink=reports / f"{stamp}-floor{floor:.2f}.jsonl")
        print(f"{floor:>7.2f} {refusal_accuracy(cases):>12.2f} "
              f"{citation_accuracy(cases):>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--agentic", action="store_true",
                        help="route through AgenticSearch, as the UI does")
    parser.add_argument("--score-floor", type=float, default=None,
                        help="override config.yaml's retrieval.score_floor. "
                             "Pass 0 together with --vector-floor 0 to run "
                             "with the floors open, which is what makes the "
                             "saved scores usable for calibration")
    parser.add_argument("--vector-floor", type=float, default=None,
                        help="override config.yaml's retrieval.vector_floor")
    parser.add_argument("--collection", default=None,
                        help="override the Qdrant collection (e.g. hybridqa)")
    parser.add_argument("--golden", type=Path, default=None,
                        help="override the golden set YAML "
                             "(default: eval/golden_set.yaml)")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_floor(score_floor=args.score_floor,
                        vector_floor=args.vector_floor, agentic=args.agentic)
        return

    cfg = Config()
    collection = args.collection or cfg.collection
    golden = args.golden or (ROOT / "golden_set.yaml")
    floor = cfg.score_floor if args.score_floor is None else args.score_floor
    vfloor = cfg.vector_floor if args.vector_floor is None else args.vector_floor

    # The stamp is chosen BEFORE the run so the streamed .jsonl and the final
    # .json share it — a killed run leaves a file you can name.
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    cases = run_cases(score_floor=args.score_floor,
                      vector_floor=args.vector_floor, agentic=args.agentic,
                      collection=collection, golden=golden,
                      sink=reports / f"{stamp}.jsonl")
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "answer_accuracy": answer_accuracy(cases),
        "answer_coverage": answer_coverage(cases),
        "refusal_accuracy": refusal_accuracy(cases),
        "citation_accuracy": citation_accuracy(cases),
        "citation_precision": citation_precision(cases),
        "multi_hop_citation_accuracy": multi_hop_citation_accuracy(cases),
        "latency": {
            "mean_s": round(sum(c["stages"]["seconds"] for c in cases) / len(cases), 2),
            "median_s": round(sorted(c["stages"]["seconds"] for c in cases)[len(cases) // 2], 2),
            "p90_s": round(sorted(c["stages"]["seconds"] for c in cases)[int(len(cases) * 0.9)], 2),
            "max_s": round(max(c["stages"]["seconds"] for c in cases), 2),
        },
        "provenance": provenance(cfg, golden, agentic=args.agentic,
                                 collection=collection, score_floor=floor,
                                 vector_floor=vfloor),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    (reports / f"{stamp}.json").write_text(
        json.dumps({"summary": report, "cases": cases},
                   indent=2, ensure_ascii=False)
    )
    print(f"\nwrote eval/reports/{stamp}.json")


if __name__ == "__main__":
    main()
