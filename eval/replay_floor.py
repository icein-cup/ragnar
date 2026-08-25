"""Replay retrieval-floor combinations over a saved eval report.

A floor has two numbers and only the pair means anything: how many
answerable questions it still answers, and how many unanswerable ones it
refuses. Re-running the pipeline once per candidate floor costs ~55 minutes
each; a report already records, per case, the rerank and vector score of
every chunk that survived, so a candidate floor can be re-applied
arithmetically in seconds.

The catch: the saved ``scores`` are the chunks that SURVIVED the floors the
run was made with. To replay a floor you need the UNCENSORED population — the
full reranked top_k — which only exists when the source report was produced
with both floors at 0.0. This script refuses to run against a report that was
not, because the arithmetic would silently under-count refusals.

Usage:
    python eval/replay_floor.py --report eval/reports/20260825-000000.json
    python eval/replay_floor.py --report <path> --score-floor 0.55 --vector-floor 0.42
    python eval/replay_floor.py --report <path> --grid   # sweep both axes

The --grid sweep prints a table of answer_coverage vs refusal_accuracy for
every (score_floor, vector_floor) combination, which is the heatmap the
config.yaml comments say was never produced.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (answer_coverage, refusal_accuracy,
                          citation_accuracy, citation_precision)

ROOT = Path(__file__).parent

SCORE_GRID = [0.45, 0.50, 0.55, 0.60, 0.65]
VECTOR_GRID = [0.35, 0.40, 0.45, 0.50]


def latest_report() -> Path:
    reports = sorted((ROOT / "reports").glob("2*.json"))
    if not reports:
        sys.exit("no eval report found — run eval/run_eval.py first")
    return reports[-1]


def _clears(rerank: float, vector: float | None, floor: float,
            vfloor: float) -> bool:
    """Mirror of retrieval.search.clears_floor, kept local so this script
    never imports the app's retrieval stack (it only needs the arithmetic)."""
    return rerank >= floor or (vector is not None and vector >= vfloor)


def _was_uncensored(report: dict) -> bool:
    """True only when the report was produced with both floors at 0.0."""
    prov = report.get("summary", {}).get("provenance", {})
    return prov.get("score_floor") == 0.0 and prov.get("vector_floor") == 0.0


def apply_floor(case: dict, floor: float, vfloor: float) -> dict:
    """Return a copy of ``case`` with the floors re-applied to its scores.

    A case is refused when NO surviving chunk clears either floor. The
    ``scores`` list is filtered to the survivors, and ``refused`` is set
    accordingly. ``answer``/``citations`` are left untouched — they describe
    what the model said given the ORIGINAL contexts, and re-scoring them
    would require re-running the LLM, which is exactly what this script
    exists to avoid. The refusal flag is the only thing a floor changes
    without a model call.
    """
    scores = case.get("scores", [])
    kept = [s for s in scores
            if _clears(s["rerank"], s.get("vector"), floor, vfloor)]
    out = dict(case)
    out["scores"] = kept
    out["refused"] = not kept
    return out


def replay(report: dict, floor: float, vfloor: float) -> list[dict]:
    return [apply_floor(c, floor, vfloor) for c in report["cases"]]


def _row(floor: float, vfloor: float, cases: list[dict]) -> str:
    return (f"{floor:>7.2f} {vfloor:>7.2f} "
            f"{answer_coverage(cases):>12.2f} "
            f"{refusal_accuracy(cases):>12.2f} "
            f"{citation_accuracy(cases):>13.2f} "
            f"{citation_precision(cases):>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=None,
                        help="eval report to replay (default: newest)")
    parser.add_argument("--score-floor", type=float, default=None,
                        help="single score_floor to evaluate")
    parser.add_argument("--vector-floor", type=float, default=None,
                        help="single vector_floor to evaluate")
    parser.add_argument("--grid", action="store_true",
                        help="sweep both axes and print the heatmap")
    args = parser.parse_args()

    report_path = args.report or latest_report()
    report = json.loads(report_path.read_text())

    if not _was_uncensored(report):
        prov = report.get("summary", {}).get("provenance", {})
        sys.exit(
            f"report {report_path.name} was produced with floors "
            f"score={prov.get('score_floor')} vector={prov.get('vector_floor')}, "
            f"not 0.0/0.0 — its saved scores are already censored, so a floor "
            f"replay would under-count refusals. Re-run eval/run_eval.py with "
            f"--score-floor 0 --vector-floor 0 first."
        )

    print(f"report: {report_path.name}  "
          f"({len(report['cases'])} cases, uncensored)")

    if args.grid:
        print(f"{'score':>7} {'vector':>7} {'coverage':>12} "
              f"{'refusal':>12} {'cit_acc':>13} {'cit_prec':>13}")
        for floor in SCORE_GRID:
            for vfloor in VECTOR_GRID:
                cases = replay(report, floor, vfloor)
                print(_row(floor, vfloor, cases))
        return

    if args.score_floor is None or args.vector_floor is None:
        sys.exit("pass both --score-floor and --vector-floor, or --grid")

    cases = replay(report, args.score_floor, args.vector_floor)
    print(_row(args.score_floor, args.vector_floor, cases))


if __name__ == "__main__":
    main()
