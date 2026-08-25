"""Hyperparameter tuning harness for RAGnar.

Orchestrates the four tuning phases in cheapest-first order. Each phase
produces a CSV + JSON under eval/reports/ and prints a ranked summary.

Phases:
    floors     — offline. Replays score_floor × vector_floor over a saved
                 uncensored report (see eval/replay_floor.py). No LLM calls.
    retrieval  — candidates × top_k. Requires re-running the pipeline per
                 combo (these params change what is retrieved, so there is no
                 offline shortcut).
    agentic    — max_hops × multi_query_count. Same: re-runs the pipeline.
    chunking   — target_tokens × overlap_tokens × table_rows_per_group.
                 Requires re-ingestion per config, so it is NOT automated
                 here; see eval/README.md "Comparing configurations".

Ranking is by answer_coverage subject to a refusal_accuracy floor — the
codebase's documented primary metric and constraint (see eval/metrics.py).
citation_precision and latency are reported as secondary columns, never
folded into a composite score.

Usage:
    python eval/tune_params.py --phase floors
    python eval/tune_params.py --phase retrieval
    python eval/tune_params.py --phase agentic
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent

# The refusal floor a config must clear to be considered. Mirrors the
# ~0.85 target the codebase treats as the production bar (config.yaml's
# model comments cite 0.90 probe refusal as the shipped quality).
MIN_REFUSAL = 0.85

FLOOR_GRID = {
    "score_floor": [0.45, 0.50, 0.55, 0.60, 0.65],
    "vector_floor": [0.35, 0.40, 0.45, 0.50],
}

RETRIEVAL_GRID = {
    "candidates": [20, 25, 30, 40],
    "top_k": [5, 8, 10, 12],
}

AGENTIC_GRID = {
    "max_hops": [1, 2, 3, 4],
    "multi_query_count": [2, 3, 5, 7],
}

# Metrics a run reports, in display order.
METRICS = [
    "answer_coverage",
    "refusal_accuracy",
    "citation_accuracy",
    "citation_precision",
    "multi_hop_citation_accuracy",
]


def _run_eval(overrides: dict[str, str], collection: str | None) -> dict | None:
    """Run eval/run_eval.py --agentic with the given overrides, parse the
    summary. Returns None on failure so a partial sweep can still be saved."""
    cmd = [sys.executable, "eval/run_eval.py", "--agentic"]
    if collection:
        cmd += ["--collection", collection]
    for key, val in overrides.items():
        cmd += [f"--{key.replace('_', '-')}", str(val)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"eval failed: {result.stderr[-500:]}", file=sys.stderr)
        return None

    summary = json.loads(result.stdout)
    row = {k: summary.get(k) for k in METRICS}
    row["latency_mean_s"] = summary.get("latency", {}).get("mean_s")
    row["latency_p90_s"] = summary.get("latency", {}).get("p90_s")
    row.update(overrides)
    return row


def _run_floors(report: Path | None) -> list[dict]:
    """Offline floor sweep via replay_floor.py — no LLM calls."""
    cmd = [sys.executable, "eval/replay_floor.py", "--grid"]
    if report:
        cmd += ["--report", str(report)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return []

    rows = []
    lines = result.stdout.splitlines()
    # Skip the "report:" header line and the column header line.
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        rows.append({
            "score_floor": float(parts[0]),
            "vector_floor": float(parts[1]),
            "answer_coverage": float(parts[2]),
            "refusal_accuracy": float(parts[3]),
            "citation_accuracy": float(parts[4]),
            "citation_precision": float(parts[5]),
        })
    return rows


def _run_grid(phase: str, collection: str | None) -> list[dict]:
    grid = {"retrieval": RETRIEVAL_GRID, "agentic": AGENTIC_GRID}[phase]
    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    rows = []
    for i, combo in enumerate(combos, 1):
        overrides = dict(zip(keys, combo))
        print(f"[{i}/{len(combos)}] {phase}: {overrides}", file=sys.stderr)
        row = _run_eval(overrides, collection)
        if row:
            rows.append(row)
    return rows


def _save(rows: list[dict], phase: str) -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if rows:
        with open(reports / f"tuning-{phase}-{stamp}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(reports / f"tuning-{phase}-{stamp}.json", "w") as fh:
        json.dump({"phase": phase, "results": rows}, fh, indent=2)
    print(f"wrote eval/reports/tuning-{phase}-{stamp}.csv/.json")


def _summarize(rows: list[dict], phase: str) -> None:
    if not rows:
        print("no results to summarize")
        return
    # Primary metric: answer_coverage. Constraint: refusal_accuracy >= MIN_REFUSAL.
    viable = [r for r in rows if r.get("refusal_accuracy", 0) >= MIN_REFUSAL]
    pool = viable or rows  # if nothing clears the bar, show the best refusal
    ranked = sorted(pool, key=lambda r: r.get("answer_coverage", 0), reverse=True)

    print(f"\n=== {phase}: top 5 by answer_coverage "
          f"(refusal_accuracy >= {MIN_REFUSAL}) ===")
    for i, r in enumerate(ranked[:5], 1):
        params = ", ".join(f"{k}={v}" for k, v in r.items()
                           if k not in METRICS + ["latency_mean_s", "latency_p90_s"])
        print(f"{i}. {params}")
        print(f"   coverage={r.get('answer_coverage'):.2f} "
              f"refusal={r.get('refusal_accuracy'):.2f} "
              f"cit_prec={r.get('citation_precision'):.2f} "
              f"p90={r.get('latency_p90_s')}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["floors", "retrieval", "agentic"],
                        required=True)
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection override")
    parser.add_argument("--report", type=Path, default=None,
                        help="for --phase floors: uncensored report to replay")
    args = parser.parse_args()

    if args.phase == "floors":
        rows = _run_floors(args.report)
    else:
        rows = _run_grid(args.phase, args.collection)

    _save(rows, args.phase)
    _summarize(rows, args.phase)


if __name__ == "__main__":
    main()
