"""Print the benchmark ledger: one row per saved report.

Generated, never hand-written. The audit that prompted this found
eval/README.md claiming the golden set held 5 cases when it held 208 — a
results table maintained by hand rots exactly the same way, and a rotted
results table is worse than none, because it still looks authoritative.

Reports written before provenance existed (see run_eval.provenance) carry
none, so their configuration is reconstructed from the project record and
marked with "~". Never read a "~" row as measured.

Usage:
    python eval/report_table.py
    python eval/report_table.py --full    # every provenance field per run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (answer_accuracy, answer_coverage, refusal_accuracy,
                          citation_accuracy, citation_precision,
                          multi_hop_citation_accuracy)

ROOT = Path(__file__).parent

# label, and the function that recomputes it from the saved cases. Older
# reports predate some of these metrics, so their summaries have holes; the
# cases are always there, so a hole is filled rather than printed as "-".
METRICS = [
    ("answer_accuracy", "answer", answer_accuracy),
    ("answer_coverage", "coverage", answer_coverage),
    ("refusal_accuracy", "refusal", refusal_accuracy),
    ("citation_accuracy", "citation", citation_accuracy),
    ("citation_precision", "precision", citation_precision),
    ("multi_hop_citation_accuracy", "multihop", multi_hop_citation_accuracy),
]

# What the pre-provenance reports were run with, recovered from config.yaml's
# comments and the project log. Reconstructed, not recorded — hence the "~".
RECONSTRUCTED = {
    "model": "qwen2.5:3b",
    "score_floor": 0.55,
    "vector_floor": 0.42,
    "golden": "golden_hybridqa_draft.yaml",
    "collection": "hybridqa",
    "agentic": True,
}


def load(path: Path) -> dict:
    report = json.loads(path.read_text())
    summary = report.get("summary", {})
    cases = report.get("cases", [])
    prov = summary.get("provenance")

    # Prefer what the run recorded; recompute only what it never had. A
    # metric added after a run still applies to it — the cases are the
    # measurement, the summary is just a cache of it.
    metrics, recomputed = {}, False
    for key, _label, fn in METRICS:
        if summary.get(key) is not None:
            metrics[key] = summary[key]
        elif cases:
            try:
                metrics[key] = fn(cases)
                recomputed = True
            except Exception:
                metrics[key] = None
        else:
            metrics[key] = None

    return {
        "name": path.stem,
        "measured": prov is not None,
        "recomputed": recomputed,
        "prov": prov or dict(RECONSTRUCTED),
        "metrics": metrics,
        "n_cases": summary.get("n_cases") or len(cases) or None,
    }


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="print every provenance field per run")
    args = parser.parse_args()

    paths = sorted((ROOT / "reports").glob("2*.json"))
    # ragas-*.json are judged runs with a different metric set; this ledger
    # covers the deterministic benchmark only.
    paths = [p for p in paths if not p.stem.startswith("ragas")]
    if not paths:
        sys.exit("no reports in eval/reports/")

    runs = [load(p) for p in paths]

    head = ["run", "n", "model", "floors", "golden"] + [l for _, l, _ in METRICS]
    rows = []
    for run in runs:
        prov = run["prov"]
        floors = f"{fmt(prov.get('score_floor'))}/{fmt(prov.get('vector_floor'))}"
        golden = prov.get("golden_sha256") or prov.get("golden", "?")
        mark = "" if run["measured"] else " ~"
        rows.append(
            [run["name"] + mark, fmt(run["n_cases"]), str(prov.get("model")),
             floors, str(golden)]
            + [fmt(run["metrics"].get(key)) for key, _, _ in METRICS]
        )

    widths = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    if any(not r["measured"] for r in runs):
        print("\n~ configuration reconstructed, not recorded — "
              "predates run_eval.provenance. Do not read as measured.")
    if any(r["recomputed"] for r in runs):
        print("  Some metrics were recomputed from the saved cases because "
              "the run predates them. Those are measured, not guessed.")

    if args.full:
        for run in runs:
            print(f"\n{run['name']}{'' if run['measured'] else '  (~ reconstructed)'}")
            for key, value in sorted(run["prov"].items()):
                print(f"  {key:18} {value}")


if __name__ == "__main__":
    main()
