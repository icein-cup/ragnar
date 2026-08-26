"""Ragas judged-metric harness.

Eval-only. By default reuses `run_eval.run_cases()` so the exact same
retrieval and answer pipeline is exercised, then scores the in-corpus cases
with Ragas metrics using an external LLM judge (OpenAI-compatible, e.g.
Ollama Cloud) and the same local embedding model via Ollama. Pass --report
to score a saved eval/run_eval.py report instead — a post-process of a run
already paid for, with no retrieval/generation re-run.

Usage:
    python eval/run_ragas.py --agentic --collection hybridqa \\
        --golden eval/golden_hybridqa_draft.yaml
    python eval/run_ragas.py --report eval/reports/<stamp>.json
    python eval/run_ragas.py --report   # scores the newest report
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Allow `python eval/run_ragas.py` to find the app packages (same shim as
# run_eval.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Legacy `ragas.metrics` names are deprecated in favour of
# `ragas.metrics.collections`, but only the legacy classes implement the
# `SingleTurnMetric` interface that `evaluate()` drives. Suppress the noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics._aspect_critic import AspectCritic

from core.config import Config
from eval.run_eval import run_cases

ROOT = Path(__file__).parent


def build_judge() -> ChatOpenAI:
    """OpenAI-compatible judge from the env, eval-only.

    The key lives only in this module's env; the app never reads it.
    """
    base_url = os.environ.get("RAGAS_JUDGE_BASE_URL", "").strip()
    api_key = os.environ.get("RAGAS_JUDGE_API_KEY", "").strip()
    if not base_url or not api_key:
        raise SystemExit(
            "RAGAS_JUDGE_BASE_URL and RAGAS_JUDGE_API_KEY must be set in the "
            "environment (see .env.example)."
        )
    model = os.environ.get("RAGAS_JUDGE_MODEL", "glm-5.2:cloud").strip()
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0,
    )


def latest_report() -> Path:
    """Newest eval/run_eval.py report, for a bare --report with no path.

    Same glob as eval/replay_floor.py:latest_report() — kept in sync rather
    than imported, since replay_floor.py deliberately never imports the app
    stack and this module already does.
    """
    reports = sorted((ROOT / "reports").glob("2*.json"))
    if not reports:
        sys.exit("no eval report found — run eval/run_eval.py first")
    return reports[-1]


def build_samples(agentic: bool = False,
                  collection: str | None = None,
                  golden: Path | None = None,
                  report: Path | None = None) -> list[SingleTurnSample]:
    """In-corpus, answered cases mapped to Ragas field names.

    Mapping to the golden set:
        question        -> user_input
        answer          -> response
        expected_answer -> reference
        contexts        -> retrieved_contexts

    Out-of-corpus cases are refused by design and have no meaningful answer
    or contexts to score, so they are skipped here. refusal_accuracy in
    run_eval.py already covers those.

    report, when given, reads a saved eval/run_eval.py report's "cases"
    instead of calling run_cases() again. A report already carries every
    field this needs (question, answer, expected_answer, contexts) plus the
    two filter fields below (out_of_corpus, refused) — see the per-case dict
    built in run_eval.py's run_cases(). Scoring a saved report is a
    post-process of a run already paid for, not a second ~37-minute pipeline
    run; agentic/collection/golden are ignored in this mode since the report
    already encodes what produced it.
    """
    if report:
        cases = json.loads(report.read_text())["cases"]
    else:
        cases = run_cases(agentic=agentic, collection=collection,
                          golden=golden)

    samples = []
    for case in cases:
        if case["out_of_corpus"] or case["refused"]:
            continue
        if not case["contexts"] or not case["answer"]:
            continue
        samples.append(SingleTurnSample(
            user_input=case["question"],
            response=case["answer"],
            reference=case["expected_answer"],
            retrieved_contexts=case["contexts"],
        ))
    return samples


def build_metrics() -> list:
    """The RAG triad plus RAGnar-specific critique.

    The three headline Ragas metrics — faithfulness, answer_relevancy,
    context_precision/recall — cover answer hallucination, on-topic
    relevance, and retrieval quality. answer_correctness / similarity and
    context_entity_recall add golden-answer and entity coverage.

    `no_invented_numbers` is a RAGnar-specific guard against hallucinated
    figures, which the house deterministic metrics cannot catch once an
    answer is actually produced.
    """
    # answer_relevancy defaults to strictness=3 — three generations per case,
    # averaged to cancel judge variance. Pointless here: the judge runs at
    # temperature=0, so all three would be identical, and Ollama Cloud's
    # OpenAI-compatible endpoint ignores `n` anyway and returns one. Pinning
    # it to 1 states what already happens and drops the "LLM returned 1
    # generations instead of requested 3" warning on every case.
    answer_relevancy.strictness = 1

    no_invented_numbers = AspectCritic(
        name="no_invented_numbers",
        definition=(
            "Does the answer state any number, amount, date, or value that "
            "cannot be found in the retrieved contexts? Answer No if a "
            "figure appears that is absent from the contexts, Yes otherwise."
        ),
    )
    return [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        answer_similarity,
        context_precision,
        context_recall,
        context_entity_recall,
        no_invented_numbers,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None,
                        help="write the report JSON to this path")
    parser.add_argument("--agentic", action="store_true",
                        help="route through AgenticSearch, as the UI does")
    parser.add_argument("--collection", default=None,
                        help="override the Qdrant collection (e.g. hybridqa)")
    parser.add_argument("--golden", type=Path, default=None,
                        help="override the golden set YAML")
    parser.add_argument("--report", nargs="?", const="latest", default=None,
                        help="score a saved eval/run_eval.py report instead "
                             "of re-running the pipeline (--agentic/"
                             "--collection/--golden are ignored in this "
                             "mode). Pass a path, or bare --report for the "
                             "newest eval/reports/*.json")
    args = parser.parse_args()

    report_path = None
    if args.report is not None:
        report_path = latest_report() if args.report == "latest" \
            else Path(args.report)

    cfg = Config()
    samples = build_samples(agentic=args.agentic, collection=args.collection,
                            golden=args.golden, report=report_path)
    if not samples:
        raise SystemExit(
            "No in-corpus answered cases to score. Ingest the corpus and "
            "expand eval/golden_set.yaml first."
        )

    judge = build_judge()
    embeddings = OllamaEmbeddings(
        model=cfg.embedding_model, base_url=cfg.ollama_url,
    )

    result = evaluate(
        EvaluationDataset(samples),
        metrics=build_metrics(),
        llm=judge,
        embeddings=embeddings,
        # ragas defaults to max_workers=16 — that many concurrent requests
        # against a single external judge risks rate-limit floods/errors.
        # Capped rather than tuned; raise only if the judge's own limits
        # are known to clear this.
        run_config=RunConfig(max_workers=4),
    )
    df = result.to_pandas()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": int(len(samples)),
        # Numeric columns only — to_pandas() also carries user_input,
        # response, reference and retrieved_contexts as object columns, and
        # .mean() on those raises TypeError under pandas 2, after the judged
        # run has already been paid for.
        "means": {
            str(col): float(value)
            for col, value in df.select_dtypes("number").mean().items()
        },
        "cases": df.to_dict(orient="records"),
    }
    print(json.dumps(report["means"], indent=2, ensure_ascii=False))

    out = args.output or (ROOT / "reports" / f"ragas-{datetime.now():%Y%m%d-%H%M%S}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()