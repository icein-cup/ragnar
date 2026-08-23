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
import json
import sys
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
from generation.answerer import Answerer, AnswerMode, classify
from eval.metrics import refusal_accuracy, citation_accuracy, multi_hop_citation_accuracy

ROOT = Path(__file__).parent


def run_cases(score_floor: float | None = None,
              vector_floor: float | None = None,
              agentic: bool = False,
              collection: str | None = None,
              golden: Path | None = None) -> list[dict]:
    """Run the golden set through retrieval + answering.

    Both floors must be passed: retrieval.search.clears_floor keeps a result
    when EITHER clears, so leaving vector_floor at its 0.0 default made every
    reranked result pass (a cosine vector_score is >= 0 in practice) and the
    floor sweep measured nothing at all.

    agentic=True routes through AgenticSearch, which is what the UI actually
    runs — the bare Search path below measures a pipeline no user hits.
    """
    cfg = Config()
    floor = cfg.score_floor if score_floor is None else score_floor
    vfloor = cfg.vector_floor if vector_floor is None else vector_floor
    collection = collection or cfg.collection
    golden = golden or (ROOT / "golden_set.yaml")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, collection, cfg.embedding_dim)
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)
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

    for entry in golden_entries:
        outcome = search.find(entry["question"])
        mode = classify(entry["question"], outcome.refused, outcome.results)

        if mode is not AnswerMode.ANSWER:
            answer_text, citations, refused = "", [], True
        else:
            answer = answerer.answer(entry["question"], outcome.results)
            answer_text = answer.text
            citations = answer.citations
            refused = answer.refused

        cases.append({
            **entry,
            "answer": answer_text,
            "citations": citations,
            "refused": refused,
            "contexts": [r.chunk.text for r in outcome.results],
        })

    return cases


SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def calibrate_floor(vector_floor: float | None = None,
                    agentic: bool = False) -> None:
    """Sweep score_floor and report which value separates the two groups best.

    The floor cannot be chosen in advance - it depends on the corpus. This
    is what the out-of-corpus golden entries exist for.

    vector_floor is PINNED for the sweep, not swept, and printed in the
    header. It has to be: the two floors are OR'd, so a vector_floor left at
    0.0 rescues every result and flattens this table into a constant. Sweep
    the other axis by re-running with a different --vector-floor.
    """
    cfg = Config()
    vfloor = cfg.vector_floor if vector_floor is None else vector_floor
    print(f"vector_floor pinned at {vfloor:.2f} "
          f"(agentic={'on' if agentic else 'off'})")
    print(f"{'floor':>7} {'refusal_acc':>12} {'citation_acc':>13}")
    for floor in SWEEP:
        cases = run_cases(score_floor=floor, vector_floor=vfloor,
                          agentic=agentic)
        print(f"{floor:>7.2f} {refusal_accuracy(cases):>12.2f} "
              f"{citation_accuracy(cases):>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--agentic", action="store_true",
                        help="route through AgenticSearch, as the UI does")
    parser.add_argument("--vector-floor", type=float, default=None,
                        help="override config.yaml's retrieval.vector_floor")
    parser.add_argument("--collection", default=None,
                        help="override the Qdrant collection (e.g. hybridqa)")
    parser.add_argument("--golden", type=Path, default=None,
                        help="override the golden set YAML "
                             "(default: eval/golden_set.yaml)")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_floor(vector_floor=args.vector_floor, agentic=args.agentic)
        return

    cases = run_cases(vector_floor=args.vector_floor, agentic=args.agentic,
                      collection=args.collection, golden=args.golden)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "agentic": args.agentic,
        "collection": args.collection or Config().collection,
        "refusal_accuracy": refusal_accuracy(cases),
        "citation_accuracy": citation_accuracy(cases),
        "multi_hop_citation_accuracy": multi_hop_citation_accuracy(cases),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (reports / f"{stamp}.json").write_text(
        json.dumps({"summary": report, "cases": cases},
                   indent=2, ensure_ascii=False)
    )
    print(f"\nwrote eval/reports/{stamp}.json")


if __name__ == "__main__":
    main()
