"""Export the reranker cross-encoder to ONNX, then prove it scores the same.

Why: the torch backend measured ~10s to rerank 30 candidates on CPU, and the
agentic path pays that once per generated query — the dominant cost of an eval
run (a 7-query case spends over a minute in the reranker alone). ONNX Runtime
executes the same weights through a graph it can optimize ahead of time,
because the model is frozen and cannot change the way a training-time graph can.

Why a script rather than exporting on the fly: BAAI/bge-reranker-v2-m3 ships no
prebuilt ONNX weights, so sentence-transformers would re-export on every load —
and the container's HuggingFace cache is ephemeral (docker-compose.yml mounts
only ./:/app), so that cost would repeat on every rebuild. Exporting once into
data/ makes it survive.

The parity check is not optional. These scores feed retrieval.score_floor, so a
shift in the last decimals can change which chunks survive and silently move
every metric in eval/. This refuses to leave an export in place that reorders
candidates against the torch baseline.

Usage:
    docker compose exec app python eval/export_reranker_onnx.py
    docker compose exec app python eval/export_reranker_onnx.py --verify-only
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.reranker import MODEL_NAME, ONNX_DIR

# Pairs the parity check scores. Deliberately spans the shapes this corpus
# actually retrieves — a markdown table row, ordinary prose, and text with no
# bearing on the query — since agreement on easy positives alone would say
# nothing about whether the ranking holds up near the floor.
PROBE_QUERY = "What river flows near the city with the $6.5 billion GDP in 2010?"
PROBE_TEXTS = [
    "| City | GDP (2010) | River |\n|---|---|---|\n| Multan | $6.5 billion | Chenab |",
    "The Chenab River flows through the Punjab plains past several major cities.",
    "Multan is a city in Punjab, Pakistan, known for its shrines and pottery.",
    "The 1988 Winter Olympics were held in Calgary, Alberta, Canada.",
    "Rainfall in the region averages 186 millimetres annually.",
    "| Team | Score |\n|---|---|\n| Red | 12 |\n| Blue | 9 |",
]

# Ranking must match exactly; raw scores may differ in the last decimals
# because ONNX fuses operations that torch runs separately. 1e-3 is far below
# anything score_floor (0.55) or vector_floor discriminates on, and well above
# float32 fusion noise.
SCORE_TOLERANCE = 1e-3


def _scores(model) -> list[float]:
    return [float(s) for s in model.predict([(PROBE_QUERY, t) for t in PROBE_TEXTS])]


def export() -> None:
    from sentence_transformers import CrossEncoder

    if ONNX_DIR.exists():
        print(f"removing existing export at {ONNX_DIR}")
        shutil.rmtree(ONNX_DIR)
    ONNX_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"exporting {MODEL_NAME} to ONNX (slow, one-off)...")
    # backend="onnx" with no prebuilt weights triggers the optimum export.
    model = CrossEncoder(MODEL_NAME, backend="onnx")
    model.save_pretrained(str(ONNX_DIR))
    print(f"wrote {ONNX_DIR}")


def verify() -> bool:
    from sentence_transformers import CrossEncoder

    if not ONNX_DIR.is_dir():
        sys.exit(f"no export at {ONNX_DIR} — run without --verify-only first")

    print("scoring probes on torch backend...")
    torch_scores = _scores(CrossEncoder(MODEL_NAME))
    print("scoring probes on onnx backend...")
    onnx_scores = _scores(CrossEncoder(str(ONNX_DIR), backend="onnx"))

    torch_rank = sorted(range(len(torch_scores)), key=lambda i: -torch_scores[i])
    onnx_rank = sorted(range(len(onnx_scores)), key=lambda i: -onnx_scores[i])
    worst = max(abs(t - o) for t, o in zip(torch_scores, onnx_scores))

    print(f"\n{'#':>3} {'torch':>12} {'onnx':>12} {'delta':>12}")
    for i, (t, o) in enumerate(zip(torch_scores, onnx_scores)):
        print(f"{i:>3} {t:>12.6f} {o:>12.6f} {abs(t - o):>12.2e}")
    print(f"\nmax score delta : {worst:.2e} (tolerance {SCORE_TOLERANCE:.0e})")
    print(f"torch ranking   : {torch_rank}")
    print(f"onnx ranking    : {onnx_rank}")

    if torch_rank != onnx_rank:
        print("\nFAIL: ONNX reorders candidates against torch. The export changes "
              "what survives score_floor, so every eval number would shift. "
              "Not safe to use.")
        return False
    if worst > SCORE_TOLERANCE:
        print(f"\nFAIL: max delta {worst:.2e} exceeds {SCORE_TOLERANCE:.0e}. Ranking "
              "held here, but scores this far apart can cross score_floor on "
              "other inputs.")
        return False
    print("\nPASS: identical ranking, scores within tolerance.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="skip the export, just re-check an existing one")
    args = parser.parse_args()

    if not args.verify_only:
        export()
    if not verify():
        sys.exit(1)


if __name__ == "__main__":
    main()
