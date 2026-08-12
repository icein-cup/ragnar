from ingestion.chunkers.semantic import SemanticChunker
from ingestion.chunkers.structural import StructuralChunker

# The ONLY place a component is selected by name. Deliberately narrow:
# chunking is the one component with an experiment attached (Ragas eval,
# a much later task), which is what justifies swappability here and
# nowhere else in this codebase.
CHUNKERS = {
    "structural": StructuralChunker,
    "semantic": SemanticChunker,
}


def build_chunker(config: dict, embedder=None):
    name = config.get("strategy", "structural")
    if name not in CHUNKERS:
        raise ValueError(
            f"Unknown chunker '{name}'. Available: {sorted(CHUNKERS)}"
        )

    if name == "semantic":
        if embedder is None:
            raise ValueError(
                "The 'semantic' chunker embeds sentences to find topic "
                "boundaries and needs an embedder — none was passed to "
                "build_chunker()."
            )
        return SemanticChunker(
            embedder,
            target_tokens=config.get("target_tokens", 500),
            rows_per_group=config.get("table_rows_per_group", 20),
            breakpoint_percentile=config.get(
                "semantic_breakpoint_percentile", 95),
        )

    return StructuralChunker(
        target_tokens=config.get("target_tokens", 500),
        overlap_tokens=config.get("overlap_tokens", 50),
        rows_per_group=config.get("table_rows_per_group", 20),
    )
