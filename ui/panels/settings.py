import streamlit as st

from ingestion.chunkers.registry import build_chunker
from ui.services import list_chat_models, list_loaded_models, warm_model

_STRATEGIES = {"Structural": "structural", "Semantic": "semantic"}


def _apply_chunker(svc, strategy: str, chunk_size: int, overlap_pct: int,
                   table_rows: int) -> None:
    """Point the ingestion pipeline at the current chunk settings.

    Only rebuilds the chunker when the settings actually changed — this runs
    on every rerun, and reconstructing a chunker each time would be needless
    churn.
    """
    cfg = (strategy, chunk_size, overlap_pct, table_rows)
    if st.session_state.get("_applied_chunk_cfg") == cfg:
        return
    svc["pipeline"].set_chunker(build_chunker(
        {
            "strategy": strategy,
            "target_tokens": chunk_size,
            "overlap_tokens": round(chunk_size * overlap_pct / 100),
            "table_rows_per_group": table_rows,
        },
        embedder=svc["embedder"],
    ))
    st.session_state["_applied_chunk_cfg"] = cfg


def render(svc) -> dict:
    """Render the Settings panel and return the per-request query settings.

    Returning them (rather than mutating shared service objects) is what
    keeps concurrent users from clobbering each other's model/temperature/
    floor choices — each request carries its own.
    """
    cfg = svc["cfg"]
    with st.expander("Settings", expanded=False):
        available_models = list_chat_models(
            cfg.ollama_url, cfg.embedding_model
        ) or [cfg.llm_model]
        default_model = (
            cfg.llm_model if cfg.llm_model in available_models
            else available_models[0]
        )
        model = st.selectbox(
            "Model", options=available_models,
            index=available_models.index(default_model),
        )
        if model == cfg.llm_model:
            st.caption("⭐ Default recommended model")
        loaded = list_loaded_models(cfg.ollama_url)
        if model in loaded:
            st.caption("🟢 Model loaded in memory")
        else:
            st.caption(
                "⚪ Not loaded — first response will include a loading delay"
            )
            if st.button("⚡ Pre-load model", key="preload_model_btn"):
                with st.spinner(f"Loading {model} into memory…"):
                    ok = warm_model(cfg.ollama_url, model)
                if ok:
                    st.toast(f"{model} is ready!", icon="✅")
                else:
                    st.warning(f"Could not pre-load {model}. Check Ollama logs.")
                st.rerun()
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
            help="0 = deterministic, always the most likely answer. Higher "
                 "values allow more varied, less predictable phrasing.",
        )

        st.markdown("**Agentic RAG**")
        enable_rewrite = st.checkbox("Query rewriting", value=True,
                                    help="Use LLM to rewrite queries for better retrieval")
        enable_multi_query = st.checkbox("Multi-query retrieval", value=True,
                                         help="Generate multiple query variants and fuse results")
        enable_multi_hop = st.checkbox("Multi-hop reasoning", value=True,
                                       help="Iteratively retrieve based on intermediate findings")
        enable_self_correction = st.checkbox("Self-correction", value=True,
                                            help="Evaluate answer completeness and re-retrieve if needed")

        st.markdown("**Retrieval**")
        use_reranker = st.checkbox(
            "Re-rank results for accuracy", value=True,
            help="A second, more accurate pass over retrieved chunks. On by "
                 "default. Turning it off makes answers faster but less "
                 "precise, and disables the similarity floor below (so the "
                 "app will answer whenever anything is retrieved).",
        )
        floor = st.slider(
            "Similarity floor", min_value=0.0, max_value=1.0,
            value=float(cfg.score_floor), step=0.05,
            disabled=not use_reranker,
            help="How relevant a document chunk must be to be used. Higher = "
                 "stricter (refuses more, safer against wrong answers); lower "
                 "= more lenient (answers more, riskier on off-topic "
                 "questions). A chunk is kept if EITHER this or the vector "
                 "floor below is cleared. Only applies when re-ranking is on.",
        )
        vector_floor = st.slider(
            "Vector floor", min_value=0.0, max_value=1.0,
            value=float(cfg.vector_floor), step=0.05,
            disabled=not use_reranker,
            help="A second, more lenient relevance check on the raw "
                 "embedding similarity, before re-ranking. Rescues content "
                 "(table rows especially) that the re-ranker scores as "
                 "neutral despite being genuinely relevant. Only applies "
                 "when re-ranking is on.",
        )

        st.markdown("**Chunking**")
        default_strategy = cfg.chunking.get("strategy", "structural")
        strategy_labels = list(_STRATEGIES)
        default_label = next(
            (label for label, value in _STRATEGIES.items()
             if value == default_strategy),
            "Structural",
        )
        strategy_label = st.selectbox(
            "Chunking strategy", options=strategy_labels,
            index=strategy_labels.index(default_label),
            help="Structural follows the document's own headings, pages, "
                 "and tables — the default, and the stronger choice when a "
                 "document is cleanly parsed. Semantic instead finds topic "
                 "shifts by meaning (embedding each sentence), which may "
                 "suit scanned or heading-poor documents better. This is "
                 "here to be compared on your corpus, not a permanent "
                 "either/or.",
        )
        strategy = _STRATEGIES[strategy_label]
        is_semantic = strategy == "semantic"

        default_tokens = cfg.chunking.get("target_tokens", 500)
        default_overlap_pct = round(
            100 * cfg.chunking.get("overlap_tokens", 50) / default_tokens
        )
        chunk_size = st.number_input(
            "Chunk size (tokens)", min_value=100, max_value=2000,
            value=default_tokens, step=50,
            help="Semantic chunking treats this as a hard ceiling, not a "
                 "target — cuts are placed at topic boundaries and only "
                 "forced by size if a topic runs long." if is_semantic else None,
        )
        overlap_pct = st.number_input(
            "Overlap (%)", min_value=0, max_value=50,
            value=default_overlap_pct, step=1,
            disabled=is_semantic,
            help="Not used by semantic chunking — its cuts are chosen to "
                 "land on topic boundaries, and overlapping across a "
                 "boundary picked that way would defeat the point."
                 if is_semantic else None,
        )
        st.caption(
            "Chunk size and overlap apply to text only. Tables (PDF tables "
            "and Excel sheets) follow the separate row-based rule below — "
            "token size doesn't apply to tabular data."
        )
        table_rows = st.number_input(
            "Table rows per chunk", min_value=5, max_value=200,
            value=cfg.chunking.get("table_rows_per_group", 20), step=5,
            help="How many table/spreadsheet rows go into one chunk, "
                 "independent of the chunk size setting above. A wide table "
                 "with long cells may read better with fewer rows per chunk; "
                 "a narrow table can fit more.",
        )
        st.caption(
            "Chunking applies to documents uploaded from now on. Documents "
            "already indexed keep the chunking they were ingested with."
        )

        _apply_chunker(svc, strategy, chunk_size, overlap_pct, table_rows)

        if st.button("Re-chunk all documents at the current settings"):
            requeued = missing = 0
            for doc in svc["registry"].all():
                if doc.status.value != "done":
                    continue
                if svc["storage"].restore_to_inbox(doc.filename, doc.doc_id):
                    svc["registry"].requeue(doc.doc_id)
                    requeued += 1
                else:
                    missing += 1
            st.success(f"Re-queued {requeued} document(s) for re-chunking")
            if missing:
                st.warning(f"{missing} skipped — original file not found")
            st.rerun()

    return {"model": model, "temperature": temperature,
            "floor": floor, "vector_floor": vector_floor,
            "use_reranker": use_reranker,
            "enable_rewrite": enable_rewrite,
            "enable_multi_query": enable_multi_query,
            "enable_multi_hop": enable_multi_hop,
            "enable_self_correction": enable_self_correction}
