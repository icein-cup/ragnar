import logging
import uuid

import httpx
import streamlit as st

# No logging config existed anywhere in the app, so ingestion's per-stage
# timing (ingestion/pipeline.py) and worker failures (ingestion/worker.py)
# only ever reached stderr at WARNING+. INFO surfaces both.
logging.basicConfig(level=logging.INFO)

from generation.guards import aggregation_refusal
from generation.answerer import (
    AnswerMode,
    classify,
    NO_RESULTS_MESSAGE,
    citation_labels,
    build_citations,
)
from history.chat_store import chat_title
from ui.services import build_services
from ui.panels import settings, documents, chats

st.set_page_config(page_title="RAGnar", page_icon="📚")

svc = build_services()

try:
    httpx.get(f"{svc['cfg'].ollama_url}/api/tags", timeout=5).raise_for_status()
except Exception:
    st.error(
        f"Cannot reach Ollama at {svc['cfg'].ollama_url}.\n\n"
        "Start it with `ollama serve`, then confirm the models are present:\n"
        "`ollama pull qwen2.5:3b` and `ollama pull bge-m3`."
    )
    if st.button("Recheck"):
        st.rerun()
    st.stop()

SIDEBAR_HEADER_CSS = """
<style>
[class*="st-key-remove_container_"] button:hover,
[class*="st-key-del_chat_container_"] button:hover {
    background-color: #ff4b4b !important;
    color: white !important;
    border-color: #ff4b4b !important;
}
/* Make the Settings/Documents/Chats expander labels read like st.header */
div[data-testid="stExpander"] summary p {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
/* Citation links */
.citation-link {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
    background-color: #1e1e2e;
    border: 1px solid #4a4a6a;
    border-radius: 4px;
    color: #89b4fa;
    font-size: 0.8rem;
    text-decoration: none;
    cursor: pointer;
    transition: background-color 0.2s;
}
.citation-link:hover {
    background-color: #313244;
    color: #b4befe;
}
/* Agentic trace expander styling */
.agentic-trace {
    font-size: 0.8rem;
    color: #a6adc8;
}
.agentic-trace .trace-label {
    color: #cdd6f4;
    font-weight: 500;
}
</style>
"""


def _open_citation(doc_id: str, page: int | None, sheet: str | None) -> None:
    """Set session state so the Documents panel will jump to the cited location."""
    st.session_state[f"show_md_{doc_id}"] = True
    st.session_state[f"scroll_to_page_{doc_id}"] = page
    st.session_state[f"scroll_to_sheet_{doc_id}"] = sheet
    st.session_state["_focus_doc_id"] = doc_id


def _render_citations(citations: list) -> None:
    """Render clickable citation badges that open the source document."""
    for cite in citations:
        if isinstance(cite, dict):
            label = cite.get("label", cite)
            doc_id = cite.get("doc_id")
            page = cite.get("page")
            sheet = cite.get("sheet")
        else:
            label = str(cite)
            doc_id = None
            page = None
            sheet = None

        if doc_id:
            st.markdown(
                f'<a class="citation-link" href="#" '
                f'title="Open {label}">📄 {label}</a>',
                unsafe_allow_html=True,
            )
            # Streamlit button to actually open the document
            if st.button(
                f"Open {label}",
                key=f"cite_btn_{doc_id}_{page or 0}_{sheet or 'none'}",
                help=f"Open {label} at the referenced location",
            ):
                _open_citation(doc_id, page, sheet)
                st.rerun()
        else:
            st.caption(label)


def _render_agentic_trace(outcome) -> None:
    """Show the agentic reasoning trace in a collapsible section."""
    trace_parts: list[str] = []
    if outcome.rewritten_query:
        trace_parts.append(f"Rewritten query: {outcome.rewritten_query}")
    if outcome.queries_executed:
        trace_parts.append(f"Queries executed: {len(outcome.queries_executed)}")
    if outcome.hops_performed:
        trace_parts.append(f"Multi-hops performed: {outcome.hops_performed}")
    if outcome.self_corrected:
        trace_parts.append(f"Self-corrected: {outcome.correction_notes or 'yes'}")

    if trace_parts:
        with st.expander("🧠 Agentic reasoning trace", expanded=False):
            for part in trace_parts:
                st.markdown(
                    f"<div class='agentic-trace'><span class='trace-label'>{part}</span></div>",
                    unsafe_allow_html=True,
                )


with st.sidebar:
    st.title("RAGnar - local RAG chat app")
    st.markdown(SIDEBAR_HEADER_CSS, unsafe_allow_html=True)

    query = settings.render(svc)
    documents.render(svc)
    chats.render(svc)

all_docs = svc["registry"].all()
if not all_docs:
    st.info("No documents indexed yet. Upload one to get started.")

selected_doc_ids = [
    d.doc_id for d in all_docs if st.session_state.get(f"sel_{d.doc_id}", True)
]
# None = unfiltered search (identical to today's behavior) whenever
# everything happens to be selected; only pass an explicit filter — which
# may be an empty list, correctly refusing — for a genuine subset.
doc_ids_filter = (
    None if set(selected_doc_ids) == {d.doc_id for d in all_docs} else selected_doc_ids
)

if "messages" not in st.session_state:
    st.session_state.messages = []
# None until the current conversation has been saved for the first time.
st.session_state.setdefault("current_chat_id", None)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                _render_citations(message["citations"])

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Build conversation history from all messages before the
        # current question (which was just appended). Strip the
        # app-only 'citations' key — the LLM doesn't need it.
        previous = st.session_state.messages[:-1]
        history = [{k: v for k, v in m.items() if k != "citations"} for m in previous]
        context_summary = svc["answerer"].summarize_history(
            history, model=query["model"]
        )

        # Use agentic search if available
        if svc.get("agentic_search") is not None:
            svc["agentic_search"]._enable_rewrite = query["enable_rewrite"]
            svc["agentic_search"]._enable_multi_query = query["enable_multi_query"]
            svc["agentic_search"]._enable_multi_hop = query["enable_multi_hop"]
            svc["agentic_search"]._enable_self_correction = query[
                "enable_self_correction"
            ]
            outcome = svc["agentic_search"].find(
                question,
                doc_ids=doc_ids_filter,
                score_floor=query["floor"],
                vector_floor=query["vector_floor"],
                use_reranker=query["use_reranker"],
                context_summary=context_summary,
                history=history,
                model=query["model"],
                temperature=query["temperature"],
            )
        else:
            outcome = svc["search"].find(
                question,
                doc_ids=doc_ids_filter,
                score_floor=query["floor"],
                vector_floor=query["vector_floor"],
                use_reranker=query["use_reranker"],
                context_summary=context_summary,
            )

        mode = classify(question, outcome.refused, outcome.results)

        if mode is AnswerMode.NO_RESULTS:
            citations = []
            rich_citations = []

            related_labels = citation_labels(outcome.related)
            if related_labels:
                with st.expander("Related documents you might check"):
                    for label in related_labels:
                        st.caption(label)

            # When there is conversation history, fall back to a
            # conversational answer so the assistant can recall things the
            # user said earlier (e.g. their name) even though no documents
            # matched. Without history there is nothing to recall, so keep
            # the canned refusal.
            if len(history) >= 2:
                text = st.write_stream(
                    svc["answerer"].converse_stream(
                        question,
                        model=query["model"],
                        temperature=query["temperature"],
                        history=history,
                    )
                )
            else:
                text = NO_RESULTS_MESSAGE
                st.markdown(text)
        elif mode is AnswerMode.AGGREGATION_REFUSED:
            text = aggregation_refusal(outcome.results)
            st.warning(text)
            citations = []
            rich_citations = []
        else:
            # Build rich citations with navigation metadata
            rich_citations = build_citations(outcome.results)
            citations = [c.label for c in rich_citations]

            # Reuse the self-correction draft when available — it was
            # built with the same prompt/context as Answerer.stream
            # would use, so displaying it directly avoids a duplicate
            # LLM call without any quality drift.
            if getattr(outcome, "draft_answer", None):
                text = outcome.draft_answer
                st.markdown(text)
            else:
                text = st.write_stream(
                    svc["answerer"].stream(
                        question,
                        outcome.results,
                        model=query["model"],
                        temperature=query["temperature"],
                        history=history,
                        context_summary=context_summary,
                    )
                )
            with st.expander("Sources"):
                _render_citations(
                    [
                        {
                            "label": c.label,
                            "doc_id": c.doc_id,
                            "page": c.page,
                            "sheet": c.sheet,
                        }
                        for c in rich_citations
                    ]
                )

        # Render agentic trace if available
        if hasattr(outcome, "hops_performed"):
            _render_agentic_trace(outcome)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "citations": rich_citations if "rich_citations" in locals() else citations,
        }
    )

    # Persist the conversation. Mint an id on first save so a chat only
    # appears in the list once it actually has content. Rerun so the newly
    # saved/updated chat shows in the Chats panel immediately (the sidebar
    # renders above this handler, so it hasn't seen the save yet this run).
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = uuid.uuid4().hex
    svc["chats"].save(
        st.session_state.current_chat_id,
        chat_title(st.session_state.messages),
        st.session_state.messages,
    )
    st.rerun()