import html
import logging
import time
import uuid

import httpx
import streamlit as st

# No logging config existed anywhere in the app, so ingestion's per-stage
# timing (ingestion/pipeline.py) and worker failures (ingestion/worker.py)
# only ever reached stderr at WARNING+. INFO surfaces both.
logging.basicConfig(level=logging.INFO)

from generation.guards import (aggregation_refusal, is_refusal,
                               refusal_text, strip_no_answer)
from generation.answerer import (
    AnswerMode,
    AnswerStream,
    classify,
    NO_RESULTS_MESSAGE,
    citation_labels,
    build_citations,
)
from history.chat_store import chat_title, dataclass_to_dict
from ui.services import build_services
from ui.static_files import file_url
from ui.panels import settings, documents, chats

st.set_page_config(page_title="RAGnar", page_icon="🪓")

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
    """Fall back to the Documents panel markdown viewer for a cited source.

    PDFs are handled as direct links in _render_citations, so this only
    covers non-PDF sources (spreadsheets, etc.) that have no page to jump
    to in a browser viewer.
    """
    st.session_state[f"show_md_{doc_id}"] = True
    st.session_state[f"scroll_to_page_{doc_id}"] = page
    st.session_state[f"scroll_to_sheet_{doc_id}"] = sheet


def _render_citations(citations: list, scope: str = "live") -> None:
    """Render clickable citation badges that open the source document.

    PDF sources with a page are rendered as direct links to the archived
    file with a ``#page=N`` fragment, which the browser's PDF viewer honors —
    this works from inside the Docker container, where host-side ``open``
    commands don't exist. Everything else falls back to the Documents panel
    markdown viewer.

    scope disambiguates the button key across messages — citation_label()
    dedupes by filename+page, so the same source cited in two different
    chat turns would otherwise produce the same widget key and crash the
    replay loop with a duplicate-element-key error.
    """
    for idx, cite in enumerate(citations):
        if isinstance(cite, dict):
            label = cite.get("label", cite)
            doc_id = cite.get("doc_id")
            filename = cite.get("filename")
            page = cite.get("page")
            sheet = cite.get("sheet")
        else:
            label = str(cite)
            doc_id = None
            filename = None
            page = None
            sheet = None

        if not doc_id:
            st.caption(label)
            continue

        is_pdf = bool(filename) and str(filename).lower().endswith(".pdf")
        if is_pdf:
            archived_name = svc["storage"].archived_path(filename, doc_id).name
            url = file_url(svc["file_base_url"], archived_name, page)
            st.markdown(
                f"📄 <a href='{url}' target='_blank'>{html.escape(str(label))}</a>",
                unsafe_allow_html=True,
            )
        else:
            if st.button(
                f"📄 {label}",
                key=f"cite_btn_{scope}_{idx}",
                help=f"Open {label} at the referenced location",
            ):
                _open_citation(doc_id, page, sheet)
                st.rerun()


def _render_agentic_trace(outcome, elapsed: float | None = None) -> None:
    """Show the query timing and, when available, the agentic reasoning trace.

    Accepts either a live AgenticSearchOutcome or the dict snapshot
    _build_trace() persisted with the message — _build_trace uses the same
    field names, so one lookup covers both.
    """
    trace = outcome if isinstance(outcome, dict) else _build_trace(outcome) or {}

    trace_parts: list[str] = []
    if elapsed is not None:
        trace_parts.append(f"Round-trip time: {elapsed:.2f}s")
    if trace.get("rewritten_query"):
        trace_parts.append(f"Rewritten query: {trace['rewritten_query']}")
    if trace.get("queries_executed"):
        trace_parts.append(f"Queries executed: {trace['queries_executed']}")
    if trace.get("hops_performed"):
        trace_parts.append(f"Multi-hops performed: {trace['hops_performed']}")
    if trace.get("self_corrected"):
        trace_parts.append(
            f"Self-corrected: {trace.get('correction_notes') or 'yes'}"
        )

    if trace_parts:
        with st.expander("🧠 Agentic reasoning trace", expanded=False):
            for part in trace_parts:
                st.markdown(
                    f"<div class='agentic-trace'><span class='trace-label'>{part}</span></div>",
                    unsafe_allow_html=True,
                )


def _citations_to_dicts(citations: list) -> list[dict]:
    """Citation dataclasses as plain dicts, JSON-safe for session state."""
    return [c if isinstance(c, dict) else dataclass_to_dict(c) for c in citations]


def _render_sources_expander(citations: list, related: list | None = None, scope: str = "live") -> None:
    """Render the Sources panel: real citations, related docs, or a no-match note."""
    with st.expander("Sources"):
        if citations:
            _render_citations(citations, scope=scope)
        if related:
            st.markdown(
                "<div class='agentic-trace'>Related documents that did not "
                "clear the relevance floor:</div>",
                unsafe_allow_html=True,
            )
            for label in related:
                st.caption(label)
        if not citations and not related:
            st.caption(
                "No documents matched this question above the current "
                "relevance floor."
            )


def _build_trace(outcome) -> dict | None:
    """Build a JSON-serializable agentic trace snapshot from a search outcome."""
    trace: dict = {}
    if getattr(outcome, "rewritten_query", None):
        trace["rewritten_query"] = outcome.rewritten_query
    if getattr(outcome, "queries_executed", None):
        trace["queries_executed"] = len(outcome.queries_executed)
    if getattr(outcome, "hops_performed", None):
        trace["hops_performed"] = outcome.hops_performed
    if getattr(outcome, "self_corrected", None):
        trace["self_corrected"] = True
        if getattr(outcome, "correction_notes", None):
            trace["correction_notes"] = outcome.correction_notes
    return trace if trace else None


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

# Reconcile background grounding verdicts that landed after the last rerun.
# answer_async/ground_async run the fabrication check on a daemon thread and
# write the verdict onto an Answer object; that object outlives the rerun, so
# we re-read it here and fold any finished verdict into the persisted message.
_pending = st.session_state.get("pending_grounding", {})
if _pending:
    _still_pending = {}
    _dirty = False
    for _idx, _answer in _pending.items():
        if _answer.grounded is None:
            _still_pending[_idx] = _answer
        else:
            _dirty = True
            if _idx < len(st.session_state.messages):
                st.session_state.messages[_idx]["grounded"] = _answer.grounded
    st.session_state.pending_grounding = _still_pending
    if _dirty and st.session_state.current_chat_id is not None:
        try:
            svc["chats"].save(
                st.session_state.current_chat_id,
                chat_title(st.session_state.messages),
                st.session_state.messages,
            )
        except Exception as exc:
            logging.exception("Failed to persist grounding verdict: %s", exc)

_VIKING_AVATAR = "🪓"

for _msg_idx, message in enumerate(st.session_state.messages):
    avatar = _VIKING_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("grounded") is False:
            st.warning(
                "⚠️ This answer may not be fully supported by the retrieved "
                "documents — it was flagged by the grounding check."
            )
        _render_sources_expander(
            message.get("citations") or [],
            message.get("related"),
            scope=str(_msg_idx),
        )
        _render_agentic_trace(
            message.get("trace") or {}, elapsed=message.get("elapsed")
        )

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=_VIKING_AVATAR):
        start = time.perf_counter()

        # Build conversation history from all messages before the current
        # question (which was just appended). Whitelist role/content rather
        # than blacklisting 'citations': a stored message also carries
        # 'related', 'elapsed' and 'trace', and blacklisting one key sent the
        # other three into the chat payload. Ollama ignores unknown message
        # fields, but an OpenAI-compatible endpoint rejects them outright.
        previous = st.session_state.messages[:-1]
        history = [
            {"role": m["role"], "content": m["content"]} for m in previous
        ]
        context_summary = svc["answerer"].summarize_history(
            history, model=query["model"]
        )

        # st.spinner's context entry triggers a frontend flush so the spinner
        # reaches the browser before the blocking search call starts.
        with st.spinner("🪓 RAGnar is running through your documents…"):
            # Use agentic search if available. Flags are passed per-call, not
            # mutated on the shared (st.cache_resource) instance — see find()'s
            # docstring in retrieval/agentic.py: concurrent sessions would
            # otherwise clobber each other's settings mid-request.
            if svc.get("agentic_search") is not None:
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
                    enable_rewrite=query["enable_rewrite"],
                    enable_multi_query=query["enable_multi_query"],
                    enable_multi_hop=query["enable_multi_hop"],
                    enable_self_correction=query["enable_self_correction"],
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

        related_labels = citation_labels(outcome.related)
        related_dicts = [
            {"label": label, "doc_id": None, "page": None, "sheet": None}
            for label in related_labels
        ]

        # Only the ANSWER branch spawns a grounding check; the other branches
        # leave this None so the commit below skips registration.
        grounded_answer = None

        if mode is AnswerMode.NO_RESULTS:
            citations = []
            rich_citations = []

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
            # Reuse the self-correction draft when available — it was
            # built with the same prompt/context as Answerer.stream
            # would use, so displaying it directly avoids a duplicate
            # LLM call without any quality drift.
            if getattr(outcome, "draft_answer", None):
                text = outcome.draft_answer
                # Read the sentinel before removing it — see AnswerStream.
                refused = is_refusal(text)
                text = refusal_text(text) if refused else strip_no_answer(text)
                st.markdown(text)
            else:
                # AnswerStream keeps the NO_ANSWER sentinel off the screen.
                # st.write_stream paints deltas as they arrive, so the token
                # cannot be stripped from the finished text — by then it has
                # already been shown — and once stripped is_refusal can no
                # longer see it. The wrapper carries that verdict out.
                answer_stream = AnswerStream(
                    svc["answerer"].stream(
                        question,
                        outcome.results,
                        model=query["model"],
                        temperature=query["temperature"],
                        history=history,
                        context_summary=context_summary,
                    )
                )
                text = st.write_stream(answer_stream)
                refused = answer_stream.refused or is_refusal(text)

            # Citations are chosen AFTER the answer exists: a model that says
            # the excerpts do not cover the question gets none. Covers the
            # streamed answer and the reused draft alike. Pruned to only
            # chunks whose text overlaps the answer (see prune_citations).
            rich_citations = [] if refused else build_citations(outcome.results, str(text))
            citations = [c.label for c in rich_citations]

            # Kick off the fabrication check in the background. It never blocks
            # the answer; the verdict lands on the Answer object and is folded
            # into the persisted message on a later rerun (see the reconcile
            # block above). Refusals are skipped — a "the excerpts do not
            # cover this" sentence would be judged UNSUPPORTED against the
            # excerpts, which is noise, not a fabrication.
            if not refused:
                grounded_answer = svc["answerer"].ground_async(
                    text, outcome.results, model=query["model"]
                )

        elapsed = time.perf_counter() - start
        logging.info("query_round_trip took %.2fs (question=%r)", elapsed, question)

        # Commit the answer to session state (and disk) BEFORE any further
        # rendering below. A streamed answer only survives st.rerun() if it
        # made it into st.session_state.messages — the replay loop at the
        # top of the script is the only thing that redraws it. If a later
        # render call (sources/timing/trace) throws or a background rerun
        # request lands, the answer must already be safe, or it's gone for
        # good and the next full render falls back to whatever was last
        # committed (i.e. the refusal from a previous turn).
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": text,
                "citations": _citations_to_dicts(rich_citations),
                "related": related_dicts,
                "elapsed": elapsed,
                "trace": _build_trace(outcome),
                "grounded": None,
            }
        )

        # Register the in-flight grounding check so a later rerun can fold its
        # verdict into the message above. The Answer object is a plain Python
        # object (not session state), so it survives the rerun and the daemon
        # thread keeps writing to it.
        if grounded_answer is not None:
            _msg_idx = len(st.session_state.messages) - 1
            st.session_state.setdefault("pending_grounding", {})[_msg_idx] = grounded_answer

        # Persist the conversation. Mint an id on first save so a chat only
        # appears in the list once it actually has content.
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = uuid.uuid4().hex
        try:
            svc["chats"].save(
                st.session_state.current_chat_id,
                chat_title(st.session_state.messages),
                st.session_state.messages,
            )
        except Exception as exc:
            logging.exception("Failed to persist chat: %s", exc)
            st.warning("Couldn't save this chat — the answer above is still shown, but a page reload may lose it.")

        _render_sources_expander(_citations_to_dicts(rich_citations), related_dicts)
        _render_agentic_trace(outcome, elapsed=elapsed)

    # Rerun so the newly saved/updated chat shows in the Chats panel
    # immediately (the sidebar renders above this handler, so it hasn't
    # seen the save yet this run).
    st.rerun()
