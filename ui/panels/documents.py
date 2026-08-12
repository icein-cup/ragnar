import streamlit as st
from pathlib import Path

from ui.services import format_eta

STATUS_ICONS = {"queued": "⏳", "processing": "⚙️", "done": "✅", "failed": "❌"}


def _get_original_path(svc, doc_id: str, filename: str) -> Path | None:
    """Get the path to the original uploaded file for a document."""
    path = svc["storage"].archived_path(filename, doc_id)
    return path if path.exists() else None


def _open_file_at_page(path: Path, page: int | None = None) -> None:
    """Open the original file using the system default application.

    For PDFs, attempts to open at the specific page using the #page=N URL
    fragment (honored by macOS Preview and most PDF viewers) or a viewer-
    specific flag as a fallback.
    """
    import platform
    import subprocess
    import urllib.parse

    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            if page and path.suffix.lower() == ".pdf":
                # Preview and most macOS PDF viewers honor the #page=N
                # fragment on a file:// URL. This opens the document and
                # jumps straight to the cited page.
                file_url = (
                    "file://"
                    + urllib.parse.quote(str(path))
                    + f"#page={page}"
                )
                subprocess.run(
                    ["open", file_url], check=False, capture_output=True
                )
            else:
                subprocess.run(["open", str(path)], check=False, capture_output=True)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(path)], check=False)
        elif system == "Windows":
            subprocess.run(["start", "", str(path)], check=False, shell=True)
    except Exception:
        pass  # Best-effort: if opening fails, the user can still view markdown


def render(svc) -> None:
    with st.expander("Documents", expanded=False):
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded = st.file_uploader(
            "Upload", type=["pdf", "xlsx", "docx"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if uploaded and st.button("Upload", type="primary"):
            for file in uploaded:
                target = svc["storage"].inbox / file.name
                target.write_bytes(file.getbuffer())
                svc["registry"].add(svc["storage"].doc_id(target), file.name,
                                    target.stat().st_size)
            # Force the uploader widget to reset to empty on the next render.
            st.session_state.uploader_key += 1
            st.rerun()

        # Files placed directly in the watched folder (outside the browser
        # upload widget) have no registry entry yet - only show the manual
        # ingest option when there's actually something like that to pick up.
        externally_dropped = [
            p for p in svc["storage"].pending_files()
            if svc["registry"].get(svc["storage"].doc_id(p)) is None
        ]
        if externally_dropped:
            st.caption(
                f"{len(externally_dropped)} file(s) found in the watched "
                "folder, not added via upload"
            )
            if st.button("Ingest inbox"):
                for path in externally_dropped:
                    svc["registry"].add(svc["storage"].doc_id(path), path.name,
                                        path.stat().st_size)
                st.success(f"Queued {len(externally_dropped)} file(s)")
                st.rerun()

        _status_strip(svc)


def _status_strip(svc) -> None:
    # Poll only while ingestion is actually running. run_every is baked
    # into the fragment decorator, so it can't be toggled at call time —
    # instead this recomputes it and re-applies the decorator on every
    # full script rerun (the pattern Streamlit's own docs use for
    # starting/stopping a fragment's auto-rerun). Polling unconditionally
    # every 2s would request a rerun in the middle of a chat answer
    # streaming elsewhere on the page (st.write_stream checks for a
    # pending rerun on every token) and silently kill it before it's saved.
    processing, queued, _ = svc["registry"].ingest_eta()
    run_every = "2s" if (processing or queued) else None

    @st.fragment(run_every=run_every)
    def _strip() -> None:
        _render_status_strip(svc)

    _strip()


def _render_status_strip(svc) -> None:
    processing, queued, eta = svc["registry"].ingest_eta()

    if processing or queued:
        st.info(
            f"Indexing — {processing} in progress, "
            f"{queued} queued{format_eta(eta)}"
        )

    docs = svc["registry"].all()

    def _select_all_changed():
        value = st.session_state.get("select_all_docs", True)
        for d in docs:
            st.session_state[f"sel_{d.doc_id}"] = value

    if docs:
        st.checkbox("Select all", value=True, key="select_all_docs",
                    on_change=_select_all_changed)
        st.caption(
            "Unchecked documents are excluded from answers — only "
            "checked ones are searched."
        )

    for doc in docs:
        icon = STATUS_ICONS[doc.status.value]

        col_check, col_view, col_remove = st.columns([0.6, 4.4, 1])
        with col_check:
            st.checkbox(f"Include {doc.filename}", value=True,
                        key=f"sel_{doc.doc_id}",
                        label_visibility="collapsed")
        with col_view:
            if st.button(f"{icon} {doc.filename}", key=f"view_{doc.doc_id}",
                         use_container_width=True):
                show_key = f"show_md_{doc.doc_id}"
                st.session_state[show_key] = not st.session_state.get(show_key, False)
        with col_remove:
            with st.container(key=f"remove_container_{doc.doc_id}"):
                if st.button("✕", key=f"rm_{doc.doc_id}",
                             help=f"Remove {doc.filename}"):
                    svc["store"].delete_by_doc(doc.doc_id)
                    svc["storage"].remove_converted(doc.doc_id)
                    svc["registry"].remove(doc.doc_id)
                    st.rerun()

        if doc.error:
            err_col, retry_col = st.columns([4, 1])
            with err_col:
                st.caption(f"↳ {doc.error}")
            if doc.status.value == "failed":
                with retry_col:
                    if st.button("🔄", key=f"retry_{doc.doc_id}",
                                 help=f"Retry ingesting {doc.filename}"):
                        # Failed documents are never archived - the original
                        # is still sitting in inbox, so requeuing alone is
                        # enough to retry.
                        svc["registry"].requeue(doc.doc_id)
                        st.rerun()

        if doc.status.value == "done" and st.session_state.get(f"show_md_{doc.doc_id}"):
            _render_document_viewer(svc, doc.doc_id, doc.filename)


def _render_document_viewer(svc, doc_id: str, filename: str) -> None:
    """Render the document viewer with optional page/sheet navigation."""
    original_path = _get_original_path(svc, doc_id, filename)
    # Read (not pop) the navigation targets so they survive across reruns
    # until the user actually clicks "Open" — popping on the first render
    # meant the button click on the next rerun always saw None.
    page_target = st.session_state.get(f"scroll_to_page_{doc_id}")
    sheet_target = st.session_state.get(f"scroll_to_sheet_{doc_id}")

    # Open original file button (top of viewer)
    if original_path:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{filename}**")
        with cols[1]:
            open_label = "Open original"
            if page_target:
                open_label = f"Open at page {page_target}"
            elif sheet_target:
                open_label = f"Open sheet {sheet_target}"
            if st.button(open_label, key=f"open_orig_{doc_id}"):
                _open_file_at_page(original_path, page_target)
                # Clear the navigation target after opening so a later
                # manual "Open original" click doesn't jump to a stale page.
                st.session_state.pop(f"scroll_to_page_{doc_id}", None)
                st.session_state.pop(f"scroll_to_sheet_{doc_id}", None)
                st.rerun()
    else:
        st.markdown(f"**{filename}**")
        st.caption("Original file not found — showing converted text only")

    # Show converted markdown
    markdown = svc["storage"].read_markdown(doc_id)
    if markdown:
        # If a specific page/sheet is targeted, try to show a jump indicator
        if page_target:
            st.info(f"📍 Jumped to content from page {page_target} — scroll to find the relevant section below.")
            st.markdown(markdown)
        elif sheet_target:
            st.info(f"📍 Jumped to sheet {sheet_target} — scroll to find the relevant section below.")
            st.markdown(markdown)
        else:
            st.markdown(markdown)
    else:
        st.markdown("_Not yet converted_")
