import streamlit as st
import pandas as pd
import json
from utils import database
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="Syntheses Archive", layout="wide")

st.title("Syntheses Archive")
st.info("Browse past syntheses. Simple, paginated list with quick preview.")

if 'synth_archive_page' not in st.session_state:
    st.session_state.synth_archive_page = 1

PAGE_SIZE = 10

def _render_summary_tabs(summary_row: dict):
    tabs = st.tabs(["Final", "Original"])
    with tabs[0]:
        if summary_row.get('content'):
            _render_copy_button(summary_row.get('content'), label="Copy Final")
            st.markdown(summary_row['content'], unsafe_allow_html=True)
        else:
            st.info("No final content available.")
    with tabs[1]:
        if summary_row.get('draft_content'):
            _render_copy_button(summary_row.get('draft_content'), label="Copy Original")
            st.markdown(summary_row['draft_content'], unsafe_allow_html=True)
        else:
            st.info("No original draft available.")

def _render_copy_button(text: str, label: str = "Copy"):
    if not text:
        return
    b64 = base64.b64encode(text.encode('utf-8')).decode()
    components.html(
        f"""
        <div>
            <button style=\"margin-bottom:8px;\" onclick=\"navigator.clipboard.writeText(atob('{b64}'))\">{label}</button>
        </div>
        """,
        height=40,
    )

query = """
SELECT id, generated_date, newsletter_count, themes_json, content, draft_content, generation_context_json
FROM summaries
ORDER BY generated_date DESC
"""

rows, total = database.fetch_paginated_data(query, params=(), page=st.session_state.synth_archive_page, page_size=PAGE_SIZE)

if not rows:
    st.info("No syntheses found.")
else:
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    st.write(f"Page {st.session_state.synth_archive_page} of {total_pages} ({total} total)")

    for r in rows:
        row = dict(r)
        # Try to show theme list; fallback to preview of final content
        themes = []
        try:
            themes_data = json.loads(row.get('themes_json') or '[]')
            if isinstance(themes_data, list):
                themes = [t.get('theme') for t in themes_data if isinstance(t, dict) and t.get('theme')]
        except json.JSONDecodeError:
            themes = []
        title_suffix = f"Themes: {', '.join(themes)}" if themes else (row.get('content') or '')[:100]
        # Add a small warning mark if citation issues were recorded
        warn = ""
        try:
            ctx = json.loads(row.get('generation_context_json') or '{}')
            if ctx.get('citation_issues'):
                warn = " ⚠ citations"
        except json.JSONDecodeError:
            pass
        with st.expander(f"{row['generated_date']} • {row['newsletter_count']} articles • ID {row['id']} — {title_suffix}{warn}"):
            _render_summary_tabs(row)

    col_prev, _, col_next = st.columns([1,8,1])
    with col_prev:
        st.button("Previous", disabled=(st.session_state.synth_archive_page <= 1), on_click=lambda: st.session_state.update({"synth_archive_page": st.session_state.synth_archive_page - 1}))
    with col_next:
        st.button("Next", disabled=(st.session_state.synth_archive_page >= total_pages), on_click=lambda: st.session_state.update({"synth_archive_page": st.session_state.synth_archive_page + 1}))
