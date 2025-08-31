# newsletter_synthesis_app/pages/1_Historical_Summaries.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import database
from config import PAGINATION_LIMIT

st.set_page_config(page_title="Historical Summaries", layout="wide")

st.title("📖 Historical Summaries")

# --- Filters ---
conn = database.get_db_connection()
themes = [row['name'] for row in conn.execute("SELECT DISTINCT name FROM themes ORDER BY name").fetchall()]
conn.close()

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_date = st.date_input("From Date", datetime.now() - timedelta(days=30))
with col2:
    end_date = st.date_input("To Date", datetime.now())
with col3:
    selected_themes = st.multiselect("Filter by Themes", options=themes)

start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.max.time())

# --- Data Fetching ---
query = "SELECT id, generated_date, newsletter_count, themes_json, content FROM summaries WHERE generated_date BETWEEN ? AND ?"
params = [start_datetime, end_datetime]

if selected_themes:
    # This is a simple LIKE search. For complex JSON, this might need refinement.
    theme_clauses = " AND ".join(["themes_json LIKE ?"] * len(selected_themes))
    query += f" AND ({theme_clauses})"
    params.extend([f'%"{theme}"%' for theme in selected_themes])

query += " ORDER BY generated_date DESC"

# --- Pagination ---
if 'summary_page' not in st.session_state:
    st.session_state.summary_page = 1

data, total_items = database.fetch_paginated_data(query, tuple(params), st.session_state.summary_page, PAGINATION_LIMIT)
total_pages = (total_items + PAGINATION_LIMIT - 1) // PAGINATION_LIMIT

# --- Display Data ---
if not data:
    st.info("No summaries found matching your criteria.")
else:
    st.write(f"Showing {len(data)} of {total_items} summaries.")

    for row in data:
        with st.expander(f"**Summary ID: {row['id']}** - Generated on {row['generated_date'].strftime('%Y-%m-%d %H:%M')} - ({row['newsletter_count']} articles)"):
            st.markdown(row['content'], unsafe_allow_html=True)

    # Pagination controls
    st.divider()
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.summary_page <= 1)):
            st.session_state.summary_page -= 1
            st.rerun()
    with page_col3:
        if st.button("Next ➡️", disabled=(st.session_state.summary_page >= total_pages)):
            st.session_state.summary_page += 1
            st.rerun()
    with page_col2:
        st.write(f"Page {st.session_state.summary_page} of {total_pages}")