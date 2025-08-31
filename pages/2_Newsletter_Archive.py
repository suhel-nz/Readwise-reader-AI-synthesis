# newsletter_synthesis_app/pages/2_Newsletter_Archive.py

import streamlit as st
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import database
from config import PAGINATION_LIMIT

st.set_page_config(page_title="Newsletter Archive", layout="wide")

st.title("🗄️ Newsletter Archive")

# --- Filters ---
conn = database.get_db_connection()
sources = [row['source'] for row in conn.execute("SELECT DISTINCT source FROM newsletters WHERE source IS NOT NULL ORDER BY source").fetchall()]
categories = [row['category'] for row in conn.execute("SELECT DISTINCT category FROM newsletters ORDER BY category").fetchall()]
# Fetching LLM tags requires more complex parsing; we'll keep it simple for now
conn.close()

col1, col2, col3 = st.columns(3)
with col1:
    selected_sources = st.multiselect("Filter by Source", options=sources)
with col2:
    selected_categories = st.multiselect("Filter by Category", options=categories)
with col3:
    search_term = st.text_input("Search by Title")

# --- Data Fetching ---
query = "SELECT id, title, source, published_date, processed_date, category, llm_tags, original_url FROM newsletters WHERE 1=1"
params = []

if selected_sources:
    query += f" AND source IN ({','.join(['?'] * len(selected_sources))})"
    params.extend(selected_sources)
if selected_categories:
    query += f" AND category IN ({','.join(['?'] * len(selected_categories))})"
    params.extend(selected_categories)
if search_term:
    query += " AND title LIKE ?"
    params.append(f"%{search_term}%")

query += " ORDER BY published_date DESC"

# --- Pagination ---
if 'archive_page' not in st.session_state:
    st.session_state.archive_page = 1

data, total_items = database.fetch_paginated_data(query, tuple(params), st.session_state.archive_page, PAGINATION_LIMIT)
total_pages = (total_items + PAGINATION_LIMIT - 1) // PAGINATION_LIMIT

# --- Display Data ---
if not data:
    st.info("No newsletters found matching your criteria.")
else:
    st.write(f"Showing {len(data)} of {total_items} newsletters.")
    
    df = pd.DataFrame(data, columns=['ID', 'Title', 'Source', 'Published', 'Processed', 'Category', 'LLM Tags', 'URL'])
    df['Published'] = pd.to_datetime(df['Published']).dt.strftime('%Y-%m-%d')
    df['Processed'] = pd.to_datetime(df['Processed']).dt.strftime('%Y-%m-%d')
    
    # Create a display version of the dataframe for st.dataframe
    display_df = df[['Title', 'Source', 'Published', 'Category']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Expander for details
    with st.expander("View Details for Selected Page"):
        for index, row in df.iterrows():
            st.markdown(f"#### {row['Title']}")
            st.markdown(f"**Source:** {row['Source']} | **Published:** {row['Published']} | [Original Link]({row['URL']})")
            try:
                llm_tags_data = json.loads(row['LLM Tags'])
                st.json(llm_tags_data, expanded=False)
            except (json.JSONDecodeError, TypeError):
                st.write("Could not display LLM tags.")
            st.divider()


    # Pagination controls
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.archive_page <= 1)):
            st.session_state.archive_page -= 1
            st.rerun()
    with page_col3:
        if st.button("Next ➡️", disabled=(st.session_state.archive_page >= total_pages)):
            st.session_state.archive_page += 1
            st.rerun()
    with page_col2:
        st.write(f"Page {st.session_state.archive_page} of {total_pages}")