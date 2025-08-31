# newsletter_synthesis_app/pages/3_Search.py

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
from utils import database, llm_processor
from config import DEFAULT_EMBEDDING_MODEL

st.set_page_config(page_title="Semantic Search", layout="wide")

st.title("🔍 Semantic Search")
st.info("Ask a question or enter keywords to find the most relevant newsletters and summaries from your knowledge base.")

# --- Search Functions ---
def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def perform_search(query):
    """
    Performs semantic search across summaries and newsletters.
    """
    with st.spinner("Generating query embedding and searching..."):
        embedding_model = st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
        query_embedding = llm_processor.generate_embedding(query, embedding_model)

        if query_embedding is None:
            st.error("Could not generate embedding for the query. Please try again.")
            return

        conn = database.get_db_connection()
        
        # Search summaries
        summary_results = []
        summaries = conn.execute("SELECT id, generated_date, content, embedding FROM summaries WHERE embedding IS NOT NULL").fetchall()
        for row in summaries:
            if row['embedding']:
                sim = cosine_similarity(np.array(query_embedding), row['embedding'])
                summary_results.append({
                    "type": "Summary",
                    "id": row['id'],
                    "date": row['generated_date'],
                    "content": row['content'],
                    "relevance": sim
                })

        # Search newsletters
        newsletter_results = []
        newsletters = conn.execute("SELECT id, title, source, published_date, embedding FROM newsletters WHERE embedding IS NOT NULL").fetchall()
        for row in newsletters:
            if row['embedding']:
                sim = cosine_similarity(np.array(query_embedding), row['embedding'])
                newsletter_results.append({
                    "type": "Newsletter",
                    "id": row['id'],
                    "date": row['published_date'],
                    "content": f"**{row['title']}**\n\n*Source: {row['source']}*",
                    "relevance": sim
                })

        conn.close()

        all_results = sorted(summary_results + newsletter_results, key=lambda x: x['relevance'], reverse=True)
        st.session_state.search_results = all_results[:50] # Limit to top 50 results

# --- UI ---
search_query = st.text_input("Enter your search query:", key="search_query_input")

if st.button("Search", type="primary"):
    if search_query:
        perform_search(search_query)
    else:
        st.warning("Please enter a search query.")

if 'search_results' in st.session_state and st.session_state.search_results:
    results = st.session_state.search_results
    st.subheader(f"Found {len(results)} relevant results")

    for result in results:
        relevance_score = f"{result['relevance']:.2%}"

        # --- FIX: Parse string date before formatting ---
        date_val = result['date']
        if isinstance(date_val, str):
            dt_obj = datetime.fromisoformat(date_val)
        else:
            dt_obj = date_val

        expander_title = f"**[{result['type']}]** {result['content'].splitlines()[0]} (Relevance: {relevance_score})"
        
        with st.expander(expander_title):
            st.markdown(f"**Relevance Score:** {relevance_score}")
            st.markdown(f"**Date:** {dt_obj.strftime('%Y-%m-%d')}")
            st.markdown("---")
            st.markdown(result['content'], unsafe_allow_html=True)

else:
    st.info("Enter a query and click Search to see results.")