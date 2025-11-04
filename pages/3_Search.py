# newsletter_synthesis_app/pages/3_Search.py

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import json
from datetime import datetime
from utils import database, llm_processor
from config import DEFAULT_EMBEDDING_MODEL, PAGINATION_LIMIT

st.set_page_config(page_title="Semantic Search", layout="wide")

st.title("🔍 Semantic Search")
st.info("Ask a question or enter keywords to find the most relevant newsletters and summaries from your knowledge base.")

# --- Search Functions ---
@st.cache_data(ttl=3600)
def get_all_searchable_items():
    """Fetches and prepares all searchable items (newsletters and summaries) from the DB."""
    conn = database.get_db_connection()
    
    newsletters_df = pd.read_sql_query(
        """
        SELECT
            id,
            title,
            source,
            original_url,
            published_date AS date,
            readwise_summary,
            llm_summary,
            llm_tags,
            embedding
        FROM newsletters
        WHERE embedding IS NOT NULL
        """,
        conn,
    )
    newsletters_df['type'] = 'Newsletter'

    summaries_df = pd.read_sql_query(
        """
        SELECT
            id,
            generated_date AS date,
            content,
            themes_json,
            newsletter_count,
            embedding
        FROM summaries
        WHERE embedding IS NOT NULL
        """,
        conn,
    )
    summaries_df['type'] = 'Summary'
    
    conn.close()
    
    combined_df = pd.concat([newsletters_df, summaries_df], ignore_index=True)
    combined_df['embedding_np'] = combined_df['embedding'].apply(lambda x: np.array(x) if x is not None else None)
    combined_df.dropna(subset=['embedding_np'], inplace=True)
    
    return combined_df

def perform_search(query):
    """
    Performs efficient, vectorized semantic search across all items.
    """
    with st.spinner("Generating query embedding and searching..."):
        embedding_model = st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
        query_embedding = llm_processor.generate_embedding(query, embedding_model)

        if query_embedding is None:
            st.error("Could not generate embedding for the query. The embedding model may have failed.")
            return []

        search_space_df = get_all_searchable_items()
        if search_space_df.empty:
            st.warning("No items with embeddings found in the database to search against.")
            return []

        embedding_matrix = np.vstack(search_space_df['embedding_np'].values)
        query_embedding_np = np.array(query_embedding)
        
        # Normalize vectors for cosine similarity
        query_norm = query_embedding_np / np.linalg.norm(query_embedding_np)
        matrix_norm = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        
        # Compute similarities in a single, fast operation
        similarities = np.dot(matrix_norm, query_norm)
        
        search_space_df['relevance'] = similarities
        
        # Sort and return the top results as a list of dictionaries
        top_results = search_space_df.sort_values(by='relevance', ascending=False).head(50)
        return top_results.to_dict('records')

def _to_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return None

def _render_llm_tags(raw_tags: str):
    if not raw_tags:
        return
    try:
        data = json.loads(raw_tags)
    except (json.JSONDecodeError, TypeError):
        return

    tags = data.get('tags', [])
    if tags and isinstance(tags, list):
        st.markdown("**Tags:** " + ", ".join(tags))

    attributes = data.get('attributes', [])
    if isinstance(attributes, dict):
        attributes = [{"attribute": k, "grounding": v, "relevance": "N/A"} for k, v in attributes.items()]
    if attributes and isinstance(attributes, list):
        st.markdown("**AI Extracted Attributes:**")
        for item in attributes:
            attr = item.get('attribute', 'N/A')
            grounding = item.get('grounding', 'No grounding text.')
            relevance = item.get('relevance', 'N/A')
            st.markdown(f"- **{attr}**: {grounding} _(Relevance: {relevance})_")

def _render_newsletter_result(result, relevance_score, dt_obj):
    title = result.get('title') or "Untitled Newsletter"
    source = result.get('source') or "Unknown Source"
    expander_title = f"**{title}** (from: {source}) (Relevance: {relevance_score})"
    with st.expander(expander_title):
        meta_line = f"**Published:** {dt_obj.strftime('%Y-%m-%d') if dt_obj else 'Unknown'} | **Relevance:** {relevance_score}"
        if result.get('original_url'):
            meta_line += f" | **[Open in Readwise Reader]({result['original_url']})**"
        st.markdown(meta_line)
        st.markdown("---")

        if result.get('llm_summary'):
            st.markdown("**AI Generated Summary:**")
            st.write(result['llm_summary'])
        if result.get('readwise_summary'):
            st.markdown("**Readwise Summary:**")
            st.write(result['readwise_summary'])

        if result.get('llm_tags'):
            st.markdown("---")
            _render_llm_tags(result['llm_tags'])

def _render_summary_result(result, relevance_score, dt_obj):
    generated_on = dt_obj.strftime('%Y-%m-%d') if dt_obj else "Unknown Date"
    expander_title = f"**Summary Pack** (Generated: {generated_on}) (Relevance: {relevance_score})"
    with st.expander(expander_title):
        st.markdown(f"**Generated:** {generated_on} | **Relevance:** {relevance_score}")
        if result.get('newsletter_count'):
            st.markdown(f"**Covers:** {result['newsletter_count']} newsletters")
        st.markdown("---")
        st.write(result.get('content') or "No summary content available.")

# --- UI ---
search_query = st.text_input("Enter your search query:", key="search_query_input")

if st.button("Search", type="primary"):
    if search_query:
        st.session_state.search_results = perform_search(search_query)
    else:
        st.warning("Please enter a search query.")

if 'search_results' in st.session_state and st.session_state.search_results:
    results = st.session_state.search_results
    st.subheader(f"Found {len(results)} relevant results")
    for result in results:
        relevance_score = f"{result['relevance']:.2%}"

        dt_obj = _to_datetime(result.get('date'))
        if result.get('type') == 'Newsletter':
            _render_newsletter_result(result, relevance_score, dt_obj)
        else:
            _render_summary_result(result, relevance_score, dt_obj)

else:
    st.info("Enter a query and click Search to see results.")
