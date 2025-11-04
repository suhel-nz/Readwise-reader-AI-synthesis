# newsletter_synthesis_app/pages/2_Newsletter_Archive.py
# (Works, snappy, has buttons, AND now has attribute search)

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from utils import database, casting, logger, helpers
from config import PAGINATION_LIMIT
MAX_BUTTONS_PER_ROW = 5

# --- SETUP: Logging and Page Config ---
log = logger.get_logger()
st.set_page_config(page_title="Newsletter Archive", layout="wide")

log.info("--- Top of Newsletter_Archive.py script run ---")

# --- CALLBACKS & HELPER FUNCTIONS (No changes needed here) ---

def set_detail_view(article_id):
    """Callback to set the view to a single article."""
    log.info(f"Callback triggered: set_detail_view with article_id={article_id}")
    st.session_state.archive_filters['article_id'] = article_id
    st.session_state.archive_page = 1

def set_tag_filter(tag):
    """Callback to set the tag filter."""
    log.info(f"Callback triggered: set_tag_filter with tag='{tag}'")
    st.session_state.archive_filters['tag'] = tag
    st.session_state.archive_page = 1

def clear_all_filters():
    """Callback to reset all filters to their default state."""
    log.info("Callback triggered: clear_all_filters")
    st.session_state.archive_filters = {
        "article_id": None, "tag": "", "keyword": "", "sources": [], "categories": [],
        "date_range": (datetime.now() - timedelta(days=30), datetime.now().date())
    }
    st.session_state.archive_page = 1

def render_metadata(llm_tags_str: str):
    """Renders tags as clickable buttons and attributes in a structured way."""
    try:
        data = json.loads(llm_tags_str)
        tags = data.get('tags', [])
        if tags and isinstance(tags, list):
            st.write("**Tags:**")
            for row_start in range(0, len(tags), MAX_BUTTONS_PER_ROW):
                row_tags = tags[row_start:row_start + MAX_BUTTONS_PER_ROW]
                cols = st.columns(len(row_tags))
                for i, tag in enumerate(row_tags):
                    cols[i].button(
                        tag, key=f"tag_{tag}_{st.session_state.widget_key_suffix}_{row_start}",
                        on_click=set_tag_filter, args=(tag,)
                    )

        attributes = data.get('attributes', [])
        if isinstance(attributes, dict): # Handle historical format
            attributes = [{"attribute": k, "relevance": "N/A", "grounding": v} for k, v in attributes.items()]
        
        if attributes and isinstance(attributes, list):
            st.markdown("**AI Extracted Attributes:**")
            for item in attributes:
                attr = item.get('attribute', 'N/A')
                grounding = item.get('grounding', 'No grounding text.')
                relevance = item.get('relevance', 'N/A')
                st.markdown(f"- **{attr}**: {grounding} _(Relevance: {relevance})_")
    except (json.JSONDecodeError, TypeError):
        log.warning("Could not parse llm_tags JSON.")

@st.cache_data(ttl=3600) # Cache the expensive data fetch for an hour
def get_all_article_embeddings():
    """Fetches all articles with their embeddings into a DataFrame."""
    conn = database.get_db_connection()
    df = pd.read_sql_query("SELECT id, title, source, original_url, embedding FROM newsletters WHERE embedding IS NOT NULL", conn)
    conn.close()
    # Pre-process embeddings into a NumPy array for efficiency
    df['embedding_np'] = df['embedding'].apply(casting.to_numpy_array)
    df.dropna(subset=['embedding_np'], inplace=True)
    return df

def find_and_display_similar_articles(current_article_id: str):
    """Finds and displays similar articles using efficient vectorized search."""
    with st.spinner("Finding similar articles..."):
        all_articles_df = get_all_article_embeddings()
        
        query_article = all_articles_df[all_articles_df['id'] == current_article_id]
        if query_article.empty or query_article.iloc[0]['embedding_np'] is None:
            st.warning("This article has no embedding for similarity search.")
            return

        query_embedding = query_article.iloc[0]['embedding_np']
        search_space_df = all_articles_df[all_articles_df['id'] != current_article_id].copy()
        if search_space_df.empty:
            st.info("No other articles with embeddings found to compare against."); return

        embedding_matrix = np.vstack(search_space_df['embedding_np'].values)
        
        # Normalize vectors for cosine similarity. Assumes embeddings are not pre-normalized.
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        embedding_matrix_norm = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        
        similarities = np.dot(embedding_matrix_norm, query_embedding_norm)
        
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        st.markdown("**Top 5 Similar Articles:**")
        for i in top_indices:
            row = search_space_df.iloc[i]
            st.markdown(f"- **{row['title']}** (from: *{row['source']}*, Similarity: {similarities[i]:.2%})")
            btn_cols = st.columns(2)
            btn_cols[0].button("Open in App", key=f"app_{row['id']}", on_click=set_detail_view, args=(row['id'],))
            btn_cols[1].link_button("Open in Readwise", url=row['original_url'])

@st.cache_data(ttl=600)
def get_filter_options_with_counts():
    conn = database.get_db_connection()
    if not conn: return {}, {}, {}
    sources_df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM newsletters WHERE source IS NOT NULL GROUP BY source ORDER BY source", conn)
    categories_df = pd.read_sql_query("SELECT category, COUNT(*) as count FROM newsletters WHERE category IS NOT NULL GROUP BY category ORDER BY category", conn)
    tags_data = conn.execute("SELECT llm_tags FROM newsletters").fetchall()
    conn.close()
    
    tags_df = pd.DataFrame(tags_data, columns=['llm_tags'])
    all_tags_list = sorted(list(set(helpers.extract_tags_from_dataframe(tags_df))))
    source_options = {f"{row['source']} ({row['count']})": row['source'] for _, row in sources_df.iterrows()} # itertuples() is faster but this is fine
    category_options = {f"{row['category']} ({row['count']})": row['category'] for _, row in categories_df.iterrows()}
    return source_options, category_options, all_tags_list

# --- Main App Logic ---
st.title("🗄️ Newsletter Archive")

# --- STATE MANAGEMENT ---
if 'archive_filters' not in st.session_state:
    st.session_state.archive_filters = {"article_id": None, "tag": "", "keyword": "", "sources": [], "categories": [], "date_range": (datetime.now() - timedelta(days=30), datetime.now().date())}
if 'archive_expand_all' not in st.session_state: st.session_state.archive_expand_all = False
if 'archive_page' not in st.session_state: st.session_state.archive_page = 1
if 'widget_key_suffix' not in st.session_state: st.session_state.widget_key_suffix = 0

query_params = st.query_params
url_article_id = query_params.get("article_id", [None])[0]
url_tag = query_params.get("tag", [None])[0]
if url_article_id:
    log.info(f"URL param 'article_id' found: {url_article_id}. Setting detail view.")
    set_detail_view(url_article_id)
    st.query_params.clear(); st.rerun()
if url_tag:
    log.info(f"URL param 'tag' found: {url_tag}. Setting tag filter.")
    set_tag_filter(url_tag)
    st.query_params.clear(); st.rerun()

# --- VIEW LOGIC ---
if st.session_state.archive_filters["article_id"]:
    st.info(f"Showing detail for article ID: `{st.session_state.archive_filters['article_id']}`")
    st.button("Back to List View", on_click=clear_all_filters)
    query = "SELECT * FROM newsletters WHERE id = ?"; params = [st.session_state.archive_filters["article_id"]]
else:
    st.subheader("Filters & Search")
    source_options, category_options, all_tags_list = get_filter_options_with_counts()
    
    # --- NEW: Combined Keyword Search ---
    c1, c2 = st.columns(2)
    st.session_state.archive_filters["keyword"] = c1.text_input(
        "Keyword Search (Title, Tags, Attributes)", 
        value=st.session_state.archive_filters.get("keyword", "")
    )
    
    try: tag_index = all_tags_list.index(st.session_state.archive_filters["tag"]) + 1 if st.session_state.archive_filters["tag"] else 0
    except ValueError: tag_index = 0
    st.session_state.archive_filters["tag"] = c2.selectbox("Filter by a specific Tag", options=[""] + all_tags_list, index=tag_index)
    
    # --- REFACTOR: Group advanced filters in an expander for a cleaner UI ---
    with st.expander("Advanced Filters"):
        c3, c4, c5 = st.columns(3)
        source_labels = {v: k for k, v in source_options.items()}
        selected_source_labels = [source_labels.get(s) for s in st.session_state.archive_filters["sources"] if s in source_labels]
        st.session_state.archive_filters["sources"] = [source_options[label] for label in c3.multiselect("Filter by Source", options=source_options.keys(), default=selected_source_labels)]
        
        category_labels = {v: k for k, v in category_options.items()}
        selected_category_labels = [category_labels.get(c) for c in st.session_state.archive_filters["categories"] if c in category_labels]
        st.session_state.archive_filters["categories"] = [category_options[label] for label in c4.multiselect("Filter by Category", options=category_options.keys(), default=selected_category_labels)]
        
        st.session_state.archive_filters["date_range"] = c5.date_input("Filter by Date Range", value=st.session_state.archive_filters["date_range"])

    if st.button("Clear All Filters"):
        clear_all_filters(); st.rerun()
    
    # --- NEW: Updated Query Logic ---
    query, params = "SELECT * FROM newsletters WHERE 1=1", []
    filters = st.session_state.archive_filters
    
    if filters.get("keyword"):
        keyword_param = f"%{filters['keyword']}%"
        query += " AND (title LIKE ? OR llm_tags LIKE ?)"
        params.extend([keyword_param, keyword_param])

    if filters["tag"]: query += " AND llm_tags LIKE ?"; params.append(f'%"{filters["tag"]}"%')
    if filters["sources"]: query += f" AND source IN ({','.join(['?'] * len(filters['sources']))})"; params.extend(filters['sources'])
    if filters["categories"]: query += f" AND category IN ({','.join(['?'] * len(filters['categories']))})"; params.extend(filters['categories'])
    if len(filters["date_range"]) == 2:
        start_date, end_date = datetime.combine(filters["date_range"][0], datetime.min.time()), datetime.combine(filters["date_range"][1], datetime.max.time())
        query += " AND published_date BETWEEN ? AND ?"; params.extend([start_date, end_date])
    query += " ORDER BY published_date DESC"
    log.info(f"Executing list view query. Filters applied: { {k:v for k,v in filters.items() if v} }")

# --- DATA FETCHING & DISPLAY ---
data, total_items = database.fetch_paginated_data(query, tuple(params), st.session_state.archive_page, PAGINATION_LIMIT)
log.info(f"DB fetch returned {len(data)} rows for the current page.")
st.divider()

if not data:
    st.warning("No articles found matching your criteria.")
else:
    if not st.session_state.archive_filters["article_id"]:
        total_pages = max(1, (total_items + PAGINATION_LIMIT - 1) // PAGINATION_LIMIT)
        st.write(f"Showing page {st.session_state.archive_page} of {total_pages} ({total_items} total articles)")
        b1, b2, _, _ = st.columns(4)
        if b1.button("Expand All"): st.session_state.archive_expand_all = True
        if b2.button("Collapse All"): st.session_state.archive_expand_all = False

    for row in data:
        st.session_state.widget_key_suffix = row['id']
        with st.expander(f"**{row['title']}** (from: {row['source']})", expanded=st.session_state.archive_expand_all or bool(st.session_state.archive_filters["article_id"])):
            st.markdown(f"**Published:** {casting.to_datetime(row['published_date']).strftime('%Y-%m-%d')} | **[Open in Readwise Reader]({row['original_url']})**")
            st.markdown("---")
            if row['llm_summary']: st.markdown("**AI Generated Summary:**"); st.write(row['llm_summary'])
            if row['readwise_summary']: st.markdown("**Readwise Summary:**"); st.write(row['readwise_summary'])
            st.markdown("---"); render_metadata(row['llm_tags']); st.markdown("---")
            if st.button("Find Similar Articles", key=f"find_similar_{row['id']}"):
                find_and_display_similar_articles(row['id'])
    if not st.session_state.archive_filters["article_id"] and total_items > PAGINATION_LIMIT:
        st.divider()
        total_pages = max(1, (total_items + PAGINATION_LIMIT - 1) // PAGINATION_LIMIT)
        p1, p2, p3 = st.columns([1, 2, 1])
        if p1.button("⬅️ Previous", disabled=(st.session_state.archive_page <= 1)):
            st.session_state.archive_page -= 1; st.rerun()
        if p3.button("Next ➡️", disabled=(st.session_state.archive_page >= total_pages)):
            st.session_state.archive_page += 1; st.rerun()
        p2.write(f"Page {st.session_state.archive_page} of {total_pages}")
