# newsletter_synthesis_app/pages/1_Synthesizer.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from collections import Counter
import litellm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import streamlit.components.v1 as components

from utils import database, processing_logic, prompts
from utils import llm_processor, helpers # Correctly import llm_processor and helpers
from config import (
    DEFAULT_SUMMARY_MODEL, DEFAULT_NAMING_MODEL, DEFAULT_REFINEMENT_MODEL, 
    AVAILABLE_MODELS, DEFAULT_EMBEDDING_MODEL
)

st.set_page_config(page_title="AI Synthesizer", layout="wide")

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=600)
def get_filter_options_with_counts():
    """Fetches unique values for filters and their counts for display."""
    conn = database.get_db_connection()
    if not conn: return {}, {}, {}
    sources_df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM newsletters WHERE source IS NOT NULL GROUP BY source ORDER BY source", conn)
    categories_df = pd.read_sql_query("SELECT category, COUNT(*) as count FROM newsletters WHERE category IS NOT NULL GROUP BY category ORDER BY category", conn)
    tags_data = conn.execute("SELECT llm_tags FROM newsletters").fetchall()
    conn.close()
    
    # Create a temporary DataFrame to reuse the helper
    tags_df = pd.DataFrame(tags_data, columns=['llm_tags'])
    all_tags_list = sorted(list(set(helpers.extract_tags_from_dataframe(tags_df))))

    # Use itertuples() for minor performance improvement over iterrows()
    source_options = {f"{row.source} ({row.count})": row.source for row in sources_df.itertuples()}
    category_options = {f"{row.category} ({row.count})": row.category for row in categories_df.itertuples()}
    
    return source_options, category_options, all_tags_list

def _render_summary_tabs(summary_row: dict):
    """Render Final vs Original summary tabs for a given summaries row."""
    final_text = summary_row.get('content') or ''
    draft_text = summary_row.get('draft_content') or ''
    tabs = st.tabs(["Final", "Original"])
    with tabs[0]:
        if final_text:
            # Copy button for Final
            _render_copy_button(final_text, label="Copy Final")
        if final_text:
            st.markdown(final_text, unsafe_allow_html=True)
        else:
            st.info("No final content available.")
    with tabs[1]:
        if draft_text:
            # Copy button for Original
            _render_copy_button(draft_text, label="Copy Original")
        if draft_text:
            st.markdown(draft_text, unsafe_allow_html=True)
        else:
            st.info("No original draft available for this synthesis.")

def _render_copy_button(text: str, label: str = "Copy"):
    if not text:
        return
    b64 = base64.b64encode(text.encode('utf-8')).decode()
    # Use a small HTML component to access Clipboard API
    components.html(
        f"""
        <div>
            <button style=\"margin-bottom:8px;\" onclick=\"navigator.clipboard.writeText(atob('{b64}'))\">{label}</button>
        </div>
        """,
        height=40,
    )

@st.cache_data(ttl=300)
def get_recent_summaries(limit: int = 5):
    conn = database.get_db_connection()
    if not conn:
        return []
    rows = conn.execute(
        """
        SELECT id, generated_date, newsletter_count, themes_json, content, draft_content, generation_context_json
        FROM summaries
        ORDER BY generated_date DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@st.cache_data(ttl=300) # Cache for 5 minutes
def estimate_synthesis_cost_time(article_count: int, models: dict) -> tuple[float, int]:
    """
    Estimates cost and time for a synthesis run using historical data from the llm_calls table.
    Falls back to heuristics if no historical data is available.
    """
    conn = database.get_db_connection()
    # Fetch historical averages for cost and duration for relevant purposes
    query = """
    SELECT purpose, AVG(cost) as avg_cost, AVG(duration_ms) as avg_duration_ms
    FROM llm_calls
    WHERE purpose IN ('synthesis_cluster_naming', 'synthesis_deep_dive', 'synthesis_final_critique')
      AND status = 'success' AND cost > 0 AND duration_ms > 0
    GROUP BY purpose
    """
    history = {row['purpose']: dict(row) for row in conn.execute(query).fetchall()}
    conn.close()

    estimated_themes = max(1, article_count // 5)

    # Get averages or fall back to heuristics
    avg_naming_cost = history.get('synthesis_cluster_naming', {}).get('avg_cost', 0.0001)
    avg_summary_cost = history.get('synthesis_deep_dive', {}).get('avg_cost', 0.0015)
    avg_critique_cost = history.get('synthesis_final_critique', {}).get('avg_cost', 0.0020)

    avg_naming_time_ms = history.get('synthesis_cluster_naming', {}).get('avg_duration_ms', 5000)
    avg_summary_time_ms = history.get('synthesis_deep_dive', {}).get('avg_duration_ms', 15000)
    avg_critique_time_ms = history.get('synthesis_final_critique', {}).get('avg_duration_ms', 10000)

    # Estimate total cost and time
    estimated_cost = (estimated_themes * (avg_naming_cost + avg_summary_cost)) + avg_critique_cost
    estimated_time_ms = (estimated_themes * (avg_naming_time_ms + avg_summary_time_ms)) + avg_critique_time_ms
    estimated_time_seconds = estimated_time_ms // 1000

    return estimated_cost, int(estimated_time_seconds)

def generate_word_cloud(df: pd.DataFrame):
    """Generates and displays a word cloud from article summaries."""
    text = " ".join(summary for summary in df['llm_summary'] if summary)
    if not text:
        st.info("No summary text available in the selected articles to generate a word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def suggest_related_tags(selected_tags: list[str], articles_df: pd.DataFrame) -> list[dict]:
    """
    Uses an LLM to suggest related tags based on the current filter context,
    using the already-fetched DataFrame of articles.
    """
    if not selected_tags:
        return []

    # Use the centralized helper to get tags from the already-filtered DataFrame
    tags_in_subset = set(helpers.extract_tags_from_dataframe(articles_df))
    # Remove already selected tags from the options
    options_for_llm = list(tags_in_subset - set(selected_tags))
    if not options_for_llm:
        return []

    prompt_template = prompts.get_prompt_template("synthesis_suggest_tags")
    prompt = prompt_template.format(selected_tags=json.dumps(selected_tags), options=json.dumps(options_for_llm))
    
    response = llm_processor.call_llm(
        DEFAULT_REFINEMENT_MODEL, [{"role": "user", "content": prompt}],
        'suggest_related_tags', json_mode=True
    )
    
    return response if isinstance(response, list) else []

def build_query(filters: dict) -> tuple[str, list]:
    """Helper function to build the SQL query from the filter state."""
    query, params = "SELECT * FROM newsletters WHERE 1=1", []
    if filters["keyword"]:
        keyword_param = f"%{filters['keyword']}%"
        query += " AND (title LIKE ? OR llm_tags LIKE ?)"
        params.extend([keyword_param, keyword_param])

    if filters["tags"]: # Changed from "tag" to "tags" for multiselect
        tag_clauses = " AND ".join(["llm_tags LIKE ?"] * len(filters["tags"]))
        query += f" AND ({tag_clauses})"; params.extend([f'%"{tag}"%' for tag in filters["tags"]])
    if filters["sources"]: query += f" AND source IN ({','.join(['?'] * len(filters['sources']))})"; params.extend(filters['sources'])
    if filters["categories"]: query += f" AND category IN ({','.join(['?'] * len(filters['categories']))})"; params.extend(filters['categories'])
    if len(filters["date_range"]) == 2:
        start_date, end_date = datetime.combine(filters["date_range"][0], datetime.min.time()), datetime.combine(filters["date_range"][1], datetime.max.time())
        query += " AND published_date BETWEEN ? AND ?"; params.extend([start_date, end_date])
    query += " ORDER BY published_date DESC"
    return query, params

# --- MAIN APP ---
st.title("🔬 AI Synthesizer")
st.info("Select a cohort of articles, get smart suggestions, preview an analysis, and generate a custom insights pack.")
prompts.initialize_prompts()
try:
    st.page_link("pages/6_Syntheses_Archive.py", label="Open Syntheses Archive", icon="📚")
except Exception:
    if st.button("Open Syntheses Archive"):
        try:
            st.switch_page("pages/6_Syntheses_Archive.py")
        except Exception:
            st.info("Open 'Syntheses Archive' from the left sidebar.")

# --- SIDEBAR: Filters and Preset Management ---
st.sidebar.header("Filter Articles for Synthesis")

presets = database.get_all_presets()
preset_options = {p['name']: p['id'] for p in presets}
selected_preset = st.sidebar.selectbox("Load Filter Preset", options=[""] + list(preset_options.keys()))

if selected_preset:
    preset_filters = database.get_preset_filters(preset_options[selected_preset])
    if preset_filters:
        # Load preset data into session state
        st.session_state.synth_filters = preset_filters
        st.success(f"Loaded preset '{selected_preset}'")
        # Use st.rerun() to apply the loaded filters immediately
        st.rerun()

source_options, category_options, all_tags_list = get_filter_options_with_counts()

if 'synth_filters' not in st.session_state:
    st.session_state.synth_filters = {"keyword": "", "tags": [], "sources": [], "categories": [], "date_range": (datetime.now() - timedelta(days=30), datetime.now().date())}
if 'tag_suggestions' not in st.session_state:
    st.session_state.tag_suggestions = []

# Filter widgets
keyword_search = st.sidebar.text_input("Keyword Search", value=st.session_state.synth_filters['keyword'])
date_range = st.sidebar.date_input("Date Range", value=st.session_state.synth_filters['date_range'])
# --- TWEAK: Changed from selectbox to multiselect ---
selected_tags = st.sidebar.multiselect("Filter by Tags", options=all_tags_list, default=st.session_state.synth_filters['tags'])


if st.session_state.tag_suggestions:
    with st.sidebar.expander("✨ Smart Suggestions", expanded=True):
        for suggestion in st.session_state.tag_suggestions:
            tag = suggestion.get("tag")
            reason = suggestion.get("reason")
            if tag and tag not in selected_tags:
                if st.button(f"+ Add '{tag}'", key=f"suggestion_{tag}", help=reason, width='stretch'):
                    st.session_state.synth_filters['tags'].append(tag)
                    st.session_state.tag_suggestions = [] # Clear suggestions after use
                    st.rerun()

selected_source_labels = st.sidebar.multiselect("Filter by Source", options=source_options.keys(), default=[k for k,v in source_options.items() if v in st.session_state.synth_filters['sources']])
selected_category_labels = st.sidebar.multiselect("Filter by Category", options=category_options.keys(), default=[k for k,v in category_options.items() if v in st.session_state.synth_filters['categories']])

# Update session state from widgets
st.session_state.synth_filters.update({
    "keyword": keyword_search, "date_range": date_range, "tags": selected_tags,
    "sources": [source_options[label] for label in selected_source_labels],
    "categories": [category_options[label] for label in selected_category_labels]
})

# (Preset management remains the same)
with st.sidebar.expander("Save or Delete Presets"):
    preset_name = st.text_input("Save current filters as new preset:")
    if st.button("Save Preset"):
        if preset_name:
            database.save_preset(preset_name, st.session_state.synth_filters)
            st.success(f"Preset '{preset_name}' saved!")
            st.rerun()
        else:
            st.warning("Please enter a name for the preset.")
    
    if presets:
        preset_to_delete = st.selectbox("Delete a preset:", options=[""] + list(preset_options.keys()))
        if st.button("Delete Preset"):
            if preset_to_delete:
                database.delete_preset(preset_options[preset_to_delete])
                st.success(f"Preset '{preset_to_delete}' deleted!")
                st.rerun()

# --- Build Query from Session State ---
query, params = build_query(st.session_state.synth_filters)
conn = database.get_db_connection()
filtered_articles_df = pd.read_sql_query(query, conn, params=tuple(params))
conn.close()

# --- Smart Suggestions UI & Logic (moved after data fetching) ---
if selected_tags:
    if st.sidebar.button("💡 Suggest Related Tags"):
        with st.spinner("Asking AI for suggestions..."):
            # Pass the already-filtered DataFrame to avoid a second DB call
            st.session_state.tag_suggestions = suggest_related_tags(
                selected_tags, filtered_articles_df
            )
            st.rerun() # Rerun to display suggestions immediately

# --- MAIN DISPLAY (The rest of the page is mostly the same) ---
# Tabs already created
recent = []  # moved to archive page
if recent:
    labels = [f"{r['generated_date']} • {r['newsletter_count']} articles • ID {r['id']}" for r in recent]
    idx = st.selectbox("View a recent synthesis:", options=list(range(len(labels))), format_func=lambda i: labels[i])
    selected = recent[idx]
    # Show validation warning if present
    try:
        ctx = json.loads(selected.get('generation_context_json') or '{}')
        if ctx.get('citation_issues'):
            st.warning(f"Citations check: {ctx.get('citation_issues')} of {ctx.get('bullets_count', 0)} bullets missing inline citations.")
    except json.JSONDecodeError:
        pass
    _render_summary_tabs(selected)
    st.divider()
else:
    pass

tabs = st.tabs(["Generate", "Final", "Original"])

with tabs[0]:
    # Inline Filters (moved from sidebar)
    st.subheader("Filters")
    presets = database.get_all_presets()
    preset_options = {p['name']: p['id'] for p in presets}
    # Flash message support to avoid rerun loop
    if st.session_state.get('synth_preset_flash'):
        st.success(st.session_state['synth_preset_flash'])
        del st.session_state['synth_preset_flash']

    selected_preset = st.selectbox(
        "Load Filter Preset",
        options=[""] + list(preset_options.keys()),
        key="synth_preset_select"
    )
    if selected_preset:
        preset_filters = database.get_preset_filters(preset_options[selected_preset])
        if preset_filters:
            st.session_state.synth_filters = preset_filters
            # Reset the selectbox and show a one-time flash after rerun
            st.session_state['synth_preset_flash'] = f"Loaded preset '{selected_preset}'"
            st.session_state['synth_preset_select'] = ""
            st.rerun()

    # Common option lists
    source_options, category_options, all_tags_list = get_filter_options_with_counts()
    if 'synth_filters' not in st.session_state:
        st.session_state.synth_filters = {"keyword": "", "tags": [], "sources": [], "categories": [], "date_range": (datetime.now() - timedelta(days=30), datetime.now().date())}
    if 'tag_suggestions' not in st.session_state:
        st.session_state.tag_suggestions = []

    c1, c2 = st.columns([2, 1])
    keyword_search = c1.text_input("Keyword Search", value=st.session_state.synth_filters['keyword'])
    date_range = c2.date_input("Date Range", value=st.session_state.synth_filters['date_range'])
    selected_tags = st.multiselect("Filter by Tags", options=all_tags_list, default=st.session_state.synth_filters['tags'])

    with st.expander("Advanced Filters"):
        c3, c4 = st.columns(2)
        selected_source_labels = c3.multiselect("Filter by Source", options=source_options.keys(), default=[k for k,v in source_options.items() if v in st.session_state.synth_filters['sources']])
        selected_category_labels = c4.multiselect("Filter by Category", options=category_options.keys(), default=[k for k,v in category_options.items() if v in st.session_state.synth_filters['categories']])

    st.session_state.synth_filters.update({
        "keyword": keyword_search, "date_range": date_range, "tags": selected_tags,
        "sources": [source_options[label] for label in selected_source_labels],
        "categories": [category_options[label] for label in selected_category_labels]
    })

    with st.expander("Save or Delete Presets"):
        preset_name = st.text_input("Save current filters as new preset:")
        if st.button("Save Preset"):
            if preset_name:
                database.save_preset(preset_name, st.session_state.synth_filters)
                st.success(f"Preset '{preset_name}' saved!")
                st.rerun()
            else:
                st.warning("Please enter a name for the preset.")
        if presets:
            preset_to_delete = st.selectbox("Delete a preset:", options=[""] + list(preset_options.keys()))
            if st.button("Delete Preset"):
                if preset_to_delete:
                    database.delete_preset(preset_options[preset_to_delete])
                    st.success(f"Preset '{preset_to_delete}' deleted!")
                    st.rerun()

    # Build query for current filters
    query, params = build_query(st.session_state.synth_filters)
    conn = database.get_db_connection()
    filtered_articles_df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()

    # Smart tag suggestions based on current selection
    if selected_tags:
        if st.button("Suggest Related Tags"):
            with st.spinner("Asking AI for suggestions..."):
                st.session_state.tag_suggestions = suggest_related_tags(selected_tags, filtered_articles_df)
                st.rerun()
    if st.session_state.tag_suggestions:
        with st.expander("Smart Suggestions", expanded=True):
            for suggestion in st.session_state.tag_suggestions:
                tag = suggestion.get("tag")
                reason = suggestion.get("reason")
                if tag and tag not in selected_tags:
                    if st.button(f"+ Add '{tag}'", key=f"suggestion_{tag}", help=reason):
                        st.session_state.synth_filters['tags'].append(tag)
                        st.session_state.tag_suggestions = []
                        st.rerun()
    st.subheader(f"Found {len(filtered_articles_df)} articles for synthesis")
    if not filtered_articles_df.empty:
        with st.expander("Preview Selected Articles"):
            st.dataframe(filtered_articles_df[['title', 'source', 'published_date']], width='stretch', hide_index=True)

        # --- NEW: Pre-Synthesis Analysis & Estimation ---
        st.divider()
        st.subheader("Pre-Synthesis Analysis & Estimation")
        
        # Get models for cost estimation
        models_for_estimation = {
            'naming': st.session_state.get('synthesis_naming_model', DEFAULT_NAMING_MODEL),
            'summary': st.session_state.get('synthesis_summary_model', DEFAULT_SUMMARY_MODEL),
            'critique': st.session_state.get('synthesis_critique_model', DEFAULT_SUMMARY_MODEL)
        }
        estimated_cost, estimated_time = estimate_synthesis_cost_time(len(filtered_articles_df), models_for_estimation)
        
        metric1, metric2 = st.columns(2)
        metric1.metric("Estimated Cost (USD)", f"~${estimated_cost:.4f}")
        metric2.metric("Estimated Time", f"~{estimated_time // 60} min {estimated_time % 60} sec")

        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("**Term Cloud from Summaries**")
            generate_word_cloud(filtered_articles_df)
        with chart2:
            st.markdown("**Top Tags in Selection**")
            tags_in_selection = helpers.extract_tags_from_dataframe(filtered_articles_df)
            if tags_in_selection:
                tag_counts = Counter(tags_in_selection).most_common(10)
                tags_dist_df = pd.DataFrame(tag_counts, columns=['Tag', 'Count']).set_index('Tag')
                st.bar_chart(tags_dist_df)

        st.divider()
        
        # --- Generation Controls ---
        st.subheader("Generate Insights Pack")
        st.write("Click the button to perform theme clustering, deep-dive summarization, and final critique on the selected articles.")

        with st.expander("Configure Synthesis Models"):
            st.selectbox("Theme Naming Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_NAMING_MODEL), key="synthesis_naming_model")
            st.selectbox("Deep Dive Summary Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_SUMMARY_MODEL), key="synthesis_summary_model")
            st.selectbox("Final Critique Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_SUMMARY_MODEL), key="synthesis_critique_model")

        MAX_ARTICLES_FOR_SYNTHESIS = 200
        is_disabled = len(filtered_articles_df) > MAX_ARTICLES_FOR_SYNTHESIS or len(filtered_articles_df) == 0

        if st.button("Generate Insights Pack", type="primary", disabled=is_disabled):
            article_ids = filtered_articles_df['id'].tolist()
            
            models = {
                'embedding': st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL),
                'naming': st.session_state.synthesis_naming_model,
                'summary': st.session_state.synthesis_summary_model,
                'critique': st.session_state.synthesis_critique_model
            }
            
            status_element = st.empty()
            def status_update(message):
                status_element.info(message)

            with st.spinner("Synthesizing insights... This is a synchronous process and may take several minutes."):
                summary_id = processing_logic.synthesize_insights(article_ids, models, status_update)

            status_element.empty()

            if summary_id:
                st.success("Successfully generated new insights pack! Open the 'Final' tab to view it.")
                conn = database.get_db_connection()
                row = conn.execute(
                    "SELECT id, generated_date, newsletter_count, themes_json, content, draft_content, generation_context_json FROM summaries WHERE id = ?",
                    (summary_id,)
                ).fetchone()
                conn.close()
                if row:
                    st.session_state.current_summary_row = dict(row)
                else:
                    st.error("Could not load the saved summary record.")
            else:
                st.error("Failed to generate insights pack. Check the logs for details.")
        
        if len(filtered_articles_df) > MAX_ARTICLES_FOR_SYNTHESIS:
            st.warning(f"Synthesis is disabled for selections over {MAX_ARTICLES_FOR_SYNTHESIS} articles. Please refine your filters.")

    else:
        st.info("Adjust the filters above to select a set of articles for synthesis.")

with tabs[1]:
    st.subheader("Current Synthesis — Final")
    current = st.session_state.get('current_summary_row')
    if not current:
        st.info("No synthesis generated in this session yet.")
    else:
        try:
            ctx = json.loads(current.get('generation_context_json') or '{}')
            if ctx.get('citation_issues'):
                st.warning(f"Citations check: {ctx.get('citation_issues')} of {ctx.get('bullets_count', 0)} bullets missing inline citations.")
        except json.JSONDecodeError:
            pass
        final_text = current.get('content') or ''
        if final_text:
            _render_copy_button(final_text, label="Copy Final")
            st.markdown(final_text, unsafe_allow_html=True)
        else:
            st.info("No final content available.")

with tabs[2]:
    st.subheader("Current Synthesis — Original")
    current = st.session_state.get('current_summary_row')
    draft_text = (current or {}).get('draft_content') or ''
    if draft_text:
        _render_copy_button(draft_text, label="Copy Original")
        st.markdown(draft_text, unsafe_allow_html=True)
    else:
        st.info("No original draft available for this synthesis.")
