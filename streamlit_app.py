# newsletter_synthesis_app/streamlit_app.py

import streamlit as st
from datetime import datetime, timedelta
import os
import sys
import math

# Add project root to the Python path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from config import (
    AVAILABLE_MODELS, AVAILABLE_EMBEDDING_MODELS, LLM_PROVIDER_LIMITS,
    DEFAULT_TAGGING_MODEL, DEFAULT_REFINEMENT_MODEL, DEFAULT_NAMING_MODEL,
    DEFAULT_SUMMARY_MODEL, DEFAULT_EMBEDDING_MODEL, HTML_CACHE_DIR, LOG_DIR,
    K_CLUSTERS_MAX
)
from utils import database, logger, summary_generator

st.set_page_config(
    page_title="Newsletter Synthesis AI",
    page_icon="🧠",
    layout="wide",
)

# --- Initialization ---
def initialize_app():
    """Initializes directories and session state variables."""
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    database.create_tables()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Welcome! Ready to generate a summary."
    if 'progress_value' not in st.session_state:
        st.session_state.progress_value = 0.0

initialize_app()

# --- Callbacks ---
def status_callback(message):
    """Updates the status message in the session state."""
    st.session_state.status_message = message

def progress_callback(value):
    """Updates the progress bar value in the session state."""
    st.session_state.progress_value = value

def handle_summary_generation(start_date, end_date, models):
    """Handles the logic for starting and running the summary generation process."""
    st.session_state.processing = True
    run_timestamp = datetime.now()
    logger.setup_logger(run_timestamp)
    
    st.session_state.status_message = "Starting summary generation..."
    st.session_state.progress_value = 0.0
    
    summary_id = summary_generator.run_summary_generation(
        start_date, end_date, models, status_callback, progress_callback
    )
    
    if summary_id:
        st.success(f"Summary generation complete! View the new summary (ID: {summary_id}).")
        st.session_state.latest_summary_id = summary_id
    else:
        st.error("Summary generation failed. Check logs for details.")
    
    st.session_state.processing = False
    st.session_state.status_message = "Process finished. Ready for next task."
    st.rerun()

def estimate_and_check_limits(num_articles, models):
    """
    Estimates the number of API calls and checks against daily limits.
    Returns a list of warning messages if limits are likely to be exceeded.
    """
    warnings = []
    if num_articles == 0:
        return ["No new articles found in the selected date range."]

    # Estimate number of clusters/themes (use a max estimate)
    estimated_themes = min(K_CLUSTERS_MAX, num_articles // 3 + 1)

    # Estimate calls per model
    calls_per_model = {model: 0 for model in AVAILABLE_MODELS}
    calls_per_model[models['tagging']] += num_articles
    calls_per_model[models['refinement']] += 1
    calls_per_model[models['naming']] += estimated_themes
    calls_per_model[models['summary']] += estimated_themes
    # Embeddings are often a different class of model, but we'll track it similarly
    if models['embedding'] not in calls_per_model: calls_per_model[models['embedding']] = 0
    calls_per_model[models['embedding']] += num_articles + 1 # N articles + 1 summary

    # Check limits for each model
    for model, estimated_calls in calls_per_model.items():
        if estimated_calls == 0:
            continue
        
        limits = LLM_PROVIDER_LIMITS.get(model, LLM_PROVIDER_LIMITS['default'])
        rpd_limit = limits.get('rpd')
        
        if rpd_limit is not None:
            current_calls, _ = database.get_daily_usage(model)
            if current_calls + estimated_calls > rpd_limit:
                warning_msg = (
                    f"**{model}:** Daily request limit may be exceeded. "
                    f"Limit: {rpd_limit}, Current Usage: {current_calls}, "
                    f"Estimated for this run: {estimated_calls}."
                )
                warnings.append(warning_msg)
    
    return warnings

# --- UI Components ---
def render_main_dashboard():
    """Renders the main dashboard UI."""
    st.title("🧠 Newsletter Synthesis AI Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generate New Summary")
        
        # Date range selection
        last_date = database.get_latest_article_date()
        default_start_datetime = (last_date + timedelta(microseconds=1)) if last_date else datetime.now() - timedelta(days=1)
        
        # Create two columns for date and time inputs
        date_col1, time_col1 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", default_start_datetime.date(), key="start_date")
        with time_col1:
            start_time = st.time_input("Start Time", key="start_time")

        date_col2, time_col2 = st.columns(2)
        with date_col2:
            end_date = st.date_input("End Date", datetime.now().date(), key="end_date")
        with time_col2:
            end_time = st.time_input("End Time", key="end_time")

        # Combine date and time to create datetime objects
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)

        st.info(f"Processing range: `{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}` to `{end_datetime.strftime('%Y-%m-%d %H:%M:%S')}`")

        # --- NEW: Proactive check display ---
        # A bit of a simplification: we'd ideally get the article count without fetching full content,
        # but for this POC, a warning based on days is a good heuristic.
        time_delta_hours = (end_datetime - start_datetime).total_seconds() / 3600
        # A rough estimate: 2 articles per hour
        estimated_articles = max(1, math.ceil(time_delta_hours * 1.5))
        
        selected_models_for_check = {
            'tagging': st.session_state.get('tagging_model', DEFAULT_TAGGING_MODEL),
            'refinement': st.session_state.get('refinement_model', DEFAULT_REFINEMENT_MODEL),
            'naming': st.session_state.get('naming_model', DEFAULT_NAMING_MODEL),
            'summary': st.session_state.get('summary_model', DEFAULT_SUMMARY_MODEL),
            'embedding': st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL),
        }
        
        limit_warnings = estimate_and_check_limits(estimated_articles, selected_models_for_check)
        if limit_warnings:
            st.warning("Potential API Limit Issues Detected (based on estimate):")
            for warning in limit_warnings:
                st.markdown(f"- {warning}")
        
        # Disable button if processing or serious warnings exist
        is_disabled = st.session_state.processing or bool(limit_warnings)

        if st.button("Generate Summary", disabled=is_disabled, type="primary"):
            with st.spinner("Initializing..."):
                selected_models = {
                    'tagging': st.session_state.get('tagging_model', DEFAULT_TAGGING_MODEL),
                    'refinement': st.session_state.get('refinement_model', DEFAULT_REFINEMENT_MODEL),
                    'naming': st.session_state.get('naming_model', DEFAULT_NAMING_MODEL),
                    'summary': st.session_state.get('summary_model', DEFAULT_SUMMARY_MODEL),
                    'embedding': st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL),
                }
                handle_summary_generation(start_datetime, end_datetime, selected_models)

    with col2:
        st.subheader("Processing Status")
        st.info(st.session_state.status_message)
        st.progress(st.session_state.progress_value)
        
        if st.session_state.processing:
            st.warning("Processing is in progress. Please do not close this tab.")

    st.divider()

    st.subheader("Latest Generated Summary")
    latest_summary_id = st.session_state.get('latest_summary_id')
    if not latest_summary_id:
        # Fetch the most recent summary from DB on first load
        conn = database.get_db_connection()
        if conn:
            latest = conn.execute("SELECT id FROM summaries ORDER BY generated_date DESC LIMIT 1").fetchone()
            if latest:
                latest_summary_id = latest['id']
            conn.close()
            
    if latest_summary_id:
        conn = database.get_db_connection()
        if conn:
            with conn:
                summary = conn.execute("SELECT * FROM summaries WHERE id = ?", (latest_summary_id,)).fetchone()
            
            if summary:
                # --- FIX: Parse string date before formatting ---
                generated_date_val = summary['generated_date']
                if isinstance(generated_date_val, str):
                    # SQLite may return ISO format strings
                    generated_dt = datetime.fromisoformat(generated_date_val)
                else:
                    # It's already a datetime object
                    generated_dt = generated_date_val

                st.markdown(f"**Summary ID:** {summary['id']} | **Generated on:** {generated_dt.strftime('%Y-%m-%d %H:%M')}")
                st.markdown(summary['content'], unsafe_allow_html=True)
            else:
                st.info("No summary available yet. Generate one to get started!")
    else:
        st.info("No summary available yet. Generate one to get started!")

def render_sidebar():
    """Renders the sidebar for LLM configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        with st.expander("LLM Model Configuration", expanded=True):
            st.selectbox("Tagging Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_TAGGING_MODEL), key="tagging_model")
            st.selectbox("Tag Refinement Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_REFINEMENT_MODEL), key="refinement_model")
            st.selectbox("Cluster Naming Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_NAMING_MODEL), key="naming_model")
            st.selectbox("Themed Summary Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_SUMMARY_MODEL), key="summary_model")
            st.selectbox("Embedding Model", AVAILABLE_EMBEDDING_MODELS, index=AVAILABLE_EMBEDDING_MODELS.index(DEFAULT_EMBEDDING_MODEL), key="embedding_model")
            st.caption("Select the LLM models for each stage of the pipeline.")


# --- Main App Execution ---
render_sidebar()
render_main_dashboard()