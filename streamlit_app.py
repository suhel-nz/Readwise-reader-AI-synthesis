# newsletter_synthesis_app/streamlit_app.py

import streamlit as st
from datetime import datetime, timedelta
import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from config import (
    AVAILABLE_MODELS, AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_TAGGING_MODEL, DEFAULT_REFINEMENT_MODEL, DEFAULT_NAMING_MODEL,
    DEFAULT_SUMMARY_MODEL, DEFAULT_EMBEDDING_MODEL, HTML_CACHE_DIR, LOG_DIR
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

# --- UI Components ---
def render_main_dashboard():
    """Renders the main dashboard UI."""
    st.title("🧠 Newsletter Synthesis AI Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generate New Summary")
        
        # Date range selection
        last_date = database.get_latest_article_date()
        default_start_date = (last_date + timedelta(microseconds=1)) if last_date else datetime.now() - timedelta(days=30)
        
        start_date = st.date_input("Fetch Articles From:", default_start_date.date())
        end_date = st.date_input("Fetch Articles Up To:", datetime.now().date())
        
        # Convert to datetime objects for the backend
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        if st.button("Generate Summary", disabled=st.session_state.processing, type="primary"):
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
            summary = conn.execute("SELECT * FROM summaries WHERE id = ?", (latest_summary_id,)).fetchone()
            conn.close()
            if summary:
                st.markdown(f"**Summary ID:** {summary['id']} | **Generated on:** {summary['generated_date'].strftime('%Y-%m-%d %H:%M')}")
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