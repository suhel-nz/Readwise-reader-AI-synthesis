# newsletter_synthesis_app/streamlit_app.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
from collections import Counter

from utils import database, prompts

st.set_page_config(
    page_title="Newsletter AI Dashboard",
    page_icon="🧠",
    layout="wide",
)

# Initialize prompts table on first run
prompts.initialize_prompts()
prompts.ensure_prompt_updates_v1()

st.title("🧠 Newsletter AI Dashboard")
st.info("Welcome to your personal knowledge AI system. Use the sidebar to navigate to the Synthesizer, Newsletter Archive, or Prompt Editor.")

# --- Data Fetching for Metrics ---
@st.cache_data(ttl=600) # Cache for 10 minutes
def get_dashboard_data():
    conn = database.get_db_connection()
    if not conn:
        return pd.DataFrame(), pd.DataFrame()
    
    # Processed articles data
    articles_df = pd.read_sql_query("SELECT processed_date, llm_tags FROM newsletters", conn)
    if not articles_df.empty:
        articles_df['processed_date'] = pd.to_datetime(articles_df['processed_date'], format='ISO8601', utc=True)

    # Summaries data
    summaries_df = pd.read_sql_query("SELECT generated_date FROM summaries", conn)
    if not summaries_df.empty:
        summaries_df['generated_date'] = pd.to_datetime(summaries_df['generated_date'], format='ISO8601', utc=True)
        
    conn.close()
    return articles_df, summaries_df

articles_df, summaries_df = get_dashboard_data()

# --- Display Metrics ---
st.subheader("Activity Overview (Last 7 Days)")

seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

if not articles_df.empty:
    recent_articles = articles_df[articles_df['processed_date'] >= seven_days_ago]
    
    all_tags = []
    for tags_json in recent_articles['llm_tags']:
        try:
            tags_data = json.loads(tags_json)
            if 'tags' in tags_data and isinstance(tags_data['tags'], list):
                all_tags.extend(tags_data['tags'])
        except (json.JSONDecodeError, TypeError):
            continue
    
    total_articles_processed_7d = len(recent_articles)
    unique_tags_discovered_7d = len(set(all_tags))
else:
    total_articles_processed_7d = 0
    unique_tags_discovered_7d = 0

if not summaries_df.empty:
    insights_packs_generated_7d = len(summaries_df[summaries_df['generated_date'] >= seven_days_ago])
else:
    insights_packs_generated_7d = 0

col1, col2, col3 = st.columns(3)
col1.metric("Articles Processed", f"{total_articles_processed_7d}")
col2.metric("Unique Tags Discovered", f"{unique_tags_discovered_7d}")
col3.metric("Insights Packs Generated", f"{insights_packs_generated_7d}")

st.divider()

# --- Visualizations ---
col_chart, col_tags = st.columns(2)

with col_chart:
    st.subheader("Articles Processed Per Day")
    if not articles_df.empty:
        daily_counts = articles_df.set_index('processed_date').resample('D').size().rename("count")
        st.bar_chart(daily_counts)
    else:
        st.info("No articles processed yet.")

with col_tags:
    st.subheader("Most Frequent Tags (Last 7 Days)")
    if 'all_tags' in locals() and all_tags:
        tag_counts = Counter(all_tags).most_common(15)
        tags_df = pd.DataFrame(tag_counts, columns=['Tag', 'Frequency'])
        st.dataframe(tags_df, width="stretch", hide_index=True)        
    else:
        st.info("No tags discovered in the last 7 days.")
