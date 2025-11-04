# newsletter_synthesis_app/pages/5_LLM_Call_History.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import database
from config import PAGINATION_LIMIT

st.set_page_config(page_title="LLM Call History", layout="wide")

st.title("📈 LLM Call History & Analytics")

# --- Filters ---
conn = database.get_db_connection()
purposes = [row['purpose'] for row in conn.execute("SELECT DISTINCT purpose FROM llm_calls ORDER BY purpose").fetchall()]
models = [row['model_used'] for row in conn.execute("SELECT DISTINCT model_used FROM llm_calls ORDER BY model_used").fetchall()]
statuses = [row['status'] for row in conn.execute("SELECT DISTINCT status FROM llm_calls ORDER BY status").fetchall()]
conn.close()

col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_purposes = st.multiselect("Filter by Purpose", options=purposes)
with col2:
    selected_models = st.multiselect("Filter by Model", options=models)
with col3:
    selected_statuses = st.multiselect("Filter by Status", options=statuses)
with col4:
    time_filter = st.selectbox("Filter by Time", ["Last 24 Hours", "Last 7 Days", "All Time"], index=2)

# --- Data Fetching ---
query = "SELECT * FROM llm_calls WHERE 1=1"
params = []

if selected_purposes:
    query += f" AND purpose IN ({','.join(['?'] * len(selected_purposes))})"
    params.extend(selected_purposes)
if selected_models:
    query += f" AND model_used IN ({','.join(['?'] * len(selected_models))})"
    params.extend(selected_models)
if selected_statuses:
    query += f" AND status IN ({','.join(['?'] * len(selected_statuses))})"
    params.extend(selected_statuses)

if time_filter == "Last 24 Hours":
    query += " AND timestamp >= ?"
    params.append(datetime.now() - timedelta(hours=24))
elif time_filter == "Last 7 Days":
    query += " AND timestamp >= ?"
    params.append(datetime.now() - timedelta(days=7))

query += " ORDER BY timestamp DESC"

# --- Analytics ---
analytics_conn = database.get_db_connection()
analytics_df = pd.read_sql_query(query, analytics_conn, params=tuple(params))
analytics_conn.close()

total_cost = analytics_df['cost'].sum()
total_calls = len(analytics_df)
success_rate = (analytics_df['status'] == 'success').sum() / total_calls if total_calls > 0 else 0

st.subheader("Analytics Overview")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total Calls", f"{total_calls:,}")
metric2.metric("Total Cost (USD)", f"${total_cost:.4f}")
metric3.metric("Success Rate", f"{success_rate:.2%}")
if total_calls > 0:
    avg_duration = analytics_df[analytics_df['duration_ms'] > 0]['duration_ms'].mean()
    metric4.metric("Avg. Duration (ms)", f"{avg_duration:,.0f}")

# --- Pagination ---
if 'llm_page' not in st.session_state:
    st.session_state.llm_page = 1

total_items = len(analytics_df)
total_pages = (total_items + PAGINATION_LIMIT - 1) // PAGINATION_LIMIT
start_idx = (st.session_state.llm_page - 1) * PAGINATION_LIMIT
end_idx = start_idx + PAGINATION_LIMIT
paginated_df = analytics_df.iloc[start_idx:end_idx]

# --- Display Data ---
st.divider()
st.subheader("Call Details")
if paginated_df.empty:
    st.info("No LLM calls found matching your criteria.")
else:
    st.dataframe(paginated_df, width="stretch")

    # Pagination controls
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.llm_page <= 1)):
            st.session_state.llm_page -= 1
            st.rerun()
    with page_col3:
        if st.button("Next ➡️", disabled=(st.session_state.llm_page >= total_pages)):
            st.session_state.llm_page += 1
            st.rerun()
    with page_col2:
        st.write(f"Page {st.session_state.llm_page} of {total_pages}")