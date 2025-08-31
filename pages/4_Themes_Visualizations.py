# newsletter_synthesis_app/pages/4_Themes_Visualizations.py

import streamlit as st
import pandas as pd
import json
from pyvis.network import Network
import streamlit.components.v1 as components
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import database

st.set_page_config(page_title="Themes & Visualizations", layout="wide")

st.title("📊 Themes & Visualizations")
st.info("Explore the relationships between identified themes and their underlying tags.")

# --- Data Fetching ---
@st.cache_data(ttl=600)
def get_theme_data():
    conn = database.get_db_connection()
    df = pd.read_sql_query("SELECT name, frequency, last_seen, associated_canonical_tags FROM themes", conn)
    conn.close()
    return df

theme_df = get_theme_data()

if theme_df.empty:
    st.warning("No themes found. Please generate a summary first to identify themes.")
else:
    # --- Interactive Graph ---
    st.subheader("Theme-Tag Relationship Graph")

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=True, directed=True)
    
    # Add theme nodes
    for index, row in theme_df.iterrows():
        theme_name = row['name']
        net.add_node(theme_name, label=theme_name, title=f"Theme\nFrequency: {row['frequency']}", color="#FF4B4B", size=25)

        # Add tag nodes and edges
        try:
            tags = json.loads(row['associated_canonical_tags'])
            for tag in tags:
                if tag not in net.get_nodes():
                    net.add_node(tag, label=tag, title=f"Tag", color="#00C49A", size=15)
                net.add_edge(theme_name, tag)
        except (json.JSONDecodeError, TypeError):
            continue

    net.show_buttons(filter_=['physics'])
    
    try:
        net.save_graph("theme_graph.html")
        with open("theme_graph.html", "r", encoding="utf-8") as html_file:
            source_code = html_file.read()
        components.html(source_code, height=610)
    except Exception as e:
        st.error(f"Could not generate graph: {e}")

    # --- Theme Data Table ---
    st.divider()
    st.subheader("All Identified Themes")
    st.dataframe(theme_df.sort_values(by="frequency", ascending=False), use_container_width=True)