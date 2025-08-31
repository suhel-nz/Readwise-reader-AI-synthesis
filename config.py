# newsletter_synthesis_app/config.py

import os

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "newsletter_app.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HTML_CACHE_DIR = os.path.join(BASE_DIR, "html_cache")

# --- LLM Model Configuration ---
# Default models for each step. These can be overridden in the Streamlit UI.
# Ensure you have the corresponding API keys in .streamlit/secrets.toml for the models you use.
DEFAULT_TAGGING_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_REFINEMENT_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_NAMING_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "gemini/gemini-2.5-pro" # A more powerful model for the final summary
DEFAULT_EMBEDDING_MODEL = "text-embedding-004" # Google's embedding model

# List of available models for the user to select from in the UI
AVAILABLE_MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "gpt-4o",
    "gpt-5",    
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "deepseek/deepseek-chat",
]

AVAILABLE_EMBEDDING_MODELS = [
    "text-embedding-004", # Google
    "text-embedding-3-small", # OpenAI
    "text-embedding-3-large", # OpenAI
]

# --- Readwise API Configuration ---
READWISE_API_BASE_URL = "https://readwise.io/api/v3/list/"
# Categories of content to fetch from Readwise
READWISE_CATEGORIES = ["article", "email", "rss", "feed"]

# --- K-Means Clustering Configuration ---
# Number of themes to identify. 'auto' will try to find the optimal number.
# You can set it to a fixed integer like 5.
K_CLUSTERS = 'auto' # or an integer e.g., 5
K_CLUSTERS_MAX = 10 # Max clusters to test if K_CLUSTERS is 'auto'

# --- UI Configuration ---
PAGINATION_LIMIT = 20 # Number of items per page in tables