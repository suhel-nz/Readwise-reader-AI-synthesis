# newsletter_synthesis_app/config.py

import os

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "newsletter_app.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HTML_CACHE_DIR = os.path.join(BASE_DIR, "html_cache")

# --- API Rate Limit Configuration ---
READWISE_RPM = 20  # Readwise has a limit of 20 requests per minute

# Define limits for various LLM providers.
# These are examples and should be verified against the provider's official documentation.
# RPD = Requests Per Day, RPM = Requests Per Minute, TPM = Tokens Per Minute
LLM_PROVIDER_LIMITS = {
    "gemini/gemini-2.5-flash-lite": {"rpd": 5000, "rpm": 1000, "tpm": 4000000},
    "gemini/gemini-2.5-flash": {"rpd": 500, "rpm": 500, "tpm": 1000000},    
    "gemini/gemini-2.5-pro": {"rpd": 500, "rpm": 50, "tpm": 2000000},
    "gemini/text-embedding-004": {"rpd": 3000, "rpm": 1000, "tpm": 1000000},
    "gpt-4o": {"rpd": 20, "rpm": 5, "tpm": 100000},
    "gpt-5": {"rpd": 20, "rpm": 5, "tpm": 100000},    
    "claude-sonnet-4-20250514": {"rpd": 20, "rpm": 5, "tpm": 40000}, # Example, check Anthropic docs
    "claude-opus-4-20250514": {"rpd": 10, "rpm": 5, "tpm": 20000},  # Example
    "deepseek/deepseek-chat": {"rpd": 50, "rpm": 10, "tpm": 100000},
    # Default values for any model not explicitly listed
    "default": {"rpd": 50, "rpm": 5, "tpm": 40000}
}


# --- LLM Model Configuration ---
DEFAULT_TAGGING_MODEL = "gemini/gemini-2.5-flash-lite"
DEFAULT_REFINEMENT_MODEL = "gemini/gemini-2.5-flash-lite"
DEFAULT_NAMING_MODEL = "gemini/gemini-2.5-flash-lite"
DEFAULT_SUMMARY_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini/text-embedding-004" # gemini/text-embedding-004

AVAILABLE_MODELS = list(LLM_PROVIDER_LIMITS.keys())
if "default" in AVAILABLE_MODELS: AVAILABLE_MODELS.remove("default")

AVAILABLE_EMBEDDING_MODELS = [ "gemini/text-embedding-004", "text-embedding-3-small", "text-embedding-3-large", ]

# --- Readwise API Configuration ---
READWISE_API_BASE_URL = "https://readwise.io/api/v3/list/"
READWISE_CATEGORIES = ["article", "email", "rss", "feed"]

# --- K-Means Clustering Configuration ---
K_CLUSTERS = 'auto'
K_CLUSTERS_MAX = 10

# --- UI Configuration ---
PAGINATION_LIMIT = 20