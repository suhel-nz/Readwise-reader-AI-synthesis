# Newsletter Synthesis AI

## 1. The Vision: My Personal Knowledge AI

I built this application to solve a problem I face all the time: an overwhelming firehose of information from dozens of newsletters, YouTube videos, articles, PDFs etc. Key insights were getting lost in the noise, and I lacked an efficient way to connect ideas across different sources over time.

This project is my attempt at solvign this. 

It's not just a summarizer; it's the foundation of a **personal knowledge AI sidekick**. My goal is to create an intelligent agent that reads alongside me, digests vast amounts of content, identifies critical themes, and allows me to query my accumulated knowledge using natural language. This system is designed to augment my thinking, surface non-obvious connections, and transform passive information consumption into an active, structured knowledge base.

Please fork this repo, and extend, tidyup and make it suit your needs. To the team at Readwise, feel free to steal the synthesis scripts / ideas to build into Reader :). 

## 2. What It Does

This Streamlit application automates the entire pipeline of newsletter consumption and analysis:

-   **Automated Collection**: It connects to the **Readwise Reader API** to fetch all new articles from specified categories (newsletters, RSS feeds, etc.) since the last run or you can specifie specific date range too.
-   **Multi-Stage AI Analysis**: Instead of a simple summary, it uses a sophisticated, multi-step process powered by large language models (LLMs) via `litellm` for maximum flexibility:
    1.  **Individual Tagging**: Each article is individually analyzed to extract precise tags and key entities.
    2.  **Theme Clustering**: All tags are collected, semantically refined, and then converted into vector embeddings. **K-Means clustering** is used to group related tags into emergent themes.
    3.  **Theme Naming**: An LLM gives each identified cluster a concise, human-readable name.
    4.  **Deep-Dive Summarization**: For each theme, the system gathers all relevant articles and instructs a powerful LLM to synthesize a deep-dive summary, complete with citations back to the original sources.
-   **Semantic Search**: It generates vector embeddings for every article and summary, enabling a powerful **semantic search** engine. You can ask natural language questions and find the most conceptually relevant content, even if the keywords don't match exactly.
-   **Operational UI Pages**: The Streamlit app exposes the full workflow across dedicated pages:
    -   `1_Synthesizer`: filter newsletters, preview context, estimate cost/time, and trigger new multi-theme syntheses.
    -   `2_Newsletter_Archive`: browse and filter all ingested articles with tag buttons, saved presets, and "more like this" recommendations.
    -   `3_Search`: run semantic search over newsletter and synthesis embeddings for question-style queries.
    -   `3_Prompt_Editor`: review and update the prompt templates stored in the database.
    -   `4_Themes_Visualizations`: explore pyvis graphs mapping themes to their canonical tags.
    -   `5_LLM_Call_History`: inspect call-level telemetry, success rates, durations, and costs.
    -   `6_Syntheses_Archive`: browse past insights packs with final vs. draft comparisons and citation checks.

## 3. High-Level Architecture

The application is designed with modularity and scalability in mind, separating concerns into a logical structure.

```
Readwise-reader-AI-synthesis/
├── streamlit_app.py               # Streamlit entry point / dashboard
├── run_ingestion.py               # CLI for incremental Readwise ingestion
├── backfill_summaries.py          # CLI to regenerate missing summaries
├── backfill_tag_embeddings.py     # CLI to backfill tag embeddings
├── config.py                      # Centralized configuration (paths, models, limits)
├── requirements.txt
├── .env.example                   # Environment variables for CLI scripts
├── .streamlit/
│   └── secrets.toml.example       # Secrets template for Streamlit deployment
├── pages/                         # Individual Streamlit pages (see above)
│   ├── 1_Synthesizer.py
│   ├── 2_Newsletter_Archive.py
│   ├── 3_Prompt_Editor.py
│   ├── 3_Search.py
│   ├── 4_Themes_Visualizations.py
│   ├── 5_LLM_Call_History.py
│   └── 6_Syntheses_Archive.py
├── utils/
│   ├── processing_logic.py        # Stage 1 ingestion + Stage 2 synthesis orchestrator
│   ├── readwise_api.py            # Readwise Reader client + HTML caching
│   ├── database.py                # SQLite access layer, migrations, helpers
│   ├── llm_processor.py           # litellm wrapper with rate limiting & logging
│   ├── helpers.py                 # Shared utilities (HTML cleaning, tag helpers)
│   ├── prompts.py                 # Prompt seeding and retrieval
│   ├── casting.py                 # Safe casting helpers
│   ├── logger.py                  # Run-scoped logging setup
│   └── rate_limiter.py            # Client-side rate limiter
└── README.md
```

-   **Frontend**: **Streamlit** is used for the entire user interface, enabling rapid development of interactive, data-driven pages.
-   **Backend Logic (`utils/`)**: `processing_logic.py` orchestrates both ingestion (`process_single_article`) and synthesis (`synthesize_insights`) while relying on helpers for data cleaning, embeddings, prompt selection, and rate limiting.
-   **LLM Abstraction (`litellm`)**: `llm_processor.py` wraps litellm for completion and embedding calls, enforces per-model rate limits, and logs usage to support cost analytics.
-   **Data Storage**: `database.py` manages schema migrations and persistence in a local **SQLite** file (`newsletter_app.db`) covering newsletters, syntheses, themes, prompts, tag embeddings, LLM call logs, daily usage stats, and saved filter presets.
-   **Data Flow**:
    1.  `run_ingestion.py` (or a backfill script) fetches updated documents from Readwise and caches raw HTML.
    2.  `processing_logic.process_single_article` extracts text, generates tags/summaries, and embeds each article.
    3.  `database.upsert_newsletter` persists metadata, tags, embeddings, and HTML cache references.
    4.  Users work in Streamlit to filter cohorts and trigger `processing_logic.synthesize_insights` for deep-dive packs.
    5.  `llm_processor` tracks every LLM interaction, updating `llm_calls` and `daily_usage_stats` for analytics.
    6.  Streamlit pages render the stored content, analytics, and visualizations directly from SQLite.

## 4. Setup Instructions

Follow these steps to get the application running on your local machine.

### Prerequisites

-   Python 3.10+ (union type hints such as `dict | None` require 3.10 or newer)
-   API keys for:
    -   Readwise Reader
    -   At least one LLM provider (e.g., Google AI Studio for Gemini, OpenAI, Anthropic)
-   A Readwise Reader library with content you want to ingest

### Step 1: Clone the repository & create a virtual environment

First, clone the project repository (or create the project directory) and navigate into it. It is highly recommended to use a virtual environment.

```bash
# Clone the repository (if applicable)
# git clone <your-repo-url>
# cd Readwise-reader-AI-synthesis

# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Configure API keys for both CLI and Streamlit

The Streamlit app reads secrets from `.streamlit/secrets.toml`, while the CLI scripts load environment variables from `.env`.

1.  Copy the provided templates and fill in your keys:

    ```bash
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    cp .env.example .env
    ```

2.  Edit both files with your credentials. **Never commit secrets to version control.**

**Example `.streamlit/secrets.toml`:**
```toml
# Readwise API Key
READWISE_API_KEY = "your_readwise_api_key_here"

# LLM Provider API Keys (add all you intend to use)
GOOGLE_API_KEY = "your_google_gemini_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
```

**Example `.env`:**
```env
READWISE_API_KEY="your_readwise_key"
GOOGLE_API_KEY="your_google_key"
OPENAI_API_KEY="your_openai_key"
```

### Step 4: Prepare local cache & log directories

The ingestion pipeline saves raw HTML in `html_cache/` and run logs in `logs/`. Create both before running the scripts:

```bash
mkdir -p html_cache logs
```

### Step 5: Ingest newsletters from Readwise

Use the ingestion CLI to fetch new or updated items. The first run seeds the SQLite database (`newsletter_app.db`).

```bash
# Fetch items updated since the last recorded run (or the past day on first use)
python run_ingestion.py

# Optional: override the lookback window or run historical backfills
python run_ingestion.py --days 7
python run_ingestion.py --start-date 2023-01-01 --end-date 2023-01-31
```

Maintenance helpers:

- `python backfill_summaries.py --dry-run` shows which legacy articles are missing LLM summaries.
- `python backfill_tag_embeddings.py --limit 10` rebuilds tag embeddings so clustering and search stay accurate.

Re-run ingestion anytime you want to pull in fresh Readwise content.

### Step 6: Launch the Streamlit application

Once your dependencies are installed and keys are configured, run the Streamlit application from your project's root directory.

```bash
streamlit run streamlit_app.py
```

Your default web browser will open a new tab with the application running. You can now configure your desired LLM models in the sidebar and generate your first summary!

## 5. Data Model & Storage

Key data lives in the SQLite database defined at `config.DB_PATH` (`newsletter_app.db`). `utils/database.py` manages schema creation and migrations, covering:

-   `newsletters`: per-article metadata, Readwise summaries, Stage 1 LLM output (`llm_summary`, `llm_tags`), embeddings, and cached HTML paths.
-   `summaries`: Stage 2 insights packs, including draft vs. final text, associated themes, embeddings, and generation context.
-   `themes`: canonical theme names with frequency counters and associated tags for visualization.
-   `tags`: unique tags with pre-computed embeddings to accelerate clustering.
-   `llm_calls`: granular LLM call telemetry (purpose, model, latency, cost) powering the analytics page.
-   `daily_usage_stats`: per-model aggregates used for quota checks and trend charts.
-   `prompts`: editable prompt templates surfaced in the Prompt Editor.
-   `filter_presets` and `app_settings`: saved UI presets plus operational metadata (e.g., last successful ingestion timestamp).

Supporting artifacts:

-   `html_cache/`: raw Readwise article HTML used during Stage 1 processing.
-   `logs/`: timestamped ingestion/backfill logs generated by `utils/logger.py`.

## 6. Future Extension Potential

This application is built on a powerful and extensible foundation. The possibilities for future development are exciting:

-   **Conversational Q&A**: Implement a chat interface that allows you to "talk to your knowledge base." By combining the semantic search capabilities with a conversational LLM, the system could provide direct answers to questions like, "What were the main arguments about AI regulation last month?" and provide citations.
-   **Automated Deep Research Reports**: Integrate with external APIs (e.g., ArXiv, academic search engines) to create a "Deep Research" function. A user could input a topic, and the system would combine insights from their internal newsletter knowledge base with new, external research to generate a comprehensive report.
-   **User Feedback Loop**: Add a simple "thumbs up/down" or rating system for generated summaries and themes. This feedback could be used to refine prompts over time or even as a dataset for fine-tuning a model.
-   **Advanced Trend Analysis**: Create a dedicated analytics page to visualize how the frequency and relationships of themes evolve over time, helping to identify emerging trends in your reading material.
-   **Cloud Deployment & Automation**: Deploy the application to a cloud service (e.g., Streamlit Community Cloud, AWS, GCP) and set up a cron job to automatically run the summary generation process daily, making it a truly autonomous system.

ENJOY!

---
