# Newsletter Synthesis AI

## 1. The Vision: My Personal Knowledge AI

I built this application to solve a problem I face daily: an overwhelming firehose of information from dozens of newsletters. Key insights were getting lost in the noise, and I lacked an efficient way to connect ideas across different sources over time.

This project is my answer. It's not just a summarizer; it's the foundation of a **personal knowledge AI system**. My goal is to create an intelligent agent that reads alongside me, digests vast amounts of content, identifies critical themes, and allows me to query my accumulated knowledge using natural language. This system is designed to augment my thinking, surface non-obvious connections, and transform passive information consumption into an active, structured knowledge base.

## 2. What It Does

This Streamlit application automates the entire pipeline of newsletter consumption and analysis:

-   **Automated Collection**: It connects to the **Readwise Reader API** to fetch all new articles from specified categories (newsletters, RSS feeds, etc.) since the last run.
-   **Multi-Stage AI Analysis**: Instead of a simple summary, it uses a sophisticated, multi-step process powered by large language models (LLMs) via `litellm` for maximum flexibility:
    1.  **Individual Tagging**: Each article is individually analyzed to extract precise tags and key entities.
    2.  **Theme Clustering**: All tags are collected, semantically refined, and then converted into vector embeddings. **K-Means clustering** is used to group related tags into emergent themes.
    3.  **Theme Naming**: An LLM gives each identified cluster a concise, human-readable name.
    4.  **Deep-Dive Summarization**: For each theme, the system gathers all relevant articles and instructs a powerful LLM to synthesize a deep-dive summary, complete with citations back to the original sources.
-   **Semantic Search**: It generates vector embeddings for every article and summary, enabling a powerful **semantic search** engine. You can ask natural language questions and find the most conceptually relevant content, even if the keywords don't match exactly.
-   **Interactive Visualization**: A dynamic network graph visualizes the relationships between identified themes and their constituent tags, providing a high-level map of your knowledge base.
-   **Comprehensive UI**: A multi-page Streamlit application provides:
    -   A dashboard to control the generation process and view the latest summary.
    -   Historical views of all past summaries and archived articles.
    -   A dedicated page for LLM call analytics to monitor costs, performance, and model usage.

## 3. High-Level Architecture

The application is designed with modularity and scalability in mind, separating concerns into a logical structure.

```
newsletter_synthesis_app/
├── .streamlit/secrets.toml    # Secure API key storage
├── pages/                     # Each Streamlit page is a separate module
├── utils/                     # Core backend logic
│   ├── database.py            # SQLite database management (CRUD operations)
│   ├── llm_processor.py       # Handles all LLM calls via litellm (completion, embeddings)
│   ├── readwise_api.py        # Manages fetching data from Readwise
│   └── summary_generator.py   # Orchestrates the main AI processing pipeline
├── config.py                  # Centralized configuration (models, paths, etc.)
├── streamlit_app.py           # Main application entry point and dashboard
└── README.md                  # This file
```

-   **Frontend**: **Streamlit** is used for the entire user interface, enabling rapid development of interactive, data-driven pages.
-   **Backend Logic (`utils/`)**: A collection of Python modules handles the core functionality. `summary_generator.py` acts as the conductor, calling other utilities to fetch data, process it with LLMs, and store it in the database.
-   **LLM Abstraction (`litellm`)**: `litellm` is used as a universal interface to various LLM providers (Google Gemini, OpenAI, Anthropic, etc.). This makes it trivial to swap models for different tasks (e.g., use a fast, cheap model for tagging and a powerful, expensive one for the final summary) directly from the UI.
-   **Data Storage**: A local **SQLite** database stores all processed data, including summaries, newsletter metadata, themes, LLM call logs, and vector embeddings for semantic search.
-   **Data Flow**:
    1.  The **Streamlit UI** triggers a generation run.
    2.  `summary_generator.py` orchestrates the process.
    3.  `readwise_api.py` fetches new articles.
    4.  `llm_processor.py` is called repeatedly for tagging, theme naming, and summarization.
    5.  Embeddings are generated for semantic search.
    6.  `database.py` saves all results to the SQLite DB.
    7.  The UI reads from the database to display results.

## 4. Setup Instructions

Follow these steps to get the application running on your local machine.

### Prerequisites

-   Python 3.13+
-   API keys for:
    -   Readwise Reader
    -   At least one LLM provider (e.g., Google AI Studio for Gemini, OpenAI, Anthropic)

### Step 1: Clone the Repository & Set Up Virtual Environment

First, clone the project repository (or create the project directory) and navigate into it. It is highly recommended to use a virtual environment.

```bash
# Clone the repository (if applicable)
# git clone <your-repo-url>
# cd newsletter_synthesis_app

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Your secret API keys are managed using Streamlit's secrets management.

1.  Create a file named `secrets.toml` inside the `.streamlit/` directory.
2.  Add your API keys to this file. **This file should never be committed to version control.**

**Example `.streamlit/secrets.toml`:**
```toml
# Readwise API Key
READWISE_API_KEY = "your_readwise_api_key_here"

# LLM Provider API Keys (add all you intend to use)
GOOGLE_API_KEY = "your_google_gemini_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
```

### Step 4: Run the Application

Once your dependencies are installed and keys are configured, run the Streamlit application from your project's root directory.

```bash
streamlit run streamlit_app.py
```

Your default web browser will open a new tab with the application running. You can now configure your desired LLM models in the sidebar and generate your first summary!

## 5. Future Extension Potential

This application is built on a powerful and extensible foundation. The possibilities for future development are exciting:

-   **Conversational Q&A**: Implement a chat interface that allows you to "talk to your knowledge base." By combining the semantic search capabilities with a conversational LLM, the system could provide direct answers to questions like, "What were the main arguments about AI regulation last month?" and provide citations.
-   **Automated Deep Research Reports**: Integrate with external APIs (e.g., ArXiv, academic search engines) to create a "Deep Research" function. A user could input a topic, and the system would combine insights from their internal newsletter knowledge base with new, external research to generate a comprehensive report.
-   **User Feedback Loop**: Add a simple "thumbs up/down" or rating system for generated summaries and themes. This feedback could be used to refine prompts over time or even as a dataset for fine-tuning a model.
-   **Advanced Trend Analysis**: Create a dedicated analytics page to visualize how the frequency and relationships of themes evolve over time, helping to identify emerging trends in your reading material.
-   **Cloud Deployment & Automation**: Deploy the application to a cloud service (e.g., Streamlit Community Cloud, AWS, GCP) and set up a cron job to automatically run the summary generation process daily, making it a truly autonomous system.

---