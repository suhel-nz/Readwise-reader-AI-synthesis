# newsletter_synthesis_app/utils/database.py

import sqlite3
import json
import numpy as np
import io
from datetime import datetime, date
import config # Use relative import
from utils.logger import get_logger
from utils.casting import to_datetime # NEW: Using our casting utility

logger = get_logger()

# --- Utility functions for embedding conversion ---
def adapt_array(arr):
    """ http://stackoverflow.com/a/31312102/190597 (SoulNibbler) """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(config.DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def _migrate_schema(conn):
    """
    Checks the existing database schema and applies necessary alterations.
    This makes the app robust to schema changes in the code.
    """
    cursor = conn.cursor()
    logger.info("Checking for necessary database migrations...")

    # --- Define the full target schema ---
    TARGET_SCHEMA = {
        "newsletters": {
            "id": "TEXT PRIMARY KEY", "title": "TEXT NOT NULL", "source": "TEXT",
            "original_url": "TEXT", "published_date": "DATETIME", "processed_date": "DATETIME NOT NULL",
            "category": "TEXT", "tags": "TEXT", "readwise_summary": "TEXT",
            "llm_summary": "TEXT", "llm_tags": "TEXT", "html_content_path": "TEXT",
            "summary_id": "INTEGER", "embedding": "ARRAY",
            "FOREIGN KEY (summary_id)": "REFERENCES summaries(id)" # Note: Foreign keys can't be added with ALTER easily
        },
        "summaries": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT", "generated_date": "DATETIME NOT NULL",
            "themes_json": "TEXT NOT NULL", "content": "TEXT NOT NULL",
            "draft_content": "TEXT", "newsletter_count": "INTEGER NOT NULL", "embedding": "ARRAY",
            "generation_context_json": "TEXT"
        }, 
        "filter_presets": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT", "name": "TEXT UNIQUE NOT NULL",
            "filters_json": "TEXT NOT NULL", "created_date": "DATETIME NOT NULL"
        },
    }

    def get_existing_columns(table_name):
        cursor.execute(f"PRAGMA table_info({table_name});")
        return {row[1] for row in cursor.fetchall()}

    for table_name, target_columns in TARGET_SCHEMA.items():
        try:
            existing_columns = get_existing_columns(table_name)
            for col_name, col_type in target_columns.items():
                if col_name not in existing_columns and not col_name.startswith("FOREIGN KEY"):
                    logger.info(f"Schema mismatch: Adding column '{col_name}' to table '{table_name}'.")
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError as e:
            # This can happen if the table doesn't exist yet, which is fine.
            if "no such table" not in str(e):
                logger.error(f"Error migrating table '{table_name}': {e}")

# --- Schema Creation ---
def create_tables():
    """Creates all necessary tables in the database if they don't exist."""
    conn = get_db_connection()
    if not conn: return

    try:
        with conn:
            # First, run migrations on existing structures
            _migrate_schema(conn)

            # Summaries table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_date DATETIME NOT NULL,
                themes_json TEXT NOT NULL,
                content TEXT NOT NULL,
                draft_content TEXT,
                newsletter_count INTEGER NOT NULL,
                embedding array, 
                generation_context_json TEXT
            );
            ''')
            # Newsletters table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS newsletters (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT,
                original_url TEXT,
                published_date DATETIME,
                processed_date DATETIME NOT NULL,
                category TEXT,
                tags TEXT,
                llm_tags TEXT,
                readwise_summary TEXT, llm_summary TEXT, llm_tags TEXT,
                html_content_path TEXT,
                summary_id INTEGER,
                embedding array,
                FOREIGN KEY (summary_id) REFERENCES summaries(id)
            );
            ''')
            # Themes table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS themes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_seen DATETIME NOT NULL,
                associated_canonical_tags TEXT
            );
            ''')
            # App Settings table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            ''')
            # LLM Calls table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                purpose TEXT NOT NULL,
                model_used TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL,
                duration_ms INTEGER,
                status TEXT,
                error_message TEXT,
                metadata TEXT
            );
            ''')
            # --- NEW: Daily Usage Stats Table ---
            conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usage_date DATE NOT NULL,
                model TEXT NOT NULL,
                call_count INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(usage_date, model)
            );
            ''')
            # NEW: prompts table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                template TEXT NOT NULL,
                description TEXT,
                last_updated DATETIME NOT NULL
            );
            ''')
            conn.execute('''
            CREATE TABLE IF NOT EXISTS filter_presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                filters_json TEXT NOT NULL,
                created_date DATETIME NOT NULL
            );
            ''')
            # --- NEW: Table to store tags and their pre-computed embeddings ---
            conn.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                tag TEXT PRIMARY KEY,
                embedding ARRAY,
                last_seen DATETIME NOT NULL
            );
            ''')
        logger.info("Database tables verified/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        conn.close()

# --- NEW: Preset Management Functions ---
def get_all_presets() -> list[dict]:
    """Retrieves all saved filter presets."""
    conn = get_db_connection()
    if not conn: return []
    try:
        rows = conn.execute("SELECT id, name FROM filter_presets ORDER BY name").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_preset_filters(preset_id: int) -> dict | None:
    """Retrieves the filter JSON for a specific preset ID."""
    conn = get_db_connection()
    if not conn: return None
    try:
        row = conn.execute("SELECT filters_json FROM filter_presets WHERE id = ?", (preset_id,)).fetchone()
        return json.loads(row['filters_json']) if row else None
    finally:
        conn.close()

def save_preset(name: str, filters: dict):
    """Saves a new filter preset to the database."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            conn.execute(
                "INSERT INTO filter_presets (name, filters_json, created_date) VALUES (?, ?, ?)",
                (name, json.dumps(filters, default=str), datetime.now())
            )
        logger.info(f"Saved new filter preset: '{name}'")
    except sqlite3.IntegrityError:
        logger.error(f"A preset with the name '{name}' already exists.")
        # In a real app, you'd raise this error to the UI
    finally:
        conn.close()

def delete_preset(preset_id: int):
    """Deletes a filter preset."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            conn.execute("DELETE FROM filter_presets WHERE id = ?", (preset_id,))
        logger.info(f"Deleted preset ID: {preset_id}")
    finally:
        conn.close()

# --- NEW: Prompt Management Functions ---
def get_prompt(name: str) -> dict | None:
    """Retrieves a single prompt template by its unique name."""
    conn = get_db_connection()
    if not conn: return None
    try:
        row = conn.execute("SELECT * FROM prompts WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_all_prompts() -> list[dict]:
    """Retrieves all prompts from the database."""
    conn = get_db_connection()
    if not conn: return []
    try:
        rows = conn.execute("SELECT * FROM prompts ORDER BY name").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def update_prompt(name: str, template: str):
    """Updates a prompt's template text."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            conn.execute(
                "UPDATE prompts SET template = ?, last_updated = ? WHERE name = ?",
                (template, datetime.now(), name)
            )
        logger.info(f"Prompt '{name}' updated successfully.")
    except sqlite3.Error as e:
        logger.error(f"Failed to update prompt '{name}': {e}")

def seed_initial_prompts(prompts_to_seed: dict):
    """Seeds the database with initial prompts if they don't exist."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            for name, data in prompts_to_seed.items():
                conn.execute(
                    """
                    INSERT INTO prompts (name, template, description, last_updated)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO NOTHING
                    """,
                    (name, data['template'], data['description'], datetime.now())
                )
        logger.info("Initial prompts seeded successfully.")
    except sqlite3.Error as e:
        logger.error(f"Failed to seed prompts: {e}")

# --- NEW: Tag Embedding Management Functions ---
def get_tags_with_embeddings(tags_list: list[str]) -> dict:
    """
    Fetches tags and their pre-computed embeddings from the DB for a given list of tags.
    Returns a dictionary of {tag: embedding_array}.
    """
    conn = get_db_connection()
    if not conn or not tags_list: return {}
    try:
        placeholders = ','.join('?' for _ in tags_list)
        query = f"SELECT tag, embedding FROM tags WHERE tag IN ({placeholders})"
        rows = conn.execute(query, tags_list).fetchall()
        # Return a dict of tag -> embedding for tags that were found and have an embedding
        return {row['tag']: row['embedding'] for row in rows if row['embedding'] is not None}
    finally:
        conn.close()

def upsert_tags(tags_to_upsert: list[dict]):
    """
    Inserts new tags with their embeddings, or updates the last_seen timestamp for existing ones.
    Expects a list of dicts: [{'tag': str, 'embedding': np.ndarray, 'last_seen': datetime}]
    """
    conn = get_db_connection()
    if not conn or not tags_to_upsert: return
    try:
        with conn:
            conn.executemany("""
                INSERT INTO tags (tag, embedding, last_seen)
                VALUES (:tag, :embedding, :last_seen)
                ON CONFLICT(tag) DO UPDATE SET
                    last_seen=excluded.last_seen
            """, tags_to_upsert)
        logger.info(f"Upserted {len(tags_to_upsert)} tags into the database.")
    except sqlite3.Error as e:
        logger.error(f"Failed to upsert tags: {e}")
    finally:
        conn.close()

# --- NEW: Daily Usage Functions ---
def get_daily_usage(model: str, usage_date: date = None) -> tuple[int, int]:
    """Retrieves the current call and token count for a given model for a specific date."""
    if usage_date is None:
        usage_date = date.today()
    
    conn = get_db_connection()
    if not conn: return (0, 0)
    try:
        cursor = conn.execute(
            "SELECT call_count, token_count FROM daily_usage_stats WHERE usage_date = ? AND model = ?",
            (usage_date, model)
        )
        row = cursor.fetchone()
        return (row['call_count'], row['token_count']) if row else (0, 0)
    finally:
        conn.close()

def update_daily_usage(model: str, calls_to_add: int, tokens_to_add: int, usage_date: date = None):
    """Atomically increments the call and token counts for a given model for a specific date."""
    if usage_date is None:
        usage_date = date.today()

    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            conn.execute("""
                INSERT INTO daily_usage_stats (usage_date, model, call_count, token_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(usage_date, model) DO UPDATE SET
                    call_count = call_count + excluded.call_count,
                    token_count = token_count + excluded.token_count
            """, (usage_date, model, calls_to_add, tokens_to_add))
        logger.info(f"Updated daily usage for {model}: +{calls_to_add} calls, +{tokens_to_add} tokens.")
    except sqlite3.Error as e:
        logger.error(f"Failed to update daily usage for {model}: {e}")
    finally:
        conn.close()

# --- Data Insertion Functions ---
def save_llm_call(data):
    """Saves a record of an LLM API call."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            conn.execute('''
            INSERT INTO llm_calls (timestamp, purpose, model_used, input_tokens, output_tokens, cost, duration_ms, status, error_message, metadata)
            VALUES (:timestamp, :purpose, :model_used, :input_tokens, :output_tokens, :cost, :duration_ms, :status, :error_message, :metadata)
            ''', data)
    except sqlite3.Error as e:
        logger.error(f"Failed to save LLM call log: {e}")
    finally:
        conn.close()

def save_summary_only(summary_data: dict) -> int | None:
    """Saves a new summary to the database without updating any newsletters.
    Expects keys: generated_date, themes, content, draft_content (optional), newsletter_count, embedding, generation_context_json (optional)
    """
    conn = get_db_connection()
    if not conn: return None
    try:
        with conn:
            cursor = conn.execute('''
            INSERT INTO summaries (generated_date, themes_json, content, draft_content, newsletter_count, embedding, generation_context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary_data['generated_date'],
                json.dumps(summary_data['themes']),
                summary_data['content'],
                summary_data.get('draft_content'),
                summary_data['newsletter_count'],
                summary_data.get('embedding'),
                summary_data.get('generation_context_json')
            ))
            summary_id = cursor.lastrowid
            logger.info(f"Saved new summary with ID: {summary_id}")
            return summary_id
    except sqlite3.Error as e:
        logger.error(f"Database transaction failed while saving summary: {e}")
        return None
    finally:
        if conn:
            conn.close()

def upsert_newsletter(newsletter_data: dict):
    """
    Inserts a new newsletter record or updates an existing one based on the ID.
    """
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            # The 'summary_id' is excluded as it's only set during synthesis
            conn.execute('''
            INSERT INTO newsletters (
                id, title, source, original_url, published_date, processed_date, 
                category, tags, readwise_summary, llm_summary, llm_tags, 
                html_content_path, embedding
            ) VALUES (
                :id, :title, :source, :original_url, :published_date, :processed_date,
                :category, :tags, :readwise_summary, :llm_summary, :llm_tags,
                :html_content_path, :embedding
            ) ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                source=excluded.source,
                original_url=excluded.original_url,
                published_date=excluded.published_date,
                processed_date=excluded.processed_date,
                category=excluded.category,
                tags=excluded.tags,
                readwise_summary=excluded.readwise_summary,
                llm_summary=excluded.llm_summary,
                llm_tags=excluded.llm_tags,
                html_content_path=excluded.html_content_path,
                embedding=excluded.embedding
            ''', newsletter_data)
        logger.info(f"Successfully upserted newsletter ID: {newsletter_data.get('id')}")
    except sqlite3.Error as e:
        logger.error(f"Database upsert failed for newsletter ID {newsletter_data.get('id')}: {e}")
    finally:
        if conn:
            conn.close()

def save_summary_and_newsletters(summary_data, newsletter_list):
    """Saves a new summary and links its associated newsletters in a transaction."""
    conn = get_db_connection()
    if not conn: return None
    try:
        with conn:
            # Insert the summary
            cursor = conn.execute('''
            INSERT INTO summaries (generated_date, themes_json, content, newsletter_count, embedding)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                summary_data['generated_date'],
                json.dumps(summary_data['themes']),
                summary_data['content'],
                len(newsletter_list),
                summary_data.get('embedding')
            ))
            summary_id = cursor.lastrowid
            logger.info(f"Saved new summary with ID: {summary_id}")

            # Update or insert newsletters
            for newsletter in newsletter_list:
                newsletter['summary_id'] = summary_id
                conn.execute('''
                INSERT INTO newsletters (id, title, source, original_url, published_date, processed_date, category, tags, llm_tags, html_content_path, summary_id, embedding)
                VALUES (:id, :title, :source, :original_url, :published_date, :processed_date, :category, :tags, :llm_tags, :html_content_path, :summary_id, :embedding)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    source=excluded.source,
                    original_url=excluded.original_url,
                    published_date=excluded.published_date,
                    processed_date=excluded.processed_date,
                    category=excluded.category,
                    tags=excluded.tags,
                    llm_tags=excluded.llm_tags,
                    html_content_path=excluded.html_content_path,
                    summary_id=excluded.summary_id,
                    embedding=excluded.embedding
                ''', newsletter)
            logger.info(f"Upserted {len(newsletter_list)} newsletters linked to summary ID {summary_id}")
            return summary_id
    except sqlite3.Error as e:
        logger.error(f"Database transaction failed: {e}")
        return None
    finally:
        conn.close()

def update_themes(themes_dict):
    """Updates the themes table with new frequencies and timestamps."""
    conn = get_db_connection()
    if not conn: return
    now = datetime.now()
    try:
        with conn:
            for theme_name, canonical_tags in themes_dict.items():
                conn.execute('''
                INSERT INTO themes (name, frequency, last_seen, associated_canonical_tags)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    frequency = frequency + 1,
                    last_seen = excluded.last_seen,
                    associated_canonical_tags = excluded.associated_canonical_tags
                ''', (theme_name, now, json.dumps(canonical_tags)))
        logger.info(f"Updated {len(themes_dict)} themes in the database.")
    except sqlite3.Error as e:
        logger.error(f"Failed to update themes: {e}")
    finally:
        conn.close()

# --- Data Retrieval Functions ---
def get_setting(key):
    """Retrieves a value from the app_settings table."""
    conn = get_db_connection()
    if not conn: return None
    try:
        cursor = conn.execute('SELECT value FROM app_settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row['value'] if row else None
    finally:
        conn.close()

def set_setting(key, value):
    """Saves or updates a value in the app_settings table."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn:
            conn.execute('''
            INSERT INTO app_settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            ''', (key, str(value)))
    finally:
        conn.close()

def get_latest_article_date():
    """Gets the most recent published_date from all processed newsletters."""
    conn = get_db_connection()
    if not conn: return None
    try:
        result = conn.execute('SELECT MAX(published_date) FROM newsletters').fetchone()[0]
        return to_datetime(result) # Use the caster for robust conversion
    finally:
        if conn:
            conn.close()

# Generic fetch function for pagination
def fetch_paginated_data(query, params=(), page=1, page_size=20):
    conn = get_db_connection()
    if not conn: return [], 0
    
    offset = (page - 1) * page_size
    
    # Get total count
    count_query = f"SELECT COUNT(*) FROM ({query})"
    total_count = conn.execute(count_query, params).fetchone()[0]
    
    # Get paginated data
    paginated_query = f"{query} LIMIT ? OFFSET ?"
    data = conn.execute(paginated_query, (*params, page_size, offset)).fetchall()
    
    conn.close()
    return data, total_count


# --- Initial DB Setup ---
# This function is called when the application starts
create_tables()
