# newsletter_synthesis_app/utils/database.py

import sqlite3
import json
import numpy as np
import io
from datetime import datetime, date
import config # Use relative import
from utils.logger import get_logger

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

# --- Schema Creation ---
def create_tables():
    """Creates all necessary tables in the database if they don't exist."""
    conn = get_db_connection()
    if not conn: return

    try:
        with conn:
            # Summaries table
            conn.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_date DATETIME NOT NULL,
                themes_json TEXT NOT NULL,
                content TEXT NOT NULL,
                newsletter_count INTEGER NOT NULL,
                embedding array
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
        logger.info("Database tables verified/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
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
        cursor = conn.execute('SELECT MAX(published_date) FROM newsletters')
        result = cursor.fetchone()[0]
        if result:
            # --- FIX: Handle both string and datetime object cases robustly ---
            if isinstance(result, str):
                return datetime.fromisoformat(result)
            # If it's already a datetime object (some drivers might do this)
            elif isinstance(result, datetime):
                return result
        return None
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