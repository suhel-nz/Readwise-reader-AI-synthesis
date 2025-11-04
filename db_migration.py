import sqlite3
import config

print("--- Starting Database Schema Migration ---")

# --- Define the target schema: What your tables SHOULD look like ---
TARGET_SCHEMA = {
    "newsletters": [
        "id", "title", "source", "original_url", "published_date",
        "processed_date", "category", "tags", "readwise_summary",
        "llm_summary", "llm_tags", "html_content_path", "summary_id", "embedding"
    ],
    "summaries": [
        "id", "generated_date", "themes_json", "content",
        "newsletter_count", "embedding", "generation_context_json"
    ],
    "filter_presets": [
        "id", "name", "filters_json", "created_date"
    ]
    # Add other tables here if they need changes in the future
}

# --- Utility function to check existing columns ---
def get_existing_columns(cursor, table_name):
    """Fetches the names of existing columns for a given table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row[1] for row in cursor.fetchall()}

# --- Main migration logic ---
def run_migration():
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        print(f"Connected to database at {config.DB_PATH}")

        for table_name, target_columns in TARGET_SCHEMA.items():
            print(f"\nChecking table: '{table_name}'...")
            existing_columns = get_existing_columns(cursor, table_name)
            print(f"Found existing columns: {existing_columns}")

            for column_name in target_columns:
                if column_name not in existing_columns:
                    # For simplicity, we'll add new columns as TEXT.
                    # SQLite is flexible with types, so this is generally safe.
                    # For more complex types like BLOB (for embeddings), you'd specify it.
                    column_type = "ARRAY" if column_name == "embedding" else "TEXT"
                    
                    print(f"-> Column '{column_name}' is missing. Adding it...")
                    try:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                        print(f"   ...Successfully added '{column_name}' to '{table_name}'.")
                    except sqlite3.OperationalError as e:
                        print(f"   ...Error adding column '{column_name}': {e}")
                else:
                    print(f"-> Column '{column_name}' already exists. OK.")

        conn.commit()
        conn.close()
        print("\n--- Database Schema Migration Complete ---")

    except sqlite3.Error as e:
        print(f"\n--- An error occurred during migration: {e} ---")

if __name__ == "__main__":
    run_migration()