# newsletter_synthesis_app/backfill_summaries.py
# python backfill_summaries.py --limit 5 # Process only 5 articles for testing
# python backfill_summaries.py --dry-run # Just show what would be done

import asyncio
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

from utils import (
    database, 
    processing_logic, 
    prompts, 
    logger,
    casting
)
from config import DEFAULT_TAGGING_MODEL, DEFAULT_EMBEDDING_MODEL

# --- Concurrency Worker ---

async def backfill_worker(article_row: dict, models: dict, semaphore: asyncio.Semaphore, log):
    """
    An async worker to process and update a single article.
    """
    async with semaphore:
        article_id = article_row['id']
        log.info(f"Starting backfill for article ID: {article_id}")
        try:
            # We reuse the main processing logic for consistency.
            # To do this, we must map the database row to a dictionary that
            # resembles the Readwise API response structure that 
            # process_single_article expects.
            published_date_obj = casting.to_datetime(article_row['published_date'])
            api_like_article_dict = {
                'id': article_row['id'],
                'title': article_row['title'],
                'author': article_row['source'],
                'url': article_row['original_url'],
                'published_at': published_date_obj.isoformat() if published_date_obj else None,
                'summary': article_row['readwise_summary'],
            }

            # Run the same processing as the main ingestion script
            processed_data = await asyncio.to_thread(
                processing_logic.process_single_article,
                api_like_article_dict, 
                models
            )
            
            if processed_data:
                # Use the existing upsert function to update the record
                await asyncio.to_thread(
                    database.upsert_newsletter,
                    processed_data
                )
                log.info(f"Successfully backfilled and saved article ID: {article_id}")
            else:
                log.warning(f"Backfill processing returned no data for article ID: {article_id}")

        except Exception as e:
            log.error(f"Worker failed during backfill for article ID {article_id}: {e}", exc_info=True)


# --- Main Backfill Logic ---

async def main(limit: int, dry_run: bool):
    """
    Main asynchronous function to find and backfill articles with missing summaries.
    """
    run_timestamp = datetime.now(timezone.utc)
    log = logger.setup_logger(run_timestamp)
    log.info("--- Starting Backfill Script ---")

    # Ensure prompts are available
    prompts.initialize_prompts()
    
    # --- 1. Identify Target Articles ---
    log.info("Finding articles with missing LLM summaries...")
    conn = database.get_db_connection()
    # Find articles where llm_summary is NULL or empty.
    # We fetch all columns needed by process_single_article.
    query = "SELECT * FROM newsletters WHERE llm_summary IS NULL OR llm_summary = ''"
    if limit:
        query += f" LIMIT {limit}"
        
    articles_to_backfill = conn.execute(query).fetchall()
    conn.close()

    if not articles_to_backfill:
        log.info("No articles found needing a backfill. All data is up-to-date. Exiting.")
        return

    log.info(f"Found {len(articles_to_backfill)} articles to backfill.")
    
    if dry_run:
        log.info("DRY RUN enabled. The following article IDs would be processed:")
        for article in articles_to_backfill:
            print(f"  - ID: {article['id']}, Title: {article['title']}")
        log.info("Exiting due to dry run. No changes were made.")
        return

    # --- Confirmation Step ---
    print(f"\nThis script will process {len(articles_to_backfill)} articles.")
    print("This will make LLM API calls and incur costs.")
    confirm = input("Are you sure you want to continue? (yes/no): ")
    if confirm.lower() != 'yes':
        log.warning("User aborted the script. Exiting.")
        print("Aborted.")
        return

    # --- 2. Process Concurrently ---
    log.info("Starting concurrent processing...")
    models = {'tagging': DEFAULT_TAGGING_MODEL, 'embedding': DEFAULT_EMBEDDING_MODEL}
    
    CONCURRENT_TASKS = 5
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    
    tasks = [backfill_worker(article, models, semaphore, log) for article in articles_to_backfill]
    
    await asyncio.gather(*tasks)

    log.info("--- Backfill Script Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill missing summaries and attributes for older articles.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of articles to process in this run (e.g., for testing)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which articles would be processed without actually running them."
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.limit, args.dry_run))