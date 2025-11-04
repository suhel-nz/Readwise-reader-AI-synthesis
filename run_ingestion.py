# newsletter_synthesis_app/run_ingestion.py
# python run_ingestion.py (for normal, incremental updates)
# python run_ingestion.py --days 7 (to look back 7 days if no previous run found)
# python run_ingestion.py --start-date 2023-01-01 (to backfill all of January 2023)
# python run_ingestion.py --start-date 2023-01-01 --end-date 2023-01-31 (to backfill all of January 2023)

import asyncio
import argparse 
from datetime import datetime, timedelta, timezone # MODIFIED: Import timezone
from dotenv import load_dotenv # NEW IMPORT
load_dotenv()
import json

from utils import (
    database, 
    readwise_api, 
    processing_logic, 
    prompts, 
    logger, 
    casting, 
    llm_processor,
    helpers
) 
from utils.casting import to_numpy_array
from config import DEFAULT_TAGGING_MODEL, DEFAULT_EMBEDDING_MODEL

# --- Concurrency Worker ---

async def process_and_save_worker(article: dict, models: dict, semaphore: asyncio.Semaphore, log):
    """
    An async-compatible worker that processes a single article and saves the result.
    Uses a semaphore to limit concurrency.
    """
    async with semaphore:
        log.info(f"Starting processing for article ID: {article.get('id')}")
        try:
            # Run the synchronous, CPU/IO-bound processing in a separate thread
            # to avoid blocking the asyncio event loop.
            processed_data = await asyncio.to_thread(
                processing_logic.process_single_article,
                article, 
                models
            )
            
            if processed_data:
                # Run the synchronous database call in a separate thread
                await asyncio.to_thread(
                    database.upsert_newsletter,
                    processed_data
                )
                log.info(f"Successfully processed and saved article ID: {article.get('id')}")
            else:
                log.warning(f"Processing returned no data for article ID: {article.get('id')}")
            return processed_data # Return the data for tag collection

        except Exception as e:
            log.error(f"Worker failed for article ID {article.get('id')}: {e}", exc_info=True)


# --- Main Ingestion Logic ---

async def main(lookback_days: int, date_override: dict = {}):
    """
    Main asynchronous function to run the ingestion pipeline with concurrent workers.
    """
    run_timestamp = datetime.now(timezone.utc) # MODIFIED: Use aware datetime
    log = logger.setup_logger(run_timestamp)
    log.info("--- Starting Ingestion Run ---")

    # Initialize prompts in DB if they don't exist
    prompts.initialize_prompts()

    # --- REFACTORED: Use a persistent setting for robust incremental updates ---
    # This ensures we fetch based on when articles were last *updated* in Readwise, not published.
    last_run_setting_key = "last_ingestion_run_end_date"
    
    if date_override.get('start_date'):
        start_date = date_override['start_date']
        log.info(f"Using manual start date override: {start_date.isoformat()}")
    else:
        last_run_str = database.get_setting(last_run_setting_key)
        if last_run_str:
            start_date = casting.to_datetime(last_run_str)
            log.info(f"Last successful run ended at {start_date.isoformat()}. Starting from there.")
        else:
            # On the very first run, only look back a short time to avoid a huge initial load.
            # Use --start-date for a full historical backfill.
            start_date = datetime.now(timezone.utc) - timedelta(days=lookback_days) # MODIFIED: Use aware datetime and correct lookback
            log.warning(f"No previous run found. Defaulting to a lookback of 1 day.")
            
    end_date = date_override.get('end_date') or datetime.now(timezone.utc) # MODIFIED: Use aware datetime

    log.info(f"Fetching articles from {start_date.isoformat()} to {end_date.isoformat()}")
    
    # Use a lambda to pass log messages from the synchronous function
    articles = readwise_api.fetch_readwise_articles(start_date, end_date, lambda msg: log.info(msg))
    
    if not articles:
        log.info("No new articles found. Exiting.")
        return
    
    # --- REFACTORED: Filter out articles that are already up-to-date ---
    article_ids_from_api = {article['id'] for article in articles}
    if article_ids_from_api:
        conn = database.get_db_connection()
        placeholders = ','.join('?' for _ in article_ids_from_api)
        query = f"SELECT id, processed_date FROM newsletters WHERE id IN ({placeholders})"
        # Ensure dates from DB are parsed as aware UTC datetimes
        existing_articles_map = {
            row['id']: casting.to_datetime(row['processed_date']).replace(tzinfo=timezone.utc) if casting.to_datetime(row['processed_date']) else None 
            for row in conn.execute(query, list(article_ids_from_api)).fetchall()}
        conn.close()
    else:
        existing_articles_map = {}

    articles_to_process = []
    for article in articles:
        article_id = article['id']
        api_updated_at = casting.to_datetime(article['updated_at'])
        
        # Process if the article is new...
        if article_id not in existing_articles_map:
            articles_to_process.append(article)
    
    log.info(f"Found {len(articles)} articles from API. After filtering for new or updated content, {len(articles_to_process)} articles will be processed.")
    
    if not articles_to_process:
        log.info("All fetched articles have already been processed. Exiting.")
        return
    
    models = {'tagging': DEFAULT_TAGGING_MODEL, 'embedding': DEFAULT_EMBEDDING_MODEL}
    
    # --- Concurrent Processing ---
    # Use a semaphore to limit the number of concurrent processing tasks.
    # This prevents overwhelming your machine and helps manage API rate limits.
    # A value of 5 means up to 5 articles will be processed in parallel.
    CONCURRENT_TASKS = 5
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    
    tasks = [process_and_save_worker(article, models, semaphore, log) for article in articles_to_process]
    
    # Gather results from the workers
    processed_results = await asyncio.gather(*tasks)

    # --- NEW: Process and embed all unique tags from this run ---
    log.info("Gathering all unique tags from this run to pre-compute embeddings...")
    all_tags_from_run = set()
    
    successful_results = [res for res in processed_results if res]
    for processed_article in successful_results:
        try:
            # Extract tags from the data that was just processed and returned by the worker
            tags_data = json.loads(processed_article.get('llm_tags', '{}'))
            tags = tags_data.get('tags', [])
            if isinstance(tags, list):
                all_tags_from_run.update(tags)
        except (json.JSONDecodeError, TypeError):
            continue
    
    if all_tags_from_run:
        existing_tags_with_embeddings = await asyncio.to_thread(database.get_tags_with_embeddings, list(all_tags_from_run))
        new_tags_to_embed = all_tags_from_run - set(existing_tags_with_embeddings.keys())
        
        if new_tags_to_embed:
            log.info(f"Found {len(new_tags_to_embed)} new tags to embed.")
            tag_tasks = [helpers.embed_and_save_tag_worker(tag, models['embedding'], semaphore, log) for tag in new_tags_to_embed]
            embedding_results = await asyncio.gather(*tag_tasks)
            
            successful_embeddings = [res for res in embedding_results if res is not None]
            if successful_embeddings:
                await asyncio.to_thread(database.upsert_tags, successful_embeddings)

    # --- NEW: Save the end time of this successful run for the next incremental run ---
    if not date_override: # Don't update the setting during manual backfills
        database.set_setting(last_run_setting_key, end_date.isoformat())

    log.info("--- Ingestion Run Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Newsletter AI ingestion pipeline.")
    parser.add_argument(
        "--days", type=int, default=1,
        help="Default number of days to look back if no previous run is found."
    )
    # --- NEW: Add arguments for manual date override ---
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Force processing from a specific start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Force processing up to a specific end date (YYYY-MM-DD). Defaults to now."
    )
    args = parser.parse_args()
    
    # --- NEW: Logic to handle the arguments ---
    # This will be passed into the main() function
    date_override = {}
    if args.start_date:
        date_override['start_date'] = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) # MODIFIED
    if args.end_date:
        date_override['end_date'] = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) # MODIFIED
    
    asyncio.run(main(args.days, date_override))