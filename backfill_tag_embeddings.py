# newsletter_synthesis_app/backfill_tag_embeddings.py
# python backfill_tag_embeddings.py
# python backfill_tag_embeddings.py --limit 10 --dry-run

import asyncio
import argparse
from datetime import datetime, timezone
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from utils import (
    database,
    llm_processor,
    logger,
    helpers,
    prompts
)
from utils.casting import to_numpy_array
from config import DEFAULT_EMBEDDING_MODEL

async def main(limit: int, dry_run: bool):
    """
    Main function to find all unique tags in the database and backfill embeddings for those missing them.
    """
    run_timestamp = datetime.now(timezone.utc)
    log = logger.setup_logger(run_timestamp)
    log.info("--- Starting Tag Embedding Backfill Script ---")

    prompts.initialize_prompts()

    # 1. Get all unique tags from the entire newsletters table
    log.info("Scanning database for all unique tags...")
    conn = database.get_db_connection()
    all_articles_df = pd.read_sql_query("SELECT llm_tags FROM newsletters", conn)
    conn.close()
    
    all_unique_tags = set(helpers.extract_tags_from_dataframe(all_articles_df))
    log.info(f"Found a total of {len(all_unique_tags)} unique tags across all articles.")

    # 2. Find out which tags already have embeddings
    existing_tags_map = database.get_tags_with_embeddings(list(all_unique_tags))
    tags_to_backfill = sorted(list(all_unique_tags - set(existing_tags_map.keys())))

    if not tags_to_backfill:
        log.info("All unique tags already have embeddings. No backfill needed. Exiting.")
        return

    log.info(f"Found {len(tags_to_backfill)} tags that need embeddings backfilled.")

    if limit:
        tags_to_backfill = tags_to_backfill[:limit]
        log.warning(f"Applying --limit. Will only process {len(tags_to_backfill)} tags.")

    if dry_run:
        log.info("DRY RUN enabled. The following tags would be processed:")
        for tag in tags_to_backfill:
            print(f"  - {tag}")
        log.info("Exiting due to dry run. No changes were made.")
        return

    # 3. Confirmation Step
    print(f"\nThis script will generate embeddings for {len(tags_to_backfill)} tags.")
    print("This will make LLM API calls and incur costs.")
    confirm = input("Are you sure you want to continue? (yes/no): ")
    if confirm.lower() != 'yes':
        log.warning("User aborted the script. Exiting.")
        print("Aborted.")
        return

    # 4. Process Concurrently
    log.info("Starting concurrent embedding generation...")
    embedding_model = DEFAULT_EMBEDDING_MODEL
    CONCURRENT_TASKS = 10 # Higher concurrency is fine for embedding
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    
    tasks = [helpers.embed_and_save_tag_worker(tag, embedding_model, semaphore, log) for tag in tags_to_backfill]
    results = await asyncio.gather(*tasks)
    
    successful_embeddings = [res for res in results if res is not None]
    if successful_embeddings:
        await asyncio.to_thread(database.upsert_tags, successful_embeddings)

    log.info("--- Tag Embedding Backfill Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill embeddings for existing tags in the database.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of tags to process in this run."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show which tags would be processed without actually running them."
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.limit, args.dry_run))
