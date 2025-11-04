# newsletter_synthesis_app/utils/readwise_api.py

import requests
import os
import streamlit as st
from datetime import datetime
import json
import config
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter # NEW


logger = get_logger()
# NEW: Instantiate a rate limiter for the Readwise API
readwise_limiter = RateLimiter(rpm=config.READWISE_RPM)

def fetch_readwise_articles(updated_after, updated_before, status_callback):
    """
    Fetches articles from the Readwise Reader API with pagination.

    Args:
        updated_after (datetime): The start date to fetch articles from.
        updated_before (datetime): The end date to fetch articles to.
        status_callback (function): A function to call with progress updates.

    Returns:
        list: A list of article dictionaries, or None if an error occurs.
    """
    # This function now has two modes of getting the API key.
    # It tries Streamlit secrets first, for when it's called from the UI (less common now).
    # It falls back to environment variables for when it's called from run_ingestion.py.
    api_key = st.secrets.get("READWISE_API_KEY") or os.environ.get("READWISE_API_KEY")
    if not api_key:
        error_msg = "Readwise API key not found. Please set it in .streamlit/secrets.toml or as an environment variable."
        logger.error(error_msg)
        # Avoid st.error if not in a Streamlit context
        if 'streamlit' in globals():
            st.error(error_msg)
        else:
            print(f"ERROR: {error_msg}")
        return None

    headers = {"Authorization": f"Token {api_key}"}
    params = {
        "category__in": ",".join(config.READWISE_CATEGORIES),
        "withHtmlContent": "true",
        "updatedAfter": updated_after.isoformat(),
        "updatedBefore": updated_before.isoformat(),
    }
    
    all_articles = []
    page_cursor = None
    page_count = 1

    while True:
        if page_cursor:
            params["pageCursor"] = page_cursor

        status_callback(f"Fetching page {page_count} from Readwise API...")
        # --- NEW: Rate limiting and retry logic ---
        max_retries = 5
        backoff_factor = 2
        data = None # Initialize data to None

        for attempt in range(max_retries):
            readwise_limiter.wait() # Wait before making the call
            
            logger.info(f"Requesting Readwise API (Attempt {attempt + 1}). Params: {params}")
            try:
                response = requests.get(config.READWISE_API_BASE_URL, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()                
                # Success, break the retry loop
                break 

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429: # Rate limit error
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Readwise API rate limit hit. Retrying in {wait_time} seconds...")
                    status_callback(f"Readwise rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP Error fetching from Readwise: {e.response.status_code} - {e.response.text}")
                    st.error(f"Failed to fetch from Readwise: {e.response.status_code} - Check logs.")
                    return None # Non-retriable HTTP error
            except requests.exceptions.RequestException as e:
                logger.error(f"Request Error fetching from Readwise: {e}")
                st.error(f"A network error occurred while contacting Readwise.")
                return None # Network error
        
        if data is None: # If all retries failed
            logger.error("Failed to fetch from Readwise after multiple retries.")
            return None

        articles_on_page = data.get("results", [])
        logger.info(f"Fetched {len(articles_on_page)} articles from page {page_count}.")        
        
        # --- CORRECTED LOGIC: Filter before extending ---
        original_count = len(articles_on_page)
        filtered_articles = [doc for doc in articles_on_page if doc.get("parent_id") is None]
        filtered_count = len(filtered_articles)
        
        if original_count != filtered_count:
            logger.info(f"Filtered out {original_count - filtered_count} items (highlights/notes) with a parent_id.")
        
        # --- CORRECTED LOGIC: Extend the list only ONCE with filtered results ---
        all_articles.extend(filtered_articles)

        page_cursor = data.get("nextPageCursor")
        if not page_cursor:
            break
        page_count += 1

    status_callback(f"Successfully fetched a total of {len(all_articles)} articles from Readwise.")
    logger.info(f"Total articles fetched: {len(all_articles)}")
    
    # Save HTML content to local cache
    for article in all_articles:
        html_content = article.get('html_content', '')
        if html_content and 'id' in article:
            file_path = os.path.join(config.HTML_CACHE_DIR, f"{article['id']}.html")
            logger.info(f"Start caching HTML for article {article['id']}.")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                # We no longer need to keep the large HTML in memory
                del article['html_content']
            except Exception as e:
                logger.error(f"Could not cache HTML for article {article['id']}: {e}")
    
    return all_articles