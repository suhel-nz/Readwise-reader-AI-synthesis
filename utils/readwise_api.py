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
    try:
        api_key = st.secrets["READWISE_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("Readwise API key not found. Please set it in .streamlit/secrets.toml")
        logger.error("READWISE_API_KEY not found in Streamlit secrets.")
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
        else: # This else belongs to the for loop, runs if loop completes without break
            st.error("Failed to fetch from Readwise after multiple retries.")
            return None
        # --- End of new logic ---

        articles_on_page = data.get("results", [])
        all_articles.extend(articles_on_page)
        logger.info(f"Fetched {len(articles_on_page)} articles on page {page_count}. Total: {len(all_articles)}.")

        page_cursor = data.get("nextPageCursor")
        if not page_cursor:
            break
        page_count += 1

    status_callback(f"Successfully fetched a total of {len(all_articles)} articles from Readwise.")
    logger.info(f"Total articles fetched: {len(all_articles)}")
    
    # Save HTML content to local cache
    for article in all_articles:
        html_content = article.get('html_content', '')
        if html_content:
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