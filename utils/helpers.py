# utils/helpers.py

# newsletter_synthesis_app/utils/summary_generator.py

from bs4 import BeautifulSoup
from utils.logger import get_logger
import json
import pandas as pd
import asyncio
from datetime import datetime
from . import llm_processor
from .casting import to_numpy_array

logger = get_logger()

async def embed_and_save_tag_worker(tag: str, model: str, semaphore: asyncio.Semaphore, log):
    """
    An async worker that generates an embedding for a single tag.
    This is a shared utility for ingestion and backfill scripts.
    """
    async with semaphore:
        log.info(f"Generating embedding for tag: '{tag}'")
        embedding = await asyncio.to_thread(llm_processor.generate_embedding, tag, model)
        if embedding:
            return {"tag": tag, "embedding": to_numpy_array(embedding), "last_seen": datetime.now()}
        return None

def extract_tags_from_dataframe(df: pd.DataFrame) -> list[str]:
    """Parses the 'llm_tags' JSON column and returns a flat list of all tags."""
    all_tags = []
    for tags_json in df['llm_tags'].dropna():
        try:
            tags = json.loads(tags_json).get('tags', [])
            if isinstance(tags, list): all_tags.extend(tags)
        except (json.JSONDecodeError, TypeError): continue
    return all_tags


def get_clean_text(html_content):
    """Extracts clean text from HTML content."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return ""
