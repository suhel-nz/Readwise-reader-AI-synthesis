#### **New/Refactored File: `utils/processing_logic.py`**
# This replaces `summary_generator.py`. It is now clean, modular, and uses the new prompts.
# newsletter_synthesis_app/utils/processing_logic.py

import json
import os
import numpy as np 
from datetime import datetime, timezone # MODIFIED
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
from utils import database, llm_processor, prompts
from utils.logger import get_logger
from utils.casting import to_numpy_array
from utils.helpers import get_clean_text
import config

logger = get_logger()

def _count_bullets_missing_citations(text: str) -> tuple[int, int]:
    """Counts bullets that lack inline citations of the form [..](..). Returns (missing, total)."""
    try:
        import re
        lines = text.splitlines()
        bullet_re = re.compile(r"^\s*(?:[-*\u2022])\s+")
        citation_re = re.compile(r"\[[^\]]+\]\([^\)]+\)")
        total = 0
        missing = 0
        for ln in lines:
            if bullet_re.search(ln):
                total += 1
                if not citation_re.search(ln):
                    missing += 1
        return missing, total
    except Exception:
        return 0, 0

def process_single_article(article: dict, models: dict) -> dict | None:
    """
    Performs the full 'Stage 1' ingestion processing for a single article.
    Returns a dictionary of all extracted metadata.
    """
    # ... logic to get clean text ...
    article_id = article.get('id')
    if not article_id:
        logger.warning("Found an article with no ID. Skipping.")

    logger.info(f"Analyzing article {article_id} for tags...")

    html_path = os.path.join(config.HTML_CACHE_DIR, f"{article_id}.html")
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        logger.warning(f"HTML cache not found for article {article_id}. Skipping.")
        return None # CRITICAL: Stop execution if the file is missing
    
    clean_text = get_clean_text(html_content)
    if not clean_text:
        logger.warning(f"Article {article_id} has no text content. Skipping.")

    prompt_template = prompts.get_prompt_template("stage1_process_article")
    prompt = prompt_template.format(article_text=clean_text[:200000]) # Limit context size
    
    llm_response = llm_processor.call_llm(
        models['tagging'], [{"role": "user", "content": prompt}],
        'stage1_process_article', metadata={'article_id': article['id']}, json_mode=True
    )

    # Generate embedding
    text_to_embed = f"Title: {article.get('title')}\nSummary: {llm_response.get('summary')}\nTags: {json.dumps(llm_response.get('tags'))}"
    embedding = llm_processor.generate_embedding(text_to_embed, models['embedding'])

    # This dictionary should contain all fields expected by database.upsert_newsletter
    return {
        "id": article.get('id'),
        "title": article.get('title', 'No Title'),
        "source": article.get('author', 'Unknown Source'),
        "original_url": article.get('url'), # Reader URL
        "published_date": article.get('published_at') or article.get('updated_at'),
        "category": article.get('category', 'unknown'),
        "tags": json.dumps(article.get('tags', {})), 
        "processed_date": datetime.now(timezone.utc), # MODIFIED: Use aware datetime
        "readwise_summary": article.get('summary', ''),
        "llm_summary": llm_response.get('summary', ''),
        "llm_tags": json.dumps(llm_response), # Store the whole rich object
        "html_content_path": html_path, # CRITICAL: Ensure path is preserved on update
        "embedding": to_numpy_array(embedding),
    }


def synthesize_insights(article_ids: list[str], models: dict, status_callback: callable) -> int | None:
    """
    Performs the 'Stage 2' on-demand synthesis for a given list of articles.
    Returns the ID of the newly created summary.
    """
    if not article_ids:
        logger.error("synthesize_insights called with no article_ids.")
        return None

    try:
        # --- 1. Fetch article data from DB for the given IDs ---
        status_callback("Fetching data for selected articles...")
        conn = database.get_db_connection()
        if not conn: return None
        
        placeholders = ','.join('?' for _ in article_ids)
        query = f"SELECT id, title, source, original_url, llm_summary, llm_tags FROM newsletters WHERE id IN ({placeholders})"
        articles = conn.execute(query, article_ids).fetchall()
        conn.close()
        
        logger.info(f"Starting synthesis for {len(articles)} articles.")

        # --- 2. Extract tags and perform K-Means clustering ---
        status_callback("Analyzing tags and identifying themes...")
        all_tags = []
        for article in articles:
            try:
                tags = json.loads(article['llm_tags']).get('tags', [])
                if isinstance(tags, list):
                    all_tags.extend(tags)
            except (json.JSONDecodeError, TypeError):
                continue
        
        unique_tags = sorted(list(set(all_tags)))
        if len(unique_tags) < 2:
            logger.warning("Not enough unique tags to perform clustering.")
            # Fallback: Treat all articles as a single theme
            clusters = {0: unique_tags}
        else:
            # --- REFACTORED: Fetch pre-computed embeddings instead of generating them ---
            status_callback("Fetching pre-computed tag embeddings...")
            tag_embeddings_map = database.get_tags_with_embeddings(unique_tags)
            
            valid_tags = list(tag_embeddings_map.keys())
            tag_embeddings = list(tag_embeddings_map.values())
            
            if len(valid_tags) < 2:
                logger.warning("Not enough tags with embeddings to perform clustering.")
                clusters = {0: valid_tags}
            else:
                tag_embeddings_np = to_numpy_array(tag_embeddings)
                
                # Determine optimal K
                max_k = min(config.K_CLUSTERS_MAX, len(valid_tags) - 1)
                num_clusters = 1 # Default to 1 cluster if auto-detection fails
                if config.K_CLUSTERS == 'auto' and max_k >= 2:
                    scores = {k: silhouette_score(tag_embeddings_np, KMeans(n_clusters=k, random_state=42, n_init='auto').fit(tag_embeddings_np).labels_) for k in range(2, max_k + 1)}
                    num_clusters = max(scores, key=scores.get) if scores else 2
                    logger.info(f"Auto-detected optimal number of clusters: {num_clusters}")
                elif isinstance(config.K_CLUSTERS, int):
                    num_clusters = min(config.K_CLUSTERS, len(valid_tags))

                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(tag_embeddings_np)
                clusters = {i: [] for i in range(num_clusters)}
                for i, tag in enumerate(valid_tags):
                    clusters[kmeans.labels_[i]].append(tag)

        # --- 3. Call LLM to name the clusters (themes) ---
        status_callback("Generating names for identified themes...")
        prompt_template_naming = prompts.get_prompt_template("synthesis_cluster_naming")
        themed_clusters = {}
        for cluster_id, tags_in_cluster in clusters.items():
            if not tags_in_cluster: continue
            
            prompt = prompt_template_naming.format(tag_list=', '.join(tags_in_cluster))
            theme_name = llm_processor.call_llm(models['naming'], [{"role": "user", "content": prompt}], 'synthesis_cluster_naming')
            
            if theme_name:
                # Clean up theme name
                clean_name = theme_name.strip().strip('"')
                themed_clusters[clean_name] = tags_in_cluster
                logger.info(f"Cluster {cluster_id} named as '{clean_name}' with tags: {tags_in_cluster}")
        
        database.update_themes(themed_clusters)
        logger.info(f"Identified and named {len(themed_clusters)} themes for this synthesis.")

        # --- 4. For each theme, construct context and call LLM for deep-dive summary ---
        status_callback("Generating deep-dive summaries for each theme...")
        draft_summary_content = ""
        final_summary_themes = []
        
        prompt_template_deepdive = prompts.get_prompt_template("synthesis_deep_dive")
        
        for theme_name, tags_in_theme in themed_clusters.items():
            # Find all articles relevant to this theme
            relevant_articles = []
            for article in articles:
                article_tags = set(json.loads(article['llm_tags']).get('tags', []))
                if not article_tags.isdisjoint(tags_in_theme):
                    relevant_articles.append(article)
            
            if not relevant_articles: continue
            
            # Construct the context for the LLM prompt
            article_context = ""
            for art in relevant_articles:
                article_context += f"Title: {art['title']}\n"
                article_context += f"Readwise URL: {art['original_url']}\n"
                article_context += f"Summary: {art['llm_summary']}\n---\n"
            
            prompt = prompt_template_deepdive.format(theme_name=theme_name, article_context=article_context)
            theme_summary = llm_processor.call_llm(models['summary'], [{"role": "user", "content": prompt}], 'synthesis_deep_dive')
            
            if theme_summary:
                draft_summary_content += f"## {theme_name}\n\n{theme_summary}\n\n"
                final_summary_themes.append({"theme": theme_name, "summary": theme_summary})

        if not draft_summary_content:
            logger.error("Failed to generate any themed summaries for this set of articles.")
            status_callback("Error: Could not generate summaries.")
            return None

        # --- 5. Combine summaries and call final critique LLM ---
        status_callback("Performing final critique and refinement...")
        prompt_template_critique = prompts.get_prompt_template("synthesis_final_critique")
        prompt = prompt_template_critique.format(draft_summary=draft_summary_content)
        
        final_summary_content = llm_processor.call_llm(models['critique'], [{"role": "user", "content": prompt}], 'synthesis_final_critique')
        
        if not final_summary_content:
            logger.warning("Final critique step failed. Using unrefined summary.")
            final_summary_content = draft_summary_content

        # --- 6. Generate embedding and save the final summary to the DB ---
        status_callback("Saving final insights pack...")

        summary_embedding_list = llm_processor.generate_embedding(final_summary_content.replace("#", ""), models['embedding'])
        summary_embedding = to_numpy_array(summary_embedding_list)

        # Minimal validation for inline citations on bullets
        missing_citations, total_bullets = _count_bullets_missing_citations(final_summary_content)

        gen_context = {
            "article_ids": article_ids,
            "citation_issues": missing_citations,
            "bullets_count": total_bullets,
        }

        summary_data_to_save = {
            "generated_date": datetime.now(timezone.utc), # MODIFIED: Use aware datetime
            "themes": final_summary_themes,
            "content": final_summary_content,
            "draft_content": draft_summary_content,
            "newsletter_count": len(articles),
            "embedding": summary_embedding,
            "generation_context_json": json.dumps(gen_context) # Store which articles were used and validation
        }
        
        summary_id = database.save_summary_only(summary_data_to_save)
        
        if summary_id:
            logger.info(f"Successfully created and saved new insights pack with ID: {summary_id}")
            return summary_id
        else:
            logger.error("Failed to save the final summary to the database.")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during synthesis: {e}", exc_info=True)
        status_callback(f"An error occurred: {e}")
        return None
