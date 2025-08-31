# newsletter_synthesis_app/utils/summary_generator.py

import os
import json
import time
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

import config
from utils import database, readwise_api, llm_processor
from utils.logger import get_logger

logger = get_logger()


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


def run_summary_generation(start_date, end_date, models, status_callback, progress_callback):
    """
    Main orchestration function for the entire summary generation pipeline.
    """
    total_steps = 8
    current_step = 0

    def update_progress(step, message):
        nonlocal current_step
        current_step = step
        status_callback(message)
        progress_callback(current_step / total_steps)

    try:
        # --- Stage 1: Fetch Articles ---
        update_progress(1, "Fetching articles from Readwise...")
        articles = readwise_api.fetch_readwise_articles(start_date, end_date, status_callback)
        if articles is None or not articles:
            status_callback("No new articles found or an error occurred. Aborting.")
            logger.warning("No articles fetched or API error.")
            progress_callback(1.0)
            return None
        logger.info(f"Fetched {len(articles)} articles.")
        database.set_setting('last_fetch_attempt', datetime.now().isoformat())
        
        # --- FIX: Corrected key from 'updatedAt' to 'updated_at' ---
        latest_date = max(datetime.fromisoformat(a['updated_at'].replace('Z', '+00:00')) for a in articles)
        database.set_setting('last_processed_article_published_date', latest_date.isoformat())

        # --- Stage 2: Individual Article Tagging ---
        all_raw_tags = []
        processed_articles_data = []
        for i, article in enumerate(articles):
            update_progress(2, f"Analyzing article {i+1}/{len(articles)} for tags...")
            
            article_id = article.get('id')
            if not article_id:
                logger.warning("Found an article with no ID. Skipping.")
                continue

            html_path = os.path.join(config.HTML_CACHE_DIR, f"{article_id}.html")
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except FileNotFoundError:
                logger.warning(f"HTML cache not found for article {article_id}. Skipping.")
                continue

            clean_text = get_clean_text(html_content)
            if not clean_text:
                logger.warning(f"Article {article_id} has no text content. Skipping.")
                continue

            prompt = f"""Analyze the following newsletter article content. Identify 5-10 key topics or keywords (tags) that accurately describe its content. Also, extract any notable entities (e.g., company names, person names, technologies) and their relevance. Focus on creating precise, distinct tags and attributes. Provide the output as a JSON object with 'tags' (a list of strings) and 'attributes' (key-value pairs describing entities/relevance).
---
Article Content (first 4000 chars):
{clean_text[:4000]}"""
            messages = [{"role": "user", "content": prompt}]
            
            tag_data = llm_processor.call_llm(models['tagging'], messages, 'extract_tags', metadata={'article_id': article_id}, json_mode=True)

            if tag_data and 'tags' in tag_data:
                all_raw_tags.extend(tag_data['tags'])
                
                # --- FIX: Use .get() for safer access and prioritize 'published_at' ---
                published_date_str = article.get('published_at') or article.get('updated_at')
                published_datetime = datetime.fromisoformat(published_date_str.replace('Z', '+00:00')) if published_date_str else datetime.now()

                newsletter_data = {
                    "id": article_id,
                    "title": article.get('title', 'No Title'),
                    "source": article.get('author', 'Unknown Source'),
                    "original_url": article.get('source_url', ''), # Corrected key
                    "published_date": published_datetime,
                    "processed_date": datetime.now(),
                    "category": article.get('category', 'unknown'),
                    "tags": json.dumps(article.get('tags', {})),
                    "llm_tags": json.dumps(tag_data),
                    "html_content_path": html_path,
                    "embedding": None
                }
                processed_articles_data.append(newsletter_data)
            else:
                logger.warning(f"Failed to extract tags for article {article_id}.")

        if not processed_articles_data:
            status_callback("Could not process any articles for tags. Aborting.")
            progress_callback(1.0)
            return None

        # --- Stage 3: Global Tag Refinement ---
        update_progress(3, "Refining all collected tags...")
        unique_tags = sorted(list(set(all_raw_tags)))
        prompt = f"""Review the following list of tags. Combine semantically similar tags into a single, canonical tag (e.g., 'AI ethics' and 'ethical AI' become 'AI Ethics'). Provide a refined, de-duplicated list of unique canonical tags as a JSON array of strings.
---
Raw Tags:
{json.dumps(unique_tags)}"""
        messages = [{"role": "user", "content": prompt}]
        canonical_tags_json = llm_processor.call_llm(models['refinement'], messages, 'refine_tags', json_mode=True)

        canonical_tags = canonical_tags_json if isinstance(canonical_tags_json, list) and canonical_tags_json else unique_tags
        logger.info(f"Refined {len(unique_tags)} raw tags into {len(canonical_tags)} canonical tags.")

        # --- Stage 4: Tag Embedding & Clustering ---
        update_progress(4, "Generating embeddings for tags and clustering...")
        
        valid_canonical_tags = []
        tag_embeddings = []
        for tag in canonical_tags:
            embedding = llm_processor.generate_embedding(tag, models['embedding'])
            if embedding is not None:
                tag_embeddings.append(embedding)
                valid_canonical_tags.append(tag)

        if len(tag_embeddings) < 2:
            status_callback("Not enough tag data to perform clustering. Aborting.")
            return None
        
        tag_embeddings = np.array(tag_embeddings)
        
        if config.K_CLUSTERS == 'auto':
            max_k = min(config.K_CLUSTERS_MAX, len(tag_embeddings) - 1)
            if max_k < 2:
                num_clusters = 2 if len(tag_embeddings) >= 2 else 1
            else:
                scores = {}
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(tag_embeddings)
                    score = silhouette_score(tag_embeddings, kmeans.labels_)
                    scores[k] = score
                num_clusters = max(scores, key=scores.get) if scores else 2
        else:
            num_clusters = min(config.K_CLUSTERS, len(tag_embeddings))
        
        if num_clusters <= 0:
            status_callback("Cannot form clusters. Aborting.")
            return None
            
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(tag_embeddings)
        clusters = {i: [] for i in range(num_clusters)}
        for i, tag in enumerate(valid_canonical_tags):
            clusters[kmeans.labels_[i]].append(tag)
        logger.info(f"Clustered tags into {num_clusters} themes.")

        # --- Stage 5: Cluster Naming ---
        update_progress(5, "Generating names for identified themes...")
        themed_clusters = {}
        for cluster_id, tags_in_cluster in clusters.items():
            if not tags_in_cluster: continue
            prompt = f"""Based on these keywords, propose a concise, overarching theme name. Provide only the theme name.
---
Keywords:
{', '.join(tags_in_cluster)}"""
            messages = [{"role": "user", "content": prompt}]
            theme_name = llm_processor.call_llm(models['naming'], messages, 'cluster_naming', metadata={'cluster_id': cluster_id})
            if theme_name: themed_clusters[theme_name.strip().strip('"')] = tags_in_cluster
        
        database.update_themes(themed_clusters)
        logger.info(f"Named {len(themed_clusters)} themes.")

        # --- Stage 6: Themed Deep-Dive Summarization ---
        final_summary_content = ""
        final_summary_themes = []

        tag_to_articles = {tag: [art for art in processed_articles_data if tag in json.loads(art['llm_tags']).get('tags', [])] for tag in valid_canonical_tags}

        for i, (theme_name, tags_in_theme) in enumerate(themed_clusters.items()):
            update_progress(6, f"Generating summary for theme '{theme_name}'...")
            relevant_articles = {art['id']: art for tag in tags_in_theme for art in tag_to_articles.get(tag, [])}
            if not relevant_articles: continue

            content_for_summary = "".join([f"--- START (Source: {art['source']}, Title: {art['title']}) ---\n{get_clean_text(open(art['html_content_path'], 'r', encoding='utf-8').read())[:3000]}\n--- END ---\n\n" for art in relevant_articles.values()])
            
            prompt = f"""You are an expert analyst. Summarize the key insights from the following articles related to the theme: '{theme_name}'. Create max 7 bullet points. For each bullet, YOU MUST cite the source in the format '(Newsletter Name: Article Title)'. Synthesize points from multiple articles where possible.
---
Article Content:
{content_for_summary}"""
            messages = [{"role": "user", "content": prompt}]
            theme_summary = llm_processor.call_llm(models['summary'], messages, 'theme_summary', metadata={'theme_name': theme_name})

            if theme_summary:
                final_summary_content += f"## {theme_name}\n\n{theme_summary}\n\n"
                final_summary_themes.append({"theme": theme_name, "summary": theme_summary})
        
        if not final_summary_content:
            status_callback("Failed to generate any themed summaries. Aborting.")
            return None

        # --- Stage 7: Generate Embeddings for Search ---
        update_progress(7, "Generating embeddings for search...")
        for i, article_data in enumerate(processed_articles_data):
            status_callback(f"Embedding newsletter {i+1}/{len(processed_articles_data)}...")
            text_to_embed = f"Title: {article_data['title']}\nTags: {json.dumps(json.loads(article_data['llm_tags']).get('tags'))}"
            embedding = llm_processor.generate_embedding(text_to_embed, models['embedding'])
            
            # --- FIX: Convert list to numpy array ---
            if embedding is not None:
                article_data['embedding'] = np.array(embedding)
            else:
                article_data['embedding'] = None # Ensure it's None if generation failed

        status_callback("Embedding the final summary...")
        summary_text_to_embed = final_summary_content.replace("#", "")
        summary_embedding_list = llm_processor.generate_embedding(summary_text_to_embed, models['embedding'])

        # --- FIX: Convert list to numpy array ---
        summary_embedding = np.array(summary_embedding_list) if summary_embedding_list is not None else None

        # --- Stage 8: Save to Database ---
        update_progress(8, "Saving results to the database...")
        summary_data_to_save = { 
            "generated_date": datetime.now(), 
            "themes": final_summary_themes, 
            "content": final_summary_content, 
            "embedding": summary_embedding }
        
        summary_id = database.save_summary_and_newsletters(summary_data_to_save, processed_articles_data)        

        if summary_id:
            status_callback(f"Successfully generated and saved summary (ID: {summary_id}).")
            logger.info(f"Pipeline complete. Summary ID: {summary_id}")
            progress_callback(1.0)
            return summary_id
        else:
            status_callback("Failed to save the summary to the database.")
            logger.error("Database save operation failed.")
            progress_callback(1.0)
            return None

    except Exception as e:
        logger.error(f"Unexpected error in summary generation pipeline: {e}", exc_info=True)
        status_callback(f"A critical error occurred: {e}")
        progress_callback(1.0)
        return None