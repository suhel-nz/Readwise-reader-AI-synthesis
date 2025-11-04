# newsletter_synthesis_app/utils/prompts.py

from utils import database
from utils.logger import get_logger

logger = get_logger()

# Define the initial, default prompts here
INITIAL_PROMPTS = {
    "stage1_process_article": {
        "description": "The first pass on a new article. Extracts tags, attributes (with grounding), and a concise summary.",
        "template": """Analyze the following article content. Your goal is to extract structured metadata.
Provide the output as a single, valid JSON object with three keys: "tags", "summary", and "attributes".

1.  "tags": A list of 5-10 precise keywords or topics.
2.  "summary": A concise, one-paragraph, factual summary of the article.
3.  "attributes": A list of JSON objects, where each object has three keys: "attribute" (the extracted entity name), "relevance" (why it's important), and "grounding" (the exact sentence from the text that supports the extraction).

Example for "attributes":
[
  {{
    "attribute": "Gemini 1.5 Pro",
    "relevance": "The main subject of the technical review, noted for its large context window.",
    "grounding": "The recent release of Gemini 1.5 Pro has generated significant buzz due to its claimed 1 million token context window."
  }}
]

---
ARTICLE CONTENT:
{article_text}"""
    },
    "synthesis_suggest_tags": {
        "description": "Suggests related tags to help a user refine their article selection for synthesis.",
        "template": """You are a research assistant. Based on the user's selected tags, your task is to suggest other relevant tags from a provided list that could help them broaden or refine their search.

User's selected tags: {selected_tags}
List of available tags to choose from: {options}

Return a JSON array of objects, with each object containing a "tag" and a "reason". Suggest a maximum of 3 tags.
Example: [{{"tag": "Cognitive Load", "reason": "Often discussed in the context of Attention Economy and its effects on focus."}}]"""
    },
    "synthesis_cluster_naming": {
        "description": "Takes a list of tags from a cluster and gives it a concise, human-readable theme name.",
        "template": """Based on the following group of keywords/tags, propose a concise, overarching theme name that accurately captures the essence of these topics.
Return ONLY the theme name as a single string, with no extra text or quotes.
---
Cluster Tags:
{tag_list}"""
    },
    "synthesis_deep_dive": {
        "description": "Takes a theme and all related articles to generate a synthesized, multi-source summary.",
        "template": """You are an expert analyst. Synthesize key insights from the following articles for the theme below.
Create at most 7 concise bullet points.
Citation rule: For every bullet, place one or more citations immediately at the end of that bullet (inline), not in a separate list.
Use this exact format for each citation: [Source: Title](readwise_url). If multiple articles support a point, list all citations at the end of the bullet.

Theme: {theme_name}
---
ARTICLES (Title, Readwise URL, and a summary of each):
{article_context}"""
    },
    "synthesis_final_critique": {
        "description": "The final editing pass. Reviews the complete generated summary to merge duplicates and improve clarity.",
        "template": """You are a senior editor. Refine the draft below for clarity and cohesion while preserving substance.
Editing guidance:
1. Merge only truly duplicative bullets; preserve distinct facts and nuances.
2. Remove fluff but do not drop meaningful information.
3. Improve flow and readability; rephrase when helpful.
4. Preserve inline citations exactly next to the statements they support. Do not move citations to a list. If merging/rephrasing, carry over all relevant citations for that bullet.

Return ONLY the final, refined summary in Markdown format. Do not add commentary.
---
DRAFT SUMMARY:
{draft_summary}"""
    }
}

def initialize_prompts():
    """Seeds the database with the initial prompts defined above."""
    database.seed_initial_prompts(INITIAL_PROMPTS)
    logger.info(f"prompts - initialized with default values.")

def get_prompt_template(name: str) -> str:
    """Gets a prompt template from the DB, falling back to the initial dictionary if needed."""
    prompt = database.get_prompt(name)
    if prompt and prompt.get('template'):
        return prompt['template']
    logger.warning(f"Prompt '{name}' not found in DB, using initial default.")
    return INITIAL_PROMPTS.get(name, {}).get('template', "")

def ensure_prompt_updates_v1():
    """One-time prompt updates to enforce inline citations and softer critique.
    Uses app_settings key to avoid overwriting user edits repeatedly.
    """
    try:
        applied = database.get_setting('prompt_update_v1_applied')
        if str(applied).lower() == 'true':
            return
        # Force update two prompts to new defaults
        for key in ("synthesis_deep_dive", "synthesis_final_critique"):
            tmpl = INITIAL_PROMPTS.get(key, {}).get('template', '')
            if tmpl:
                database.update_prompt(key, tmpl)
        database.set_setting('prompt_update_v1_applied', True)
        logger.info("Prompt updates v1 applied to DB.")
    except Exception as e:
        logger.warning(f"Failed to apply prompt updates v1: {e}")
