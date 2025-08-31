# newsletter_synthesis_app/utils/llm_processor.py

import litellm
import streamlit as st
import time
import json
from datetime import datetime
from utils import database
from utils.logger import get_logger

logger = get_logger()

# Configure LiteLLM settings
litellm.telemetry = False

def set_llm_keys():
    """
    Sets API keys for LiteLLM from Streamlit secrets.
    This function should be called once at the start of the app.
    """
    if "keys_set" not in st.session_state:
        # This is a simplified approach. For production, you might manage keys more dynamically.
        if st.secrets.get("OPENAI_API_KEY"):
            litellm.openai_key = st.secrets["OPENAI_API_KEY"]
        if st.secrets.get("ANTHROPIC_API_KEY"):
            litellm.anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        if st.secrets.get("GOOGLE_API_KEY"):
            litellm.google_key = st.secrets["GOOGLE_API_KEY"]
        if st.secrets.get("DEEPSEEK_API_KEY"):
            litellm.deepseek_key = st.secrets["DEEPSEEK_API_KEY"]
        st.session_state.keys_set = True
        logger.info("LiteLLM API keys configured.")


def call_llm(model, messages, purpose, metadata={}, json_mode=False):
    """
    Makes a call to an LLM using litellm, with error handling, retries, and logging.
    """
    set_llm_keys()
    
    start_time = time.time()
    log_data = {
        "timestamp": datetime.now(),
        "purpose": purpose,
        "model_used": model,
        "metadata": json.dumps(metadata)
    }

    try:
        kwargs = {"model": model, "messages": messages, "max_retries": 3}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Extract details from response
        content = response.choices[0].message.content
        usage = response.usage
        
        log_data.update({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost": litellm.completion_cost(completion_response=response),
            "duration_ms": duration_ms,
            "status": "success",
            "error_message": None
        })
        
        database.save_llm_call(log_data)
        logger.info(f"LLM call successful for purpose '{purpose}' with model {model}. Duration: {duration_ms}ms.")
        
        if json_mode:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response for purpose '{purpose}': {content}")
                # Log this as an error as well
                log_data['status'] = 'error'
                log_data['error_message'] = 'JSONDecodeError'
                database.save_llm_call(log_data)
                return None
        return content

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_message = f"{type(e).__name__}: {e}"
        logger.error(f"LLM call failed for purpose '{purpose}' with model {model}. Error: {error_message}")
        
        log_data.update({
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0,
            "duration_ms": duration_ms,
            "status": "error",
            "error_message": error_message,
        })
        database.save_llm_call(log_data)
        return None


def generate_embedding(text, model):
    """
    Generates an embedding for a given text using litellm.
    """
    set_llm_keys()
    
    start_time = time.time()
    log_data = {
        "timestamp": datetime.now(),
        "purpose": "generate_embedding",
        "model_used": model,
        "metadata": "{}"
    }

    try:
        response = litellm.embedding(model=model, input=[text])
        duration_ms = int((time.time() - start_time) * 1000)
        
        embedding = response.data[0]['embedding']
        usage = response.usage
        
        log_data.update({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": 0, # Embedding models don't have output tokens in the same way
            "cost": litellm.completion_cost(completion_response=response),
            "duration_ms": duration_ms,
            "status": "success",
            "error_message": None
        })
        database.save_llm_call(log_data)
        logger.info(f"Embedding generation successful with model {model}. Duration: {duration_ms}ms.")
        return embedding

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_message = f"{type(e).__name__}: {e}"
        logger.error(f"Embedding generation failed with model {model}. Error: {error_message}")
        
        log_data.update({
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0,
            "duration_ms": duration_ms,
            "status": "error",
            "error_message": error_message,
        })
        database.save_llm_call(log_data)
        return None