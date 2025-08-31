# newsletter_synthesis_app/utils/llm_processor.py

import litellm
import streamlit as st
import time
import json
from datetime import datetime
from utils import database
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
import config

logger = get_logger()

# Configure LiteLLM settings
litellm.telemetry = False

# NEW: Dictionary to hold rate limiter instances for each model
llm_limiters = {}

def get_model_limiter(model_name: str) -> RateLimiter:
    """Gets or creates a RateLimiter for a specific model."""
    if model_name not in llm_limiters:
        limits = config.LLM_PROVIDER_LIMITS.get(model_name, config.LLM_PROVIDER_LIMITS['default'])
        rpm = limits.get('rpm')
        if rpm:
            llm_limiters[model_name] = RateLimiter(rpm=rpm)
        else:
            # If no RPM limit is defined, return a dummy limiter that does nothing
            class DummyLimiter:
                def wait(self): pass
            llm_limiters[model_name] = DummyLimiter()
    return llm_limiters[model_name]

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

def _log_llm_call(log_data):
    """Helper to save LLM call log and update daily stats."""
    database.save_llm_call(log_data)
    if log_data['status'] == 'success':
        total_tokens = log_data.get('input_tokens', 0) + log_data.get('output_tokens', 0)
        database.update_daily_usage(
            model=log_data['model_used'],
            calls_to_add=1,
            tokens_to_add=total_tokens
        )

def call_llm(model, messages, purpose, metadata={}, json_mode=False):
    """
    Makes a call to an LLM using litellm, with error handling, retries, and logging.
    """
    set_llm_keys()

    # --- NEW: Check RPD limit before making a call ---
    limits = config.LLM_PROVIDER_LIMITS.get(model, config.LLM_PROVIDER_LIMITS['default'])
    rpd_limit = limits.get('rpd')
    if rpd_limit is not None:
        current_calls, _ = database.get_daily_usage(model)
        if current_calls >= rpd_limit:
            error_msg = f"Daily request limit ({rpd_limit}) for model {model} has been reached."
            logger.error(error_msg)
            st.error(error_msg)
            return None # Stop execution if RPD limit is hit
    # --- End of RPD check ---
    
    start_time = time.time()
    log_data = {
        "timestamp": datetime.now(),
        "purpose": purpose,
        "model_used": model,
        "metadata": json.dumps(metadata)
    }

    limiter = get_model_limiter(model)
    max_retries = 5
    backoff_factor = 2

    for attempt in range(max_retries):
        limiter.wait() # Enforce RPM limit

        try:
            kwargs = {"model": model, "messages": messages, "max_retries": 0} # Disable litellm's retry
            if json_mode: kwargs["response_format"] = {"type": "json_object"}

            response = litellm.completion(**kwargs)
            
            duration_ms = int((time.time() - start_time) * 1000)
            content = response.choices[0].message.content
            usage = response.usage
            cost = litellm.completion_cost(completion_response=response) or 0.0
            
            log_data.update({
                "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                "cost": cost, "duration_ms": duration_ms, "status": "success", "error_message": None
            })
            _log_llm_call(log_data)
            logger.info(f"LLM call successful for '{purpose}' with {model}. Duration: {duration_ms}ms.")
            
            if json_mode:
                try: return json.loads(content)
                except json.JSONDecodeError:
                    log_data.update({"status": "error", "error_message": "JSONDecodeError"})
                    _log_llm_call(log_data) # Log failure
                    return None
            return content

        except litellm.RateLimitError as e:
            wait_time = backoff_factor * (2 ** attempt)
            logger.warning(f"LLM rate limit for {model}. Retrying in {wait_time}s... Error: {e}")
            time.sleep(wait_time)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"LLM call failed for '{purpose}' with {model}. Error: {error_message}")
            log_data.update({ "input_tokens": 0, "output_tokens": 0, "cost": 0, "duration_ms": duration_ms, "status": "error", "error_message": error_message, })
            _log_llm_call(log_data)
            return None # Non-retriable error
            
    # If all retries fail
    logger.error(f"LLM call for '{purpose}' failed after {max_retries} retries.")
    return None


def generate_embedding(text, model):
    """
    Generates an embedding for a given text using litellm.
    """
    set_llm_keys()

    limits = config.LLM_PROVIDER_LIMITS.get(model, config.LLM_PROVIDER_LIMITS.get('default_embedding', {}))
    rpd_limit = limits.get('rpd')
    if rpd_limit is not None:
        current_calls, _ = database.get_daily_usage(model)
        if current_calls >= rpd_limit:
            error_msg = f"Daily request limit ({rpd_limit}) for model {model} has been reached."
            logger.error(error_msg)
            st.error(error_msg)
            return None
    
    start_time = time.time()
    log_data = {
        "timestamp": datetime.now(),
        "purpose": "generate_embedding",
        "model_used": model,
        "metadata": "{}"
    }

    limiter = get_model_limiter(model)
    max_retries = 5
    backoff_factor = 2

    for attempt in range(max_retries):
        limiter.wait()
        try:
            logger.info(f"Generating embedding with {model}, for {text[:30]}...")
            response = litellm.embedding(model=model, input=[text])
            duration_ms = int((time.time() - start_time) * 1000)
            
            embedding = response.data[0]['embedding']
            #logger.info(f"Generated embedding with {model}: {embedding[:10]}")
            usage = response.usage

            #logger.info(f"Attempting to get cost.")
            cost = 0.0 #litellm.completion_cost(completion_response=response) or 0.0
            
            #logger.info(f"Attempting to log_update.")
            log_data.update({
                "input_tokens": usage.prompt_tokens, "output_tokens": 0, "cost": cost,
                "duration_ms": duration_ms, "status": "success", "error_message": None
            })
            _log_llm_call(log_data)
            logger.info(f"Embedding successful with {model}. Duration: {duration_ms}ms.")
            return embedding
        except litellm.RateLimitError as e:
            wait_time = backoff_factor * (2 ** attempt)
            logger.warning(f"Embedding rate limit for {model}. Retrying in {wait_time}s... Error: {e}")
            time.sleep(wait_time)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Embedding failed with {model}. Error: {error_message}")
            log_data.update({ "input_tokens": 0, "output_tokens": 0, "cost": 0, "duration_ms": duration_ms, "status": "error", "error_message": error_message, })
            _log_llm_call(log_data)
            return None
    
    logger.error(f"Embedding generation with {model} failed after {max_retries} retries.")
    return None