# newsletter_synthesis_app/utils/logger.py

import logging
import os
from datetime import datetime
import config # Use aboslute import

# Ensure the log directory exists
os.makedirs(config.LOG_DIR, exist_ok=True)

# Store the log file path globally for a single run
log_file_path = None

def setup_logger(run_timestamp):
    """
    Sets up a new logger for a specific run, creating a timestamped file.
    """
    global log_file_path
    log_file_name = f"{run_timestamp.strftime('%Y%m%d_%H%M%S')}_run.log"
    log_file_path = os.path.join(config.LOG_DIR, log_file_name)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also log to console
        ]
    )
    logging.info(f"Logger initialized. Log file: {log_file_path}")
    return logging.getLogger(__name__)

def get_logger():
    """
    Returns the currently configured logger. If not set up, initializes a default one.
    """
    if not logging.root.hasHandlers():
        # Fallback for cases where setup_logger hasn't been called
        setup_logger(datetime.now())
    return logging.getLogger(__name__)

def get_log_file_path():
    """
    Returns the path to the current run's log file.
    """
    return log_file_path