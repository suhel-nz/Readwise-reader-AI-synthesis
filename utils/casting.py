# newsletter_synthesis_app/utils/casting.py

from datetime import datetime
import numpy as np

def to_datetime(value) -> datetime | None:
    """Safely converts a value (string or datetime) to a datetime object."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None
    return None

def to_numpy_array(value) -> np.ndarray | None:
    """Safely converts a value (list or ndarray) to a numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value)
    return None