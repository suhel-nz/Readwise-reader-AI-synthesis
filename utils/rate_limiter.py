# newsletter_synthesis_app/utils/rate_limiter.py

import time
from collections import deque
from utils.logger import get_logger

logger = get_logger()

class RateLimiter:
    """
    A simple client-side rate limiter to ensure a maximum number of requests per minute.
    """
    def __init__(self, rpm: int):
        """
        Initializes the RateLimiter.
        Args:
            rpm (int): The maximum number of requests per minute.
        """
        if rpm <= 0:
            raise ValueError("RPM must be a positive integer.")
        self.rpm = rpm
        self.interval = 60.0 / self.rpm
        self.call_timestamps = deque()

    def wait(self):
        """
        Blocks until a new request can be made without exceeding the RPM limit.
        """
        # Prune timestamps older than 60 seconds from the left of the deque
        now = time.monotonic()
        while self.call_timestamps and self.call_timestamps[0] <= now - 60:
            self.call_timestamps.popleft()

        # If we have made `rpm` requests in the last minute, wait.
        if len(self.call_timestamps) >= self.rpm:
            oldest_call_time = self.call_timestamps[0]
            time_to_wait = oldest_call_time - (now - 60)
            if time_to_wait > 0:
                logger.info(f"RPM limit of {self.rpm} reached. Waiting for {time_to_wait:.2f} seconds.")
                time.sleep(time_to_wait)
        
        # Record the current call time
        self.call_timestamps.append(time.monotonic())