"""
Rate Limit Handling for OpenAI API Calls

This module provides robust handling of rate limits for OpenAI API calls,
including exponential backoff, request queueing, and circuit breakers.
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
import httpx

from ..config import settings
from ..monitoring.tracing import trace_method, trace_async_method

logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
AsyncFunc = Callable[..., Any]


class RateLimitExceeded(Exception):
    """Exception raised when rate limits are exceeded beyond retries"""
    pass


class CircuitBreakerOpen(Exception):
    """Exception raised when the circuit breaker is open"""
    pass


class RateLimitHandler:
    """
    Handles rate limiting for OpenAI API calls
    
    Implements:
    - Exponential backoff with jitter for retries
    - Circuit breaker pattern to avoid overwhelming the API
    - Request queuing to manage concurrency
    """
    
    def __init__(
        self,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        jitter_factor: float = 0.1,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        max_concurrent_requests: int = 10
    ):
        """
        Initialize the rate limit handler
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            jitter_factor: Random jitter factor to add to backoff times
            circuit_breaker_threshold: Number of failures to trigger circuit breaker
            circuit_breaker_timeout: Time in seconds to keep circuit breaker open
            max_concurrent_requests: Maximum number of concurrent requests
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.jitter_factor = jitter_factor
        
        # Circuit breaker state
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.failure_count = 0
        self.circuit_open = False
        self.circuit_open_time = 0
        
        # Request queueing
        self.max_concurrent_requests = max_concurrent_requests
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.retried_requests = 0
        self.failed_requests = 0
        
    @trace_method("rate_limit.calculate_backoff")
    def calculate_backoff(self, retry_attempt: int) -> float:
        """
        Calculate backoff time with exponential increase and jitter
        
        Args:
            retry_attempt: Current retry attempt number
            
        Returns:
            Backoff time in seconds
        """
        # Calculate exponential backoff
        backoff = min(
            self.max_backoff,
            self.initial_backoff * (2 ** retry_attempt)
        )
        
        # Add random jitter
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * backoff
        backoff = backoff + jitter
        
        return backoff
    
    @trace_method("rate_limit.check_circuit_breaker")
    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open
        
        Returns:
            True if circuit is open (requests should be blocked)
        """
        # If circuit is open, check if it's time to close it
        if self.circuit_open:
            elapsed = time.time() - self.circuit_open_time
            if elapsed >= self.circuit_breaker_timeout:
                logger.info("Circuit breaker closing")
                self.circuit_open = False
                self.failure_count = 0
                return False
            return True
            
        return False
    
    @trace_method("rate_limit.record_success")
    def record_success(self) -> None:
        """Record a successful API call"""
        self.successful_requests += 1
        # Reset failure count on success
        self.failure_count = 0
    
    @trace_method("rate_limit.record_failure")
    def record_failure(self, rate_limited: bool = False) -> None:
        """
        Record a failed API call
        
        Args:
            rate_limited: Whether the failure was due to rate limiting
        """
        self.failed_requests += 1
        
        # Update circuit breaker state for rate limit failures
        if rate_limited:
            self.failure_count += 1
            
            # Check if circuit breaker should open
            if self.failure_count >= self.circuit_breaker_threshold:
                logger.warning("Circuit breaker opening due to rate limit failures")
                self.circuit_open = True
                self.circuit_open_time = time.time()
    
    def retry_decorator(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to add retry logic to synchronous functions
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function with retry logic
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.total_requests += 1
            
            # Check if circuit breaker is open
            if self.check_circuit_breaker():
                self.failed_requests += 1
                raise CircuitBreakerOpen("Circuit breaker is open, request rejected")
            
            retry_attempt = 0
            while retry_attempt <= self.max_retries:
                try:
                    result = func(*args, **kwargs)
                    self.record_success()
                    return result
                    
                except Exception as e:
                    # Check if this is a rate limit error
                    is_rate_limit = (
                        isinstance(e, httpx.HTTPStatusError) and 
                        getattr(e, "status_code", 0) == 429
                    )
                    
                    if not is_rate_limit or retry_attempt == self.max_retries:
                        # Record failure and re-raise if not rate limit or max retries exceeded
                        self.record_failure(is_rate_limit)
                        raise
                    
                    # Calculate backoff time and retry
                    retry_attempt += 1
                    self.retried_requests += 1
                    backoff = self.calculate_backoff(retry_attempt)
                    
                    logger.warning(
                        f"Rate limit hit, retrying in {backoff:.2f}s (attempt {retry_attempt}/{self.max_retries})"
                    )
                    time.sleep(backoff)
            
            # This shouldn't be reached, but just in case
            raise RateLimitExceeded("Max retries exceeded")
            
        return wrapper
    
    def async_retry_decorator(self, func: AsyncFunc) -> AsyncFunc:
        """
        Decorator to add retry logic to async functions
        
        Args:
            func: Async function to decorate
            
        Returns:
            Decorated async function with retry logic
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.total_requests += 1
            
            # Check if circuit breaker is open
            if self.check_circuit_breaker():
                self.failed_requests += 1
                raise CircuitBreakerOpen("Circuit breaker is open, request rejected")
            
            # Acquire semaphore for concurrency control
            async with self.request_semaphore:
                retry_attempt = 0
                while retry_attempt <= self.max_retries:
                    try:
                        result = await func(*args, **kwargs)
                        self.record_success()
                        return result
                        
                    except Exception as e:
                        # Check if this is a rate limit error
                        is_rate_limit = (
                            isinstance(e, httpx.HTTPStatusError) and 
                            getattr(e, "status_code", 0) == 429
                        )
                        
                        if not is_rate_limit or retry_attempt == self.max_retries:
                            # Record failure and re-raise if not rate limit or max retries exceeded
                            self.record_failure(is_rate_limit)
                            raise
                        
                        # Calculate backoff time and retry
                        retry_attempt += 1
                        self.retried_requests += 1
                        backoff = self.calculate_backoff(retry_attempt)
                        
                        logger.warning(
                            f"Rate limit hit, retrying in {backoff:.2f}s (attempt {retry_attempt}/{self.max_retries})"
                        )
                        await asyncio.sleep(backoff)
                
                # This shouldn't be reached, but just in case
                raise RateLimitExceeded("Max retries exceeded")
                
        return cast(AsyncFunc, wrapper)
    
    @trace_method("rate_limit.get_stats")
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limit handler statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "retried_requests": self.retried_requests,
            "failed_requests": self.failed_requests,
            "circuit_breaker_open": self.circuit_open,
            "failure_count": self.failure_count,
            "max_concurrent_requests": self.max_concurrent_requests,
        }


# Create singleton instance with default settings
rate_limit_handler = RateLimitHandler()


# Function decorators for easy use
def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add retry logic to synchronous functions"""
    return rate_limit_handler.retry_decorator(func)


def with_async_retry(func: AsyncFunc) -> AsyncFunc:
    """Decorator to add retry logic to async functions"""
    return rate_limit_handler.async_retry_decorator(func)
