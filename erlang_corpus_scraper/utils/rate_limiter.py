"""
Rate limiter utilities for API requests.
Handles GitHub API rate limiting and general request throttling.
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock
from datetime import datetime, timedelta

@dataclass
class RateLimitInfo:
    """Rate limit information from API headers."""
    limit: int
    remaining: int
    reset_time: int
    used: int

class RateLimiter:
    """Thread-safe rate limiter for API requests."""
    
    def __init__(self, requests_per_hour: int = 5000, buffer_percentage: float = 0.1):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_hour: Maximum requests allowed per hour
            buffer_percentage: Safety buffer (e.g., 0.1 = use only 90% of limit)
        """
        self.requests_per_hour = requests_per_hour
        self.buffer_percentage = buffer_percentage
        self.effective_limit = int(requests_per_hour * (1 - buffer_percentage))
        
        self.requests_made = 0
        self.window_start = time.time()
        self.lock = Lock()
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Rate limiter initialized: {self.effective_limit} requests/hour "
                        f"(original: {requests_per_hour}, buffer: {buffer_percentage:.1%})")
    
    def wait_if_needed(self) -> bool:
        """
        Check if we need to wait due to rate limiting.
        
        Returns:
            True if request can proceed, False if permanently rate limited
        """
        with self.lock:
            current_time = time.time()
            
            # Reset window if an hour has passed
            if current_time - self.window_start >= 3600:
                self.requests_made = 0
                self.window_start = current_time
                self.logger.debug("Rate limit window reset")
            
            # Check if we're at the limit
            if self.requests_made >= self.effective_limit:
                time_until_reset = 3600 - (current_time - self.window_start)
                
                if time_until_reset > 0:
                    self.logger.warning(f"Rate limit reached ({self.requests_made}/{self.effective_limit}). "
                                      f"Sleeping for {time_until_reset:.0f} seconds")
                    time.sleep(time_until_reset)
                    
                    # Reset after sleeping
                    self.requests_made = 0
                    self.window_start = time.time()
                
            return True
    
    def record_request(self, response_headers: Optional[Dict[str, str]] = None) -> Optional[RateLimitInfo]:
        """
        Record that a request was made and update rate limit info from headers.
        
        Args:
            response_headers: HTTP response headers containing rate limit info
            
        Returns:
            RateLimitInfo if headers contained rate limit data
        """
        with self.lock:
            self.requests_made += 1
            
            rate_limit_info = None
            
            # Parse GitHub rate limit headers if available
            if response_headers:
                rate_limit_info = self._parse_github_headers(response_headers)
                
                if rate_limit_info:
                    # Update our internal tracking based on API response
                    self._update_from_api(rate_limit_info)
            
            self.logger.debug(f"Request recorded. Total in window: {self.requests_made}")
            return rate_limit_info
    
    def _parse_github_headers(self, headers: Dict[str, str]) -> Optional[RateLimitInfo]:
        """Parse GitHub API rate limit headers."""
        try:
            # GitHub uses these headers for rate limiting
            limit = headers.get('X-RateLimit-Limit')
            remaining = headers.get('X-RateLimit-Remaining') 
            reset_time = headers.get('X-RateLimit-Reset')
            used = headers.get('X-RateLimit-Used')
            
            if all(v is not None for v in [limit, remaining, reset_time]):
                return RateLimitInfo(
                    limit=int(limit),
                    remaining=int(remaining),
                    reset_time=int(reset_time),
                    used=int(used) if used else int(limit) - int(remaining)
                )
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse rate limit headers: {e}")
            
        return None
    
    def _update_from_api(self, rate_limit_info: RateLimitInfo):
        """Update internal rate limiting based on API response."""
        current_time = time.time()
        
        # If API reset time is in the future, align our window
        if rate_limit_info.reset_time > current_time:
            time_to_reset = rate_limit_info.reset_time - current_time
            
            # If it's more than an hour in the future, something's wrong
            if time_to_reset <= 3600:
                # Adjust our request count based on API info
                self.requests_made = rate_limit_info.used
                
                # If we're close to the limit, be more conservative
                if rate_limit_info.remaining < 100:
                    self.logger.warning(f"API rate limit low: {rate_limit_info.remaining} remaining")
                    
                    # If very close to limit, wait until reset
                    if rate_limit_info.remaining < 10:
                        sleep_time = min(time_to_reset, 300)  # Max 5 minute wait
                        self.logger.warning(f"Very close to rate limit, sleeping {sleep_time:.0f}s")
                        time.sleep(sleep_time)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        with self.lock:
            current_time = time.time()
            window_elapsed = current_time - self.window_start
            
            return {
                "requests_made": self.requests_made,
                "effective_limit": self.effective_limit,
                "window_elapsed_seconds": window_elapsed,
                "requests_remaining": max(0, self.effective_limit - self.requests_made),
                "time_until_reset": max(0, 3600 - window_elapsed),
                "current_rate": self.requests_made / max(window_elapsed / 3600, 0.001)  # requests/hour
            }

class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on server responses."""
    
    def __init__(self, initial_requests_per_hour: int = 5000, **kwargs):
        super().__init__(initial_requests_per_hour, **kwargs)
        
        self.consecutive_429s = 0  # Count of consecutive rate limit errors
        self.adaptive_delay = 0.0  # Additional delay between requests
        self.min_delay = 0.1  # Minimum delay between requests
        self.max_delay = 10.0  # Maximum adaptive delay
        
        self.success_count = 0  # Count successful requests
        self.error_count = 0   # Count failed requests
        
    def handle_429_response(self, retry_after: Optional[int] = None):
        """Handle HTTP 429 (Too Many Requests) response."""
        with self.lock:
            self.consecutive_429s += 1
            self.error_count += 1
            
            # Exponential backoff for adaptive delay
            self.adaptive_delay = min(
                self.max_delay,
                self.min_delay * (2 ** self.consecutive_429s)
            )
            
            self.logger.warning(f"Rate limited (429). Consecutive: {self.consecutive_429s}, "
                              f"adaptive delay now: {self.adaptive_delay:.1f}s")
            
            # If server provided retry-after header, respect it
            if retry_after:
                sleep_time = min(retry_after, 300)  # Max 5 minutes
                self.logger.info(f"Server requested retry after {retry_after}s, sleeping {sleep_time}s")
                time.sleep(sleep_time)
    
    def handle_success_response(self):
        """Handle successful response - reduce adaptive delay."""
        with self.lock:
            self.success_count += 1
            
            # Reset consecutive errors and reduce adaptive delay
            if self.consecutive_429s > 0:
                self.consecutive_429s = max(0, self.consecutive_429s - 1)
                
                # Gradually reduce adaptive delay
                self.adaptive_delay = max(
                    self.min_delay,
                    self.adaptive_delay * 0.8
                )
                
                self.logger.debug(f"Success response, adaptive delay reduced to: {self.adaptive_delay:.1f}s")
    
    def wait_if_needed(self) -> bool:
        """Enhanced wait logic with adaptive delays."""
        # First check standard rate limiting
        if not super().wait_if_needed():
            return False
        
        # Then apply adaptive delay if needed
        if self.adaptive_delay > self.min_delay:
            self.logger.debug(f"Applying adaptive delay: {self.adaptive_delay:.1f}s")
            time.sleep(self.adaptive_delay)
        else:
            # Always have minimum delay to be respectful
            time.sleep(self.min_delay)
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced status including adaptive information."""
        status = super().get_status()
        status.update({
            "consecutive_429s": self.consecutive_429s,
            "adaptive_delay": self.adaptive_delay,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(1, self.success_count + self.error_count)
        })
        return status

class GitHubRateLimiter(AdaptiveRateLimiter):
    """Specialized rate limiter for GitHub API."""
    
    def __init__(self, token_provided: bool = False):
        # GitHub limits: 5000/hour with token, 60/hour without
        requests_per_hour = 5000 if token_provided else 60
        super().__init__(requests_per_hour, buffer_percentage=0.1)
        
        self.token_provided = token_provided
        self.core_limit_info: Optional[RateLimitInfo] = None
        self.search_limit_info: Optional[RateLimitInfo] = None
        
        # GitHub has separate limits for different API endpoints
        self.search_requests_made = 0
        self.search_window_start = time.time()
        self.search_limit = 30 if token_provided else 10  # Search API has lower limits
        
    def wait_for_search_api(self) -> bool:
        """Special handling for GitHub search API which has lower limits."""
        with self.lock:
            current_time = time.time()
            
            # Reset search window (1 minute for search API)
            if current_time - self.search_window_start >= 60:
                self.search_requests_made = 0
                self.search_window_start = current_time
            
            # Check search API limit
            if self.search_requests_made >= self.search_limit:
                time_until_reset = 60 - (current_time - self.search_window_start)
                
                if time_until_reset > 0:
                    self.logger.warning(f"Search API rate limit reached. "
                                      f"Sleeping for {time_until_reset:.0f} seconds")
                    time.sleep(time_until_reset)
                    self.search_requests_made = 0
                    self.search_window_start = time.time()
            
            self.search_requests_made += 1
            return True
    
    def record_search_request(self, response_headers: Optional[Dict[str, str]] = None):
        """Record a search API request."""
        self.record_request(response_headers)
        
        # Parse search-specific rate limit headers
        if response_headers:
            search_limit = response_headers.get('X-RateLimit-Limit')
            search_remaining = response_headers.get('X-RateLimit-Remaining')
            
            if search_limit and search_remaining:
                self.logger.debug(f"Search API: {search_remaining}/{search_limit} remaining")

def create_github_rate_limiter(token_provided: bool = False) -> GitHubRateLimiter:
    """Factory function to create appropriate GitHub rate limiter."""
    return GitHubRateLimiter(token_provided)

def main():
    """Test rate limiter functionality."""
    logging.basicConfig(level=logging.DEBUG)
    
    # Test basic rate limiter
    limiter = RateLimiter(requests_per_hour=10)  # Very low for testing
    
    print("Testing rate limiter...")
    for i in range(15):
        print(f"Request {i+1}: ", end="")
        
        start_time = time.time()
        limiter.wait_if_needed()
        limiter.record_request()
        elapsed = time.time() - start_time
        
        print(f"took {elapsed:.1f}s")
        
        # Show status every few requests
        if (i + 1) % 5 == 0:
            status = limiter.get_status()
            print(f"Status: {status['requests_made']}/{status['effective_limit']} requests, "
                  f"{status['requests_remaining']} remaining")

if __name__ == "__main__":
    main()
