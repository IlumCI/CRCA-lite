"""
Shodan API Client Integration

Wrapper around Shodan API with rate limiting, caching, and error handling.
"""

from typing import Dict, List, Optional, Any
import time
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger

try:
    import shodan
    SHODAN_AVAILABLE = True
except ImportError:
    SHODAN_AVAILABLE = False
    logger.warning("shodan library not available. Install with: pip install shodan")


@dataclass
class ShodanResult:
    """Result from Shodan API search."""
    
    ip: str
    port: int
    service: str
    banner: Optional[str] = None
    hostnames: List[str] = field(default_factory=list)
    location: Optional[Dict[str, Any]] = None
    org: Optional[str] = None
    asn: Optional[str] = None
    os: Optional[str] = None
    product: Optional[str] = None
    version: Optional[str] = None
    vulns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShodanClient:
    """
    Shodan API client with rate limiting and caching.
    
    Args:
        api_key: Shodan API key (or from SHODAN_API_KEY env var)
        rate_limit: Requests per second (default: 1.0 for free tier)
        cache_ttl: Cache TTL in seconds (default: 3600)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 1.0,
        cache_ttl: int = 3600
    ) -> None:
        """
        Initialize Shodan client.
        
        Args:
            api_key: Shodan API key. If None, reads from SHODAN_API_KEY env var.
            rate_limit: Maximum requests per second.
            cache_ttl: Cache time-to-live in seconds.
        """
        if not SHODAN_AVAILABLE:
            raise ImportError(
                "shodan library is required. Install with: pip install shodan"
            )
        
        self.api_key = api_key or os.getenv("SHODAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Shodan API key required. Set SHODAN_API_KEY env var or pass api_key parameter."
            )
        
        self.api = shodan.Shodan(self.api_key)
        self.rate_limit = rate_limit
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[float, List[ShodanResult]]] = {}
        self._last_request_time: float = 0.0
        self._request_lock = False
        
        logger.info("Shodan client initialized")
    
    def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.rate_limit <= 0:
            return
        
        min_interval = 1.0 / self.rate_limit
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            time.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    def _get_cache_key(self, query: str, page: int = 1) -> str:
        """Generate cache key for query."""
        return f"{query}:{page}"
    
    def _is_cache_valid(self, cache_time: float) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - cache_time) < self.cache_ttl
    
    def search(
        self,
        query: str,
        page: int = 1,
        use_cache: bool = True
    ) -> List[ShodanResult]:
        """
        Search Shodan for devices matching query.
        
        Args:
            query: Shodan search query (e.g., "product:Apache httpd")
            page: Page number (default: 1)
            use_cache: Whether to use cached results (default: True)
            
        Returns:
            List of ShodanResult objects
            
        Raises:
            shodan.APIError: If API request fails
        """
        cache_key = self._get_cache_key(query, page)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cache_time, cached_results = self._cache[cache_key]
            if self._is_cache_valid(cache_time):
                logger.debug(f"Returning cached results for query: {query}")
                return cached_results
        
        # Rate limiting
        self._rate_limit_wait()
        
        try:
            logger.info(f"Searching Shodan: {query} (page {page})")
            results = self.api.search(query, page=page)
            
            shodan_results: List[ShodanResult] = []
            
            for match in results.get("matches", []):
                result = ShodanResult(
                    ip=match.get("ip_str", ""),
                    port=match.get("port", 0),
                    service=match.get("product", "unknown"),
                    banner=match.get("data", ""),
                    hostnames=match.get("hostnames", []),
                    location=match.get("location", {}),
                    org=match.get("org", ""),
                    asn=match.get("asn", ""),
                    os=match.get("os", ""),
                    product=match.get("product", ""),
                    version=match.get("version", ""),
                    vulns=list(match.get("vulns", {}).keys()) if match.get("vulns") else [],
                    metadata=match
                )
                shodan_results.append(result)
            
            # Cache results
            if use_cache:
                self._cache[cache_key] = (time.time(), shodan_results)
            
            logger.info(f"Found {len(shodan_results)} devices")
            return shodan_results
            
        except shodan.APIError as e:
            logger.error(f"Shodan API error: {e}")
            raise
    
    def search_multiple_pages(
        self,
        query: str,
        max_results: Optional[int] = None,
        max_pages: int = 10
    ) -> List[ShodanResult]:
        """
        Search multiple pages of results.
        
        Args:
            query: Shodan search query
            max_results: Maximum number of results to return
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of ShodanResult objects
        """
        all_results: List[ShodanResult] = []
        page = 1
        
        while page <= max_pages:
            try:
                results = self.search(query, page=page)
                if not results:
                    break
                
                all_results.extend(results)
                
                if max_results and len(all_results) >= max_results:
                    return all_results[:max_results]
                
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
        
        return all_results
    
    def get_host_info(self, ip: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific host.
        
        Args:
            ip: IP address to query
            
        Returns:
            Host information dictionary or None if not found
        """
        self._rate_limit_wait()
        
        try:
            logger.debug(f"Fetching host info for {ip}")
            host_info = self.api.host(ip)
            return host_info
        except shodan.APIError as e:
            logger.warning(f"Could not fetch host info for {ip}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get information about API plan and limits.
        
        Returns:
            Dictionary with API information
        """
        self._rate_limit_wait()
        
        try:
            info = self.api.info()
            return info
        except shodan.APIError as e:
            logger.error(f"Error fetching API info: {e}")
            return {}

