"""
TTL-based cache for live market data.

Provides time-based expiration for API responses to balance
freshness with rate limiting.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple


class TTLCache:
    """Time-based cache with automatic expiration."""

    def __init__(self, default_ttl: int = 30):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Tuple of (value, is_stale). Value is None if not found.
        """
        if key not in self._cache:
            return None, False

        value, expires_at = self._cache[key]
        is_stale = datetime.now() > expires_at
        return value, is_stale

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self._cache[key] = (value, expires_at)

    def invalidate(self, key: str) -> bool:
        """
        Remove key from cache.

        Returns:
            True if key existed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, (_, expires_at) in self._cache.items()
            if now > expires_at
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        total = len(self._cache)
        expired = sum(1 for _, expires_at in self._cache.values() if now > expires_at)

        return {
            "total_entries": total,
            "valid_entries": total - expired,
            "expired_entries": expired,
        }
