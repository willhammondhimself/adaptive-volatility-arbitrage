"""
LRU cache service for expensive computations.
"""

import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """Least Recently Used (LRU) cache with size limit."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, moving to end (most recent)."""
        if key not in self._cache:
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """Add value to cache, evicting oldest if at capacity."""
        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            # Add new entry
            self._cache[key] = value

            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove first (oldest)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    @staticmethod
    def hash_dict(data: dict) -> str:
        """Create deterministic hash for dictionary."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
