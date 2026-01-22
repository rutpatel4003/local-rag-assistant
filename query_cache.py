import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
import threading


@dataclass
class CacheEntry:
    """Single cache entry with query embedding, results, and metadata"""
    query: str
    embedding: np.ndarray
    docs: List[Document]
    timestamp: float
    last_access: float
    hits: int = 0


class SemanticQueryCache:
    """
    Semantic cache using embedding similarity for query matching.

    Features:
    - Cosine similarity threshold (default 0.90)
    - TTL-based expiration (default 1 hour)
    - LRU eviction when max size reached
    - Thread-safe operations
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_size: int = 1000
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings (safe)"""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - entry.timestamp) > self.ttl_seconds

    def _evict_expired(self):
        """Remove all expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if (current_time - entry.timestamp) > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            print(f"  Cache: Evicted {len(expired_keys)} expired entries")

    def _evict_lru(self):
        """
        Evict least recently used entry if cache is full
        """
        if len(self._cache) >= self.max_size:
            lru_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_access
            )
            del self._cache[lru_key]
            print(f"Cache: Evicted LRU entry (size limit reached)")

    def get(
        self,
        query: str,
        embedding: np.ndarray
    ) -> Optional[List[Document]]:
        """
        Retrieve cached results for semantically similar query.
        """
        with self._lock:
            # first check for exact query match (fast path)
            if query in self._cache:
                entry = self._cache[query]
                if not self._is_expired(entry):
                    entry.hits += 1
                    entry.last_access = time.time()
                    self._hits += 1
                    print(f"  Cache HIT (exact): '{query[:50]}...' [{entry.hits} hits]")
                    return entry.docs
                else:
                    del self._cache[query]

            # check semantic similarity (slower path)
            best_similarity = 0.0
            best_entry = None

            for entry in self._cache.values():
                if self._is_expired(entry):
                    continue

                similarity = self._cosine_similarity(embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if best_entry and best_similarity >= self.similarity_threshold:
                best_entry.hits += 1
                best_entry.last_access = time.time()
                self._hits += 1
                print(f"Cache HIT (semantic): sim={best_similarity:.3f}, '{query[:50]}...'")
                return best_entry.docs

            # cache miss
            self._misses += 1
            return None

    def put(
        self,
        query: str,
        embedding: np.ndarray,
        docs: List[Document]
    ):
        """
        Store query results in cache.
        """
        with self._lock:
            # evict expired entries periodically
            if len(self._cache) > 0 and len(self._cache) % 100 == 0:
                self._evict_expired()

            # evict LRU if needed
            self._evict_lru()

            # store new entry
            entry = CacheEntry(
                query=query,
                embedding=embedding,
                docs=docs,
                timestamp=time.time(),
                last_access=time.time(),
                hits=0
            )
            self._cache[query] = entry

    def clear(self):
        """
        Clear all cache 
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            print("Cache: Cleared all entries")

    def stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "total_queries": total,
            }

    def print_stats(self):
        """
        Print cache statistics
        """
        stats = self.stats()
        print(f"\n{'='*50}")
        print(f"Cache Statistics:")
        print(f"  Size: {stats['size']}/{self.max_size}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.1f}%")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"{'='*50}\n")


# global cache instance
_global_cache: Optional[SemanticQueryCache] = None


def get_cache() -> SemanticQueryCache:
    """
    Get or create global cache instance
    """
    global _global_cache
    if _global_cache is None:
        from config import Config
        _global_cache = SemanticQueryCache(
            similarity_threshold=Config.Performance.CACHE_SIMILARITY_THRESHOLD,
            ttl_seconds=Config.Performance.CACHE_TTL_SECONDS,
            max_size=Config.Performance.CACHE_MAX_SIZE
        )
    return _global_cache


def clear_cache():
    """
    Clear global cache
    """
    global _global_cache
    if _global_cache:
        _global_cache.clear()
