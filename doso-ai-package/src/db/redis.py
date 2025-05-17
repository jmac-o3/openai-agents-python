"""
Redis client configuration and utilities with enhanced TTL management

This module provides Redis caching functionality including:
- Connection pooling
- Key generation
- TTL-based caching policies
- Pattern-based cache invalidation
- Cache statistics tracking
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from redis import Redis
from redis.connection import ConnectionPool

from ..config import settings
from ..monitoring.tracing import trace_async_method, trace_method

logger = logging.getLogger(__name__)

# Create Redis connection pool with appropriate settings
redis_pool = ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
    health_check_interval=30,
)

# Default TTL values (in seconds) for different cache types
TTL_CONFIG = {
    "default": 3600,  # 1 hour
    "inventory": 1800,  # 30 minutes
    "market_data": 7200,  # 2 hours
    "dealer_profile": 86400,  # 24 hours
    "recommendations": 43200,  # 12 hours
    "allocation": 900,  # 15 minutes
    "analysis": 3600,  # 1 hour
    "constraints": 21600,  # 6 hours
    "orders": 1800,  # 30 minutes
}


def get_redis() -> Redis:
    """Get Redis client instance from the connection pool"""
    return Redis(connection_pool=redis_pool)


def cache_key(*args: Any) -> str:
    """
    Generate cache key from arguments
    
    Args:
        *args: Key components to join
        
    Returns:
        Cache key string
    """
    return ":".join(str(arg) for arg in args)


@trace_async_method("redis.get_cached_data")
async def get_cached_data(key: str) -> Optional[dict]:
    """
    Get data from Redis cache with monitoring
    
    Args:
        key: Cache key
        
    Returns:
        Cached data or None if not found
    """
    redis = get_redis()
    start_time = time.time()
    
    try:
        data = redis.get(key)
        
        # Track cache hit/miss
        if data:
            redis.hincrby("cache:stats", "hits", 1)
            result = json.loads(data)
            
            # Update last access time
            redis.hset("cache:last_access", key, int(time.time()))
            
            return result
        else:
            redis.hincrby("cache:stats", "misses", 1)
            return None
            
    except Exception as e:
        logger.error(f"Redis get error: {str(e)}", extra={"key": key})
        redis.hincrby("cache:stats", "errors", 1)
        return None
    finally:
        # Track operation timing
        duration_ms = (time.time() - start_time) * 1000
        redis.lpush("cache:timings", duration_ms)
        redis.ltrim("cache:timings", 0, 999)  # Keep last 1000 timings


@trace_async_method("redis.set_cached_data")
async def set_cached_data(
    key: str, 
    data: dict, 
    expire_seconds: Optional[int] = None,
    cache_type: str = "default"
) -> None:
    """
    Set data in Redis cache with appropriate TTL
    
    Args:
        key: Cache key
        data: Data to cache
        expire_seconds: Custom expiration time in seconds 
        cache_type: Type of cached data for TTL determination
    """
    redis = get_redis()
    
    # Get TTL from config or use default
    ttl = expire_seconds or TTL_CONFIG.get(cache_type, TTL_CONFIG["default"])
    
    try:
        # Store data with expiration
        redis.setex(key, ttl, json.dumps(data))
        
        # Track cache key by type for later management
        redis.sadd(f"cache:keys:{cache_type}", key)
        redis.hincrby("cache:stats", "sets", 1)
        
        # Store metadata about this cache entry
        redis.hset("cache:metadata", key, json.dumps({
            "type": cache_type,
            "created_at": int(time.time()),
            "ttl": ttl
        }))
        
    except Exception as e:
        logger.error(f"Redis set error: {str(e)}", 
                    extra={"key": key, "cache_type": cache_type})
        redis.hincrby("cache:stats", "errors", 1)


@trace_async_method("redis.invalidate_cache")
async def invalidate_cache(key: str) -> None:
    """
    Invalidate cache for given key
    
    Args:
        key: Cache key to invalidate
    """
    redis = get_redis()
    
    try:
        # Remove key and track invalidations
        redis.delete(key)
        redis.hincrby("cache:stats", "invalidations", 1)
        
        # Clean up metadata and type tracking
        redis.hdel("cache:metadata", key)
        redis.hdel("cache:last_access", key)
        
        # Remove from all type sets
        for cache_type in TTL_CONFIG.keys():
            redis.srem(f"cache:keys:{cache_type}", key)
            
    except Exception as e:
        logger.error(f"Redis invalidate error: {str(e)}", extra={"key": key})
        redis.hincrby("cache:stats", "errors", 1)


@trace_async_method("redis.invalidate_pattern")
async def invalidate_by_pattern(pattern: str) -> int:
    """
    Invalidate all cache keys matching a pattern
    
    Args:
        pattern: Redis key pattern to match (e.g., "user:*")
        
    Returns:
        Number of keys invalidated
    """
    redis = get_redis()
    
    try:
        # Find all matching keys
        keys = redis.keys(pattern)
        
        if not keys:
            return 0
            
        # Delete all keys
        redis.delete(*keys)
        count = len(keys)
        
        # Update stats
        redis.hincrby("cache:stats", "pattern_invalidations", 1)
        redis.hincrby("cache:stats", "invalidations", count)
        
        # Clean up metadata and tracking
        for key in keys:
            redis.hdel("cache:metadata", key)
            redis.hdel("cache:last_access", key)
            
            # Remove from all type sets
            for cache_type in TTL_CONFIG.keys():
                redis.srem(f"cache:keys:{cache_type}", key)
        
        return count
        
    except Exception as e:
        logger.error(f"Redis pattern invalidate error: {str(e)}", 
                    extra={"pattern": pattern})
        redis.hincrby("cache:stats", "errors", 1)
        return 0


@trace_async_method("redis.extend_ttl")
async def extend_cache_ttl(key: str, additional_seconds: int) -> bool:
    """
    Extend the TTL of a cache key
    
    Args:
        key: Cache key to extend
        additional_seconds: Additional seconds to add to TTL
        
    Returns:
        True if TTL was extended, False if key not found
    """
    redis = get_redis()
    
    try:
        # Check if key exists
        current_ttl = redis.ttl(key)
        
        if current_ttl <= 0:
            return False
            
        # Extend TTL
        new_ttl = current_ttl + additional_seconds
        redis.expire(key, new_ttl)
        
        # Update metadata
        metadata = redis.hget("cache:metadata", key)
        if metadata:
            metadata_dict = json.loads(metadata)
            metadata_dict["ttl"] = new_ttl
            redis.hset("cache:metadata", key, json.dumps(metadata_dict))
        
        return True
        
    except Exception as e:
        logger.error(f"Redis extend TTL error: {str(e)}", 
                    extra={"key": key, "additional_seconds": additional_seconds})
        return False


@trace_async_method("redis.get_cache_stats")
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get cache usage statistics
    
    Returns:
        Dictionary with cache statistics
    """
    redis = get_redis()
    
    try:
        # Gather stats
        stats = redis.hgetall("cache:stats") or {}
        
        # Convert string values to integers
        stats = {k: int(v) for k, v in stats.items()}
        
        # Calculate hit rate
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        total = hits + misses
        hit_rate = (hits / total) * 100 if total > 0 else 0
        
        # Get timing statistics
        timings = redis.lrange("cache:timings", 0, -1)
        timings = [float(t) for t in timings] if timings else [0]
        
        # Count keys by type
        key_counts = {}
        for cache_type in TTL_CONFIG.keys():
            count = redis.scard(f"cache:keys:{cache_type}")
            key_counts[cache_type] = count
            
        return {
            "hit_rate": hit_rate,
            "hits": hits,
            "misses": misses, 
            "sets": stats.get("sets", 0),
            "invalidations": stats.get("invalidations", 0),
            "errors": stats.get("errors", 0),
            "avg_response_time_ms": sum(timings) / len(timings),
            "key_count": sum(key_counts.values()),
            "key_counts_by_type": key_counts
        }
        
    except Exception as e:
        logger.error(f"Redis stats error: {str(e)}")
        return {"error": str(e)}


@trace_async_method("redis.clean_expired_keys")
async def clean_expired_keys(cache_type: Optional[str] = None) -> int:
    """
    Clean up expired and unused keys
    
    Args:
        cache_type: Optional cache type to clean up (all types if None)
        
    Returns:
        Number of keys removed
    """
    redis = get_redis()
    removed = 0
    
    try:
        # If specific type provided, clean only that type
        types_to_clean = [cache_type] if cache_type else TTL_CONFIG.keys()
        
        for ctype in types_to_clean:
            # Get keys of this type
            keys = redis.smembers(f"cache:keys:{ctype}")
            
            for key in keys:
                # Check if key still exists
                if not redis.exists(key):
                    # Key expired, clean up metadata
                    redis.srem(f"cache:keys:{ctype}", key)
                    redis.hdel("cache:metadata", key)
                    redis.hdel("cache:last_access", key)
                    removed += 1
        
        return removed
        
    except Exception as e:
        logger.error(f"Redis cleanup error: {str(e)}")
        return removed
