"""
    Caching - solve the problem of repeated queries
        Query cache: 
        Embeding Cache
        Vector Cache
        LLM Response Cache 
    
    Redis - a popular in-memory data structure store, used as a database, cache, and message broker
"""

import redis
import json
import hashlib

# Initialize redis cache
cache = redis.Redis(host="localhost", port=6379, db=0)

def get_cached_response(query, context):
    """
    Check if the response for the given query and context is already cached
    
    Args:
        query (str): The query to check
        context (str): The context to check
    
    Returns:
        str: The cached response if found, None otherwise
    """

    # create cache key
    cache_key = hashlib.md5(f"{query}_{context}".encode()).hexdigest()

    # check cache hit or miss
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # generate response 
    response = generate_rag_response(query, context)

    # cache for 1 hour 
    cache.setex(cache_key, 3600, json.dumps(response))

    return response
    