"""RetrievalCache: query embeddings and full retrieval results.

Uses fakeredis directly (bypassing the connect() fallback logic, which
redis_backend's own tests would cover) so these focus on the cache's own
behaviour: correct keys, TTLs, and — critically — that every failure path
degrades to "cache miss" rather than raising into the caller.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retrieval.cache import RetrievalCache
from retrieval.types import ScoredChunk


@pytest.fixture
def cache_with_fake_redis(fake_redis):
    cache = RetrievalCache.__new__(RetrievalCache)
    cache._redis = fake_redis
    cache._backend = "in-process"
    return cache


@pytest.fixture
def cache_disabled():
    cache = RetrievalCache.__new__(RetrievalCache)
    cache._redis = None
    cache._backend = "disabled"
    return cache


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embedding_round_trips(cache_with_fake_redis):
    cache_with_fake_redis.set_embedding("text-embedding-3-small", "hello", [0.1, 0.2, 0.3])
    assert cache_with_fake_redis.get_embedding("text-embedding-3-small", "hello") == [0.1, 0.2, 0.3]


def test_embedding_miss_returns_none(cache_with_fake_redis):
    assert cache_with_fake_redis.get_embedding("text-embedding-3-small", "never cached") is None


def test_embedding_keys_are_scoped_by_model(cache_with_fake_redis):
    """The same text under two models must not collide — the vectors differ."""
    cache_with_fake_redis.set_embedding("model-a", "same text", [1.0])
    assert cache_with_fake_redis.get_embedding("model-b", "same text") is None


def test_embedding_ttl_is_set(cache_with_fake_redis, fake_redis):
    from retrieval.cache import _EMBEDDING_TTL

    cache_with_fake_redis.set_embedding("m", "text", [1.0])
    key = cache_with_fake_redis._embedding_key("m", "text")
    assert abs(fake_redis.ttl(key) - _EMBEDDING_TTL) <= 2


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


def test_retrieval_round_trips_chunks(cache_with_fake_redis):
    chunks = [
        ScoredChunk(chunk_id="3", text="Cem works at TELUS.", score=0.9, section="Experience"),
        ScoredChunk(chunk_id="1", text="Cem studied at UBC.", score=0.7, section="Education"),
    ]
    cache_with_fake_redis.set_retrieval("fp-1", "dense:5:where does he work", chunks)
    result = cache_with_fake_redis.get_retrieval("fp-1", "dense:5:where does he work")

    assert result == chunks


def test_retrieval_miss_returns_none(cache_with_fake_redis):
    assert cache_with_fake_redis.get_retrieval("fp-1", "never cached") is None


def test_a_rebuilt_corpus_invalidates_old_entries_by_fingerprint(cache_with_fake_redis):
    """The whole point of scoping by fingerprint: a re-index must not serve
    a stale answer from before it."""
    chunks = [ScoredChunk(chunk_id="1", text="old content", score=1.0)]
    cache_with_fake_redis.set_retrieval("fp-old", "dense:5:query", chunks)

    assert cache_with_fake_redis.get_retrieval("fp-new", "dense:5:query") is None


def test_a_malformed_cache_entry_is_treated_as_a_miss(cache_with_fake_redis, fake_redis):
    key = cache_with_fake_redis._retrieval_key("fp-1", "dense:5:query")
    fake_redis.set(key, "not valid json for a chunk list")

    assert cache_with_fake_redis.get_retrieval("fp-1", "dense:5:query") is None


# ---------------------------------------------------------------------------
# Degradation — a cache is an optimization, never a dependency
# ---------------------------------------------------------------------------


def test_disabled_cache_reads_return_none(cache_disabled):
    assert cache_disabled.get_embedding("m", "text") is None
    assert cache_disabled.get_retrieval("fp", "key") is None


def test_disabled_cache_writes_do_not_raise(cache_disabled):
    cache_disabled.set_embedding("m", "text", [1.0])  # must not raise
    cache_disabled.set_retrieval("fp", "key", [])  # must not raise


def test_a_backend_exception_on_read_degrades_to_a_miss():
    cache = RetrievalCache.__new__(RetrievalCache)
    cache._redis = MagicMock()
    cache._redis.get.side_effect = ConnectionError("redis down")

    assert cache.get_embedding("m", "text") is None
    assert cache.get_retrieval("fp", "key") is None


def test_a_backend_exception_on_write_does_not_raise():
    cache = RetrievalCache.__new__(RetrievalCache)
    cache._redis = MagicMock()
    cache._redis.set.side_effect = ConnectionError("redis down")

    cache.set_embedding("m", "text", [1.0])
    cache.set_retrieval("fp", "key", [])


def test_backend_property_reports_the_active_backend(cache_with_fake_redis, cache_disabled):
    assert cache_with_fake_redis.backend == "in-process"
    assert cache_disabled.backend == "disabled"
