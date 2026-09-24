"""Caches query embeddings and full retrieval results.

Both target the same expensive operations behind resume search: an OpenAI
embedding call, and — for the fused/reranked strategies — a cross-encoder
pass. Redis when ``REDIS_URL`` is set and reachable, an in-process fallback
otherwise, following the same degrade-never-raise pattern as session memory:
a cache being unavailable means paying the real cost again, never an error.

Retrieval-result entries are keyed with the corpus fingerprint (source bytes
plus chunker version), so a rebuilt index — a new resume, a changed chunking
strategy — cannot serve a stale answer from before the rebuild. The key
simply stops matching, and the caller recomputes.

Concretely useful today, not just forward-looking: eval/ablate_retrieval.py
runs eleven strategy variants over the same 24 referenced questions, and
several of them re-embed the identical query text — dense_only@3 and
dense_only@5 both embed the same string for different top_k, and every RRF and
reranked variant recomputes the fused candidate list from scratch. Without
caching, one query is embedded up to nine times across that one ablation run.
"""

from __future__ import annotations

import hashlib
import json
import logging

import redis_backend
from config import load_settings
from retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)

_EMBEDDING_TTL = 60 * 60 * 24 * 7  # a week: an embedding for a fixed (model, text) never changes
_RETRIEVAL_TTL = 60 * 60 * 6  # six hours: cheap to recompute, and the corpus can change underneath it


class RetrievalCache:
    """Query-embedding and retrieval-result cache."""

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url if redis_url is not None else load_settings().redis_url
        self._redis, self._backend = redis_backend.connect(url, purpose="retrieval cache")

    @property
    def backend(self) -> str:
        return self._backend

    # -- embeddings -------------------------------------------------------

    def get_embedding(self, model: str, text: str) -> list[float] | None:
        if not self._redis:
            return None
        try:
            raw = self._redis.get(self._embedding_key(model, text))
        except Exception as exc:
            logger.warning("Embedding cache read failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    def set_embedding(self, model: str, text: str, vector: list[float]) -> None:
        if not self._redis:
            return
        try:
            self._redis.set(self._embedding_key(model, text), json.dumps(vector), ex=_EMBEDDING_TTL)
        except Exception as exc:
            logger.warning("Embedding cache write failed: %s", exc)

    @staticmethod
    def _embedding_key(model: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        return f"embed:{model}:{digest}"

    # -- retrieval results --------------------------------------------------

    def get_retrieval(self, corpus_fingerprint: str, cache_key: str) -> list[ScoredChunk] | None:
        if not self._redis:
            return None
        try:
            raw = self._redis.get(self._retrieval_key(corpus_fingerprint, cache_key))
        except Exception as exc:
            logger.warning("Retrieval cache read failed: %s", exc)
            return None
        if not raw:
            return None
        try:
            return [ScoredChunk(**row) for row in json.loads(raw)]
        except (TypeError, ValueError) as exc:
            logger.warning("Retrieval cache entry was malformed, ignoring it: %s", exc)
            return None

    def set_retrieval(self, corpus_fingerprint: str, cache_key: str, chunks: list[ScoredChunk]) -> None:
        if not self._redis:
            return
        try:
            payload = json.dumps(
                [
                    {"chunk_id": c.chunk_id, "text": c.text, "score": c.score, "section": c.section}
                    for c in chunks
                ]
            )
            self._redis.set(self._retrieval_key(corpus_fingerprint, cache_key), payload, ex=_RETRIEVAL_TTL)
        except Exception as exc:
            logger.warning("Retrieval cache write failed: %s", exc)

    @staticmethod
    def _retrieval_key(corpus_fingerprint: str, cache_key: str) -> str:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
        return f"retrieval:{corpus_fingerprint}:{digest}"
