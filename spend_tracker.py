"""Daily spend cap tracking for api.py (Phase 4.1).

Reuses the same Redis-or-in-process degradation SessionMemory and
RetrievalCache already use (redis_backend.connect) — a real Redis client
when REDIS_URL is reachable, fakeredis in-process otherwise. Unlike
RateLimiter, this deliberately keeps working correctly across multiple
instances when Redis is configured: an in-process-only spend cap resets on
every restart, which meaningfully undermines the one thing it exists to
guarantee — bounding real-money cost on a publicly deployed app.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import redis_backend
from config import load_settings

logger = logging.getLogger(__name__)

# A day's key outlives the day it counts by one, so a request arriving right
# at a UTC day boundary can never race a key that already expired.
_SPEND_KEY_TTL = 60 * 60 * 24 * 2


class SpendTracker:
    """Cumulative spend for the current UTC day, keyed by calendar date."""

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url if redis_url is not None else load_settings().redis_url
        self._redis, self._backend = redis_backend.connect(url, purpose="spend tracker")

    @property
    def backend(self) -> str:
        return getattr(self, "_backend", redis_backend.BACKEND_DISABLED)

    @staticmethod
    def _key(day: str | None = None) -> str:
        day = day or datetime.now(UTC).strftime("%Y-%m-%d")
        return f"spend:{day}"

    def today_total(self) -> float:
        if not self._redis:
            return 0.0
        try:
            raw = self._redis.get(self._key())
            return float(raw) if raw else 0.0
        except Exception as exc:
            logger.warning("Failed to read today's spend, treating it as zero: %s", exc)
            return 0.0

    def add(self, amount_usd: float) -> float:
        """Record spend for today, returning the new running total.

        Fails open: if tracking itself breaks, this never blocks the request
        that is trying to record its own cost — it logs and returns whatever
        today_total() can still read, same degrade-not-crash discipline as
        every other Redis-backed feature in this project.
        """
        if not self._redis or amount_usd <= 0:
            return self.today_total()
        try:
            key = self._key()
            total = self._redis.incrbyfloat(key, amount_usd)
            self._redis.expire(key, _SPEND_KEY_TTL)
            return float(total)
        except Exception as exc:
            logger.warning("Failed to record spend: %s", exc)
            return self.today_total()
