"""In-process sliding-window rate limiting for api.py (Phase 4.1).

Single-instance only, unlike SessionMemory/RetrievalCache/SpendTracker: those
degrade from Redis to in-process, but this has no Redis-backed mode at all.
A rate limiter's job is bounding *concurrent* request rate, which a portfolio
deployment's actual scale (one instance) never needs made correct across
replicas — building that distributed correctness here would be effort spent
on a scale this project doesn't operate at. Documented, not silently assumed
away: a real multi-replica deployment would need a shared backend for this
to mean anything across instances, the same caveat LIMITATIONS.md already
gives the spend cap.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding window: at most `limit` calls per `window_seconds`, per key."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        """True and records a hit if `key` is under its limit; False and
        records nothing if it is not — a rejected call must not itself count
        toward the window it was rejected from."""
        now = time.monotonic() if now is None else now
        hits = self._hits[key]
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.limit:
            return False

        hits.append(now)
        return True

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds until `key` would next be allowed, 0 if it already would be."""
        now = time.monotonic() if now is None else now
        hits = self._hits[key]
        if len(hits) < self.limit:
            return 0.0
        return max(0.0, hits[0] + self.window_seconds - now)
