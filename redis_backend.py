"""Shared Redis-or-fallback connection logic.

Session memory and the retrieval cache both need the same thing: a real Redis
client when ``REDIS_URL`` is set and reachable, an in-process fallback
otherwise, and never a raised exception just because a cache is unavailable —
a cache is an optimization, and losing it must never take the app down with
it. One connection routine, so both features degrade identically instead of
each reinventing the same three-way fallback.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

BACKEND_REDIS = "redis"
BACKEND_IN_PROCESS = "in-process"
BACKEND_DISABLED = "disabled"

_REDIS_SCHEMES = ("redis", "rediss", "unix")


def redact_url(url: str) -> str:
    """``url`` without its credentials or query string, so it is safe to log.

    ``rediss://default:secret@host:6379/0?x=y`` becomes ``rediss://host:6379/0``. A string that is not
    a recognisable Redis URL is replaced outright rather than echoed, since a malformed one
    (``user:secret@host``) can still carry the password.
    """
    try:
        parts = urlsplit(url)
        host, port = parts.hostname or "", parts.port
    except ValueError:
        return "<unparseable Redis URL>"
    if parts.scheme not in _REDIS_SCHEMES:
        return "<unrecognised Redis URL>"
    if ":" in host:  # an IPv6 literal needs its brackets back
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port else host
    return f"{parts.scheme}://{netloc}{parts.path}"


def connect(redis_url: str, *, purpose: str) -> tuple[object | None, str]:
    """Return ``(client, backend_name)``.

    ``purpose`` names the caller in log messages only ("session memory",
    "retrieval cache") — it has no effect on behaviour.
    """
    client = None
    backend = BACKEND_DISABLED

    if redis_url:
        try:
            import redis as redis_lib

            candidate = redis_lib.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            candidate.ping()
            client, backend = candidate, BACKEND_REDIS
        except Exception as exc:
            logger.warning(
                "Redis unreachable at %s — %s falls back in-process: %s", redact_url(redis_url), purpose, exc
            )

    if client is None:
        try:
            import fakeredis

            client, backend = fakeredis.FakeRedis(decode_responses=True), BACKEND_IN_PROCESS
        except Exception as exc:
            logger.warning("No backend available for %s: %s", purpose, exc)

    return client, backend
