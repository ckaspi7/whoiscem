from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_SESSION_TTL = 60 * 60 * 24 * 30  # 30 days


class SessionMemory:
    """Redis-backed rolling summary for cross-session memory.

    Falls back to no-op gracefully when Redis is unavailable so the app
    never crashes due to a missing cache connection.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = None
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            import redis as redis_lib
            self._redis = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=2)
            self._redis.ping()
        except Exception as exc:
            logger.warning("Redis unavailable — session memory disabled: %s", exc)
            self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def load_summary(self, session_id: str) -> str:
        if not self._redis:
            return ""
        try:
            return self._redis.get(self._key(session_id)) or ""
        except Exception as exc:
            logger.warning("Failed to load session memory: %s", exc)
            return ""

    def save_summary(self, session_id: str, summary: str) -> None:
        if not self._redis:
            return
        try:
            self._redis.set(self._key(session_id), summary, ex=_SESSION_TTL)
        except Exception as exc:
            logger.warning("Failed to save session memory: %s", exc)

    def build_summary(self, messages: List[dict], openai_client) -> str:
        """Summarise the conversation to 3 sentences for future context injection."""
        if not messages:
            return ""
        transcript = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages if isinstance(m["content"], str)
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarise this conversation in exactly 3 sentences for future context injection.",
                    },
                    {"role": "user", "content": transcript},
                ],
                temperature=0,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Failed to build conversation summary: %s", exc)
            return ""

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:summary"
