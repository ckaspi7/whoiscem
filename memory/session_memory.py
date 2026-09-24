from __future__ import annotations

import logging

import redis_backend
from config import load_settings

logger = logging.getLogger(__name__)

_SESSION_TTL = 60 * 60 * 24 * 30  # 30 days

# Re-exported for callers and tests that referred to these before the
# connection logic moved into redis_backend.
BACKEND_REDIS = redis_backend.BACKEND_REDIS
BACKEND_IN_PROCESS = redis_backend.BACKEND_IN_PROCESS
BACKEND_DISABLED = redis_backend.BACKEND_DISABLED


class SessionMemory:
    """Rolling conversation summary keyed by session id.

    Backed by Redis when ``REDIS_URL`` is set. With no URL — or when the server
    is unreachable — it falls back to an in-process store so the feature still
    works with zero infrastructure, at the cost of not surviving a restart or
    being shared between processes. Every failure path degrades instead of
    raising: the app never goes down because a cache is missing.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url if redis_url is not None else load_settings().redis_url
        self._redis, self._backend = redis_backend.connect(url, purpose="session memory")

    @property
    def available(self) -> bool:
        return self._redis is not None

    @property
    def backend(self) -> str:
        """``redis`` | ``in-process`` | ``disabled`` — surfaced in the UI so the
        deployment's real behaviour is visible rather than assumed."""
        return getattr(self, "_backend", BACKEND_DISABLED)

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

    def build_summary(self, messages: list[dict], openai_client) -> str:
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
                        "content": (
                            "Summarise this conversation in exactly 3 sentences for future context injection."
                        ),
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
