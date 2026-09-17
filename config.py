"""Runtime configuration, resolved from the environment in one place.

Every external backend is selected here rather than at the call site, so the
same code path runs against an on-disk store on a laptop, a container under
docker-compose, and a managed service in production. Call sites depend on
``Settings``; the concrete client is built by ``retrieval.backends``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Loaded here so every caller sees the same environment, whether it came in
# through streamlit, the eval harness, a script, or a test. Real environment
# variables always win over .env, which keeps CI and Streamlit Cloud authoritative.
load_dotenv()

QdrantMode = Literal["embedded", "server", "cloud"]

QDRANT_MODES: tuple[str, ...] = ("embedded", "server", "cloud")

DEFAULT_QDRANT_MODE = "embedded"
DEFAULT_QDRANT_PATH = ".qdrant"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_RESUME_PATH = os.path.join("data", "resume.md")


class ConfigError(ValueError):
    """Raised when the environment describes a backend that cannot be built."""


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process.

    Attributes:
        qdrant_mode: ``embedded`` (on-disk, no server), ``server`` (host/port),
            or ``cloud`` (URL + API key).
        qdrant_path: Storage directory used by ``embedded`` mode. One process at
            a time may hold it — the client takes an exclusive lock.
        redis_url: Empty means "no Redis server"; session memory falls back to
            an in-process store rather than turning itself off.
        resume_path: Resume indexed on first run — Markdown or PDF.
    """

    qdrant_mode: QdrantMode = DEFAULT_QDRANT_MODE
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_host: str = DEFAULT_QDRANT_HOST
    qdrant_port: int = DEFAULT_QDRANT_PORT
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    redis_url: str = ""
    resume_path: str = DEFAULT_RESUME_PATH

    def __post_init__(self) -> None:
        if self.qdrant_mode not in QDRANT_MODES:
            raise ConfigError(
                f"QDRANT_MODE must be one of {', '.join(QDRANT_MODES)} — got {self.qdrant_mode!r}"
            )
        if self.qdrant_mode == "cloud" and not self.qdrant_url:
            raise ConfigError("QDRANT_MODE=cloud requires QDRANT_URL (and usually QDRANT_API_KEY)")


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build ``Settings`` from environment variables (``os.environ`` by default)."""
    src = os.environ if env is None else env

    def get(key: str, default: str = "") -> str:
        return str(src.get(key, default)).strip()

    port_raw = get("QDRANT_PORT", str(DEFAULT_QDRANT_PORT))
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ConfigError(f"QDRANT_PORT must be an integer — got {port_raw!r}") from exc

    return Settings(
        qdrant_mode=get("QDRANT_MODE", DEFAULT_QDRANT_MODE).lower(),  # type: ignore[arg-type]
        qdrant_path=get("QDRANT_PATH", DEFAULT_QDRANT_PATH),
        qdrant_host=get("QDRANT_HOST", DEFAULT_QDRANT_HOST),
        qdrant_port=port,
        qdrant_url=get("QDRANT_URL"),
        qdrant_api_key=get("QDRANT_API_KEY"),
        redis_url=get("REDIS_URL"),
        resume_path=get("RESUME_PATH", DEFAULT_RESUME_PATH),
    )
