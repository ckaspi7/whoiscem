"""Concrete Qdrant clients, one per deployment mode.

``embedded`` keeps the index in a local directory so the app runs with no
infrastructure at all; ``server`` and ``cloud`` talk to a real Qdrant. The
three are interchangeable behind :class:`~retrieval.vectorstore.QdrantVectorStore`.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from config import ConfigError, Settings, load_settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10


def create_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    """Build the Qdrant client described by ``settings`` (env-derived by default)."""
    cfg = settings or load_settings()

    if cfg.qdrant_mode == "embedded":
        logger.info("Qdrant: embedded mode at %s", cfg.qdrant_path)
        return QdrantClient(path=cfg.qdrant_path)

    if cfg.qdrant_mode == "server":
        logger.info("Qdrant: server mode at %s:%s", cfg.qdrant_host, cfg.qdrant_port)
        return QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port, timeout=_TIMEOUT_SECONDS)

    if cfg.qdrant_mode == "cloud":
        logger.info("Qdrant: cloud mode at %s", cfg.qdrant_url)
        return QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key or None,
            timeout=_TIMEOUT_SECONDS,
        )

    raise ConfigError(f"Unsupported QDRANT_MODE: {cfg.qdrant_mode!r}")
