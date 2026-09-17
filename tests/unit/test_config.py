from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import (
    DEFAULT_QDRANT_PATH,
    DEFAULT_RESUME_PATH,
    ConfigError,
    Settings,
    load_settings,
)

# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def test_defaults_to_embedded_with_empty_environment():
    settings = load_settings(env={})
    assert settings.qdrant_mode == "embedded"
    assert settings.qdrant_path == DEFAULT_QDRANT_PATH
    assert settings.resume_path == DEFAULT_RESUME_PATH
    assert settings.redis_url == ""


def test_server_mode_reads_host_and_port():
    settings = load_settings(env={"QDRANT_MODE": "server", "QDRANT_HOST": "qdrant", "QDRANT_PORT": "6334"})
    assert settings.qdrant_mode == "server"
    assert settings.qdrant_host == "qdrant"
    assert settings.qdrant_port == 6334


def test_cloud_mode_reads_url_and_key():
    settings = load_settings(
        env={
            "QDRANT_MODE": "cloud",
            "QDRANT_URL": "https://example.cloud.qdrant.io:6333",
            "QDRANT_API_KEY": "secret",
        }
    )
    assert settings.qdrant_mode == "cloud"
    assert settings.qdrant_url == "https://example.cloud.qdrant.io:6333"
    assert settings.qdrant_api_key == "secret"


def test_mode_is_case_insensitive_and_trimmed():
    assert load_settings(env={"QDRANT_MODE": " Server "}).qdrant_mode == "server"


def test_unknown_mode_is_rejected():
    with pytest.raises(ConfigError, match="QDRANT_MODE"):
        load_settings(env={"QDRANT_MODE": "sqlite"})


def test_cloud_mode_without_url_is_rejected():
    with pytest.raises(ConfigError, match="QDRANT_URL"):
        load_settings(env={"QDRANT_MODE": "cloud"})


def test_non_numeric_port_is_rejected():
    with pytest.raises(ConfigError, match="QDRANT_PORT"):
        load_settings(env={"QDRANT_PORT": "six-three-three-three"})


def test_reads_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", "/tmp/whoiscem-index")
    assert load_settings().qdrant_path == "/tmp/whoiscem-index"


# ---------------------------------------------------------------------------
# Client construction — one adapter, three modes
# ---------------------------------------------------------------------------


def test_embedded_mode_builds_on_disk_client():
    with patch("retrieval.backends.QdrantClient") as client_cls:
        from retrieval.backends import create_qdrant_client

        create_qdrant_client(Settings(qdrant_mode="embedded", qdrant_path="/tmp/idx"))

    client_cls.assert_called_once_with(path="/tmp/idx")


def test_server_mode_builds_host_port_client():
    with patch("retrieval.backends.QdrantClient") as client_cls:
        from retrieval.backends import create_qdrant_client

        create_qdrant_client(Settings(qdrant_mode="server", qdrant_host="qdrant", qdrant_port=6333))

    kwargs = client_cls.call_args.kwargs
    assert kwargs["host"] == "qdrant"
    assert kwargs["port"] == 6333


def test_cloud_mode_passes_url_and_api_key():
    with patch("retrieval.backends.QdrantClient") as client_cls:
        from retrieval.backends import create_qdrant_client

        create_qdrant_client(
            Settings(
                qdrant_mode="cloud",
                qdrant_url="https://example.cloud.qdrant.io:6333",
                qdrant_api_key="secret",
            )
        )

    kwargs = client_cls.call_args.kwargs
    assert kwargs["url"] == "https://example.cloud.qdrant.io:6333"
    assert kwargs["api_key"] == "secret"


def test_cloud_mode_without_key_sends_none():
    with patch("retrieval.backends.QdrantClient") as client_cls:
        from retrieval.backends import create_qdrant_client

        create_qdrant_client(Settings(qdrant_mode="cloud", qdrant_url="https://example.io"))

    assert client_cls.call_args.kwargs["api_key"] is None


def test_vectorstore_accepts_an_injected_client():
    injected = MagicMock()
    with (
        patch("retrieval.vectorstore.OpenAIEmbeddings"),
        patch("retrieval.backends.QdrantClient") as client_cls,
    ):
        from retrieval.vectorstore import QdrantVectorStore

        store = QdrantVectorStore(client=injected, settings=Settings())
        store.collection_exists()

    client_cls.assert_not_called()
    injected.collection_exists.assert_called_once_with("resume_chunks")
