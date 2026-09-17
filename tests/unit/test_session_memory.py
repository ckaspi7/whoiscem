from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memory.session_memory import _SESSION_TTL, SessionMemory


@pytest.fixture
def memory_with_fake_redis(fake_redis):
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = fake_redis
    return mem


@pytest.fixture
def memory_no_redis():
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None
    return mem


def test_available_when_redis_connected(memory_with_fake_redis):
    assert memory_with_fake_redis.available is True


def test_unavailable_when_redis_none(memory_no_redis):
    assert memory_no_redis.available is False


def test_new_session_returns_empty_summary(memory_with_fake_redis):
    result = memory_with_fake_redis.load_summary("brand-new-session-id")
    assert result == ""


def test_existing_session_loads_summary(memory_with_fake_redis):
    memory_with_fake_redis.save_summary("sess-123", "Cem likes music.")
    result = memory_with_fake_redis.load_summary("sess-123")
    assert result == "Cem likes music."


def test_summary_written_after_save(memory_with_fake_redis, fake_redis):
    memory_with_fake_redis.save_summary("s1", "Summary text.")
    raw = fake_redis.get("session:s1:summary")
    assert raw == "Summary text."


def test_summary_ttl_set_to_30_days(memory_with_fake_redis, fake_redis):
    memory_with_fake_redis.save_summary("s2", "test")
    ttl = fake_redis.ttl("session:s2:summary")
    # fakeredis returns TTL within a second of what was set
    assert abs(ttl - _SESSION_TTL) <= 2


def test_missing_redis_load_returns_empty(memory_no_redis):
    result = memory_no_redis.load_summary("any-id")
    assert result == ""


def test_missing_redis_save_does_not_raise(memory_no_redis):
    memory_no_redis.save_summary("any-id", "some summary")  # should not raise


def test_build_summary_calls_gpt4o_mini():
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None

    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="3-sentence summary."))]
    )

    messages = [
        {"role": "human", "content": "Hi"},
        {"role": "ai", "content": "Hello!"},
    ]
    result = mem.build_summary(messages, openai_client)

    assert result == "3-sentence summary."
    call_kwargs = openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"


def test_build_summary_empty_messages():
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None
    result = mem.build_summary([], MagicMock())
    assert result == ""
