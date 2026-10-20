from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


def test_build_summary_asks_the_model_layer_and_reports_real_usage_and_model():
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content="3-sentence summary.",
        usage_metadata={"input_tokens": 80, "output_tokens": 25, "total_tokens": 105},
        response_metadata={"model_name": "gpt-6-luna"},
    )

    messages = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
    ]
    result, usage = mem.build_summary(messages, llm)

    assert result == "3-sentence summary."
    assert usage == {"input": 80, "output": 25, "model": "gpt-6-luna"}
    sent = llm.invoke.call_args.args[0]
    assert "exactly 3 sentences" in sent[0].content
    assert "Human: Hi" in sent[1].content and "Ai: Hello!" in sent[1].content


def test_build_summary_reads_text_out_of_a_block_list_response():
    """Claude can answer with a list of content blocks; the summary must be its text."""
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content=[
            {"type": "thinking", "thinking": "", "signature": "x"},
            {"type": "text", "text": " Summary. "},
        ]
    )

    result, _ = mem.build_summary([HumanMessage(content="Hi")], llm)

    assert result == "Summary."


def test_build_summary_empty_messages():
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None
    result, usage = mem.build_summary([], MagicMock())
    assert result == ""
    assert usage == {"input": 0, "output": 0}


def test_build_summary_failure_reports_zero_usage():
    mem = SessionMemory.__new__(SessionMemory)
    mem._redis = None
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")

    result, usage = mem.build_summary([HumanMessage(content="Hi")], llm)

    assert result == ""
    assert usage == {"input": 0, "output": 0}
