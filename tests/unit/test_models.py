"""models.py: which model gets built how, and what makes the two providers interchangeable.

Nothing here calls a provider: the model classes are patched, and the graph-level tests
use small runnable fakes. The point is the rules that decide what a real call looks like.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import chatbot
import models
from cost import canonical_model, estimate_cost, graph_cost, usage_from_response

# ---------------------------------------------------------------------------
# Building a model: the settings each one requires
# ---------------------------------------------------------------------------


def test_the_provider_is_read_from_the_model_name():
    assert models.provider_of("claude-haiku-5-5") == "anthropic"
    assert models.provider_of("gpt-6-luna") == "openai"
    assert models.provider_of("gpt-4o-mini") == "openai"


def test_gpt_4o_mini_keeps_the_temperature_it_is_given():
    with patch("models.ChatOpenAI") as cls:
        models.build_chat_model("gpt-4o-mini", temperature=0.7, streaming=True)

    kwargs = cls.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["stream_usage"] is True and kwargs["streaming"] is True
    assert "reasoning_effort" not in kwargs


def test_gpt_6_luna_takes_no_temperature_and_no_reasoning():
    with patch("models.ChatOpenAI") as cls:
        models.build_chat_model("gpt-6-luna", temperature=0.7)

    kwargs = cls.call_args.kwargs
    assert "temperature" not in kwargs
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["stream_usage"] is True


def test_claude_haiku_takes_no_temperature_and_runs_with_thinking_off():
    with patch("langchain_anthropic.ChatAnthropic") as cls:
        models.build_chat_model("claude-haiku-5-5", temperature=0.7, streaming=True)

    kwargs = cls.call_args.kwargs
    assert kwargs["model_name"] == "claude-haiku-5-5"
    assert "temperature" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["max_tokens"] == models.ANTHROPIC_MAX_TOKENS
    assert kwargs["stream_usage"] is True and kwargs["streaming"] is True


def test_a_claude_model_never_builds_an_openai_client():
    with patch("models.ChatOpenAI") as openai_cls, patch("langchain_anthropic.ChatAnthropic"):
        models.build_chat_model("claude-haiku-5-5")

    openai_cls.assert_not_called()


def test_a_gateway_in_the_environment_is_warned_about_before_a_claude_call(monkeypatch, caplog):
    """The SDK honours ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN; the project's key
    would be sent to whatever they name."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    with patch("langchain_anthropic.ChatAnthropic"), caplog.at_level(logging.WARNING, logger="models"):
        models.build_chat_model("claude-haiku-5-5")

    assert "ANTHROPIC_BASE_URL" in caplog.text


def test_no_gateway_warning_when_the_environment_is_clean(monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with patch("langchain_anthropic.ChatAnthropic"), caplog.at_level(logging.WARNING, logger="models"):
        models.build_chat_model("claude-haiku-5-5")

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# Reading a response as text
# ---------------------------------------------------------------------------


def test_text_of_a_plain_string_response():
    assert models.message_text(AIMessage(content="Hello.")) == "Hello."


def test_text_of_a_block_list_skips_thinking_and_joins_text():
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Cem works at TELUS."},
            {"type": "text", "text": " In Vancouver."},
        ]
    )
    assert models.message_text(message) == "Cem works at TELUS. In Vancouver."


def test_text_of_a_thinking_only_response_is_empty_not_the_block_repr():
    message = AIMessage(content=[{"type": "thinking", "thinking": "", "signature": "sig"}])
    assert models.message_text(message) == ""


def test_text_of_a_bare_mock_falls_back_to_its_string_content():
    assert models.message_text(MagicMock(content="resume")) == "resume"


def test_a_block_list_answer_is_flattened_and_keeps_its_id_and_usage():
    message = AIMessage(
        content=[{"type": "text", "text": "Answer."}],
        id="msg-1",
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )

    plain = models.as_plain_message(message)

    assert plain.content == "Answer."
    assert plain.id == "msg-1"
    assert plain.usage_metadata == message.usage_metadata


def test_a_string_answer_is_returned_untouched():
    message = AIMessage(content="Answer.")
    assert models.as_plain_message(message) is message


def test_a_message_asking_for_tool_calls_keeps_its_blocks_for_the_next_round():
    """The provider expects its own blocks back with the tool results."""
    message = AIMessage(
        content=[{"type": "text", "text": "Let me look."}],
        tool_calls=[{"name": "get_resume_info", "args": {"query": "x"}, "id": "call-1", "type": "tool_call"}],
    )
    assert models.as_plain_message(message) is message


# ---------------------------------------------------------------------------
# The fallback
# ---------------------------------------------------------------------------


class _Fake(Runnable):
    """A chat model stand-in: replays one reply or raises, and records what bound tools."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls, self.bound = reply, error, 0, None

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 - Runnable's signature
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply

    def bind_tools(self, tools):
        self.bound = tools
        return self


def _reply(text, model):
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        response_metadata={"model_name": model},
    )


def test_a_healthy_primary_never_touches_the_fallback():
    primary, fallback = (
        _Fake(_reply("from primary", "gpt-6-luna")),
        _Fake(_reply("from fallback", "claude-haiku-5-5")),
    )

    answer = models.with_fallback(primary, fallback).invoke([HumanMessage(content="hi")])

    assert answer.content == "from primary"
    assert fallback.calls == 0


def test_a_failing_primary_is_answered_by_the_fallback_and_the_switch_is_logged(caplog):
    primary = _Fake(error=RuntimeError("credit balance is too low"))
    fallback = _Fake(_reply("from fallback", "claude-haiku-5-5"))

    with caplog.at_level(logging.WARNING, logger="models"):
        answer = models.with_fallback(
            primary, fallback, primary_name="gpt-6-luna", fallback_name="claude-haiku-5-5"
        ).invoke([HumanMessage(content="hi")])

    assert answer.content == "from fallback"
    assert (
        "gpt-6-luna failed (RuntimeError: credit balance is too low); answering with claude-haiku-5-5"
        in caplog.text
    )


def test_when_both_fail_the_error_surfaces_instead_of_being_swallowed():
    with pytest.raises(RuntimeError, match="primary down"):
        models.with_fallback(
            _Fake(error=RuntimeError("primary down")), _Fake(error=RuntimeError("fallback down"))
        ).invoke([HumanMessage(content="hi")])


def test_no_fallback_returns_the_primary_itself():
    primary = _Fake(_reply("x", "gpt-6-luna"))
    assert models.with_fallback(primary, None) is primary


def test_tools_are_bound_to_each_model_before_they_are_wrapped():
    primary, fallback = _Fake(_reply("p", "gpt-6-luna")), _Fake(_reply("f", "claude-haiku-5-5"))
    tools = ["tool-a", "tool-b"]
    by_name = {"gpt-6-luna": primary, "claude-haiku-5-5": fallback}

    with patch("models.build_chat_model", side_effect=lambda name, **_: by_name[name]):
        models.build_runnable("gpt-6-luna", "claude-haiku-5-5", tools=tools)

    assert primary.bound == tools and fallback.bound == tools


def test_a_fallback_equal_to_the_primary_is_ignored():
    only = _Fake(_reply("x", "gpt-6-luna"))
    with patch("models.build_chat_model", return_value=only):
        built = models.build_runnable("gpt-6-luna", "gpt-6-luna")

    assert built is only


# ---------------------------------------------------------------------------
# The fallback inside the real graph: whole turn, priced at the model that answered
# ---------------------------------------------------------------------------


def _initial_state(text):
    return {
        "messages": [HumanMessage(content=text)],
        "next_step": "",
        "search_query": "",
        "route": "",
        "tool_result": "",
        "context_used": "",
        "context_chunks": [],
        "tool_error": "",
        "node_latencies": {},
        "faithfulness_score": None,
        "retry_count": 0,
        "trajectory": [],
        "agent_rounds": 0,
        "token_usage": {},
    }


def _graph(primary_error, **kwargs):
    primary = _Fake(error=primary_error) if primary_error else _Fake(_reply("from primary", "gpt-6-luna"))
    fallback = _Fake(_reply("from fallback", "claude-haiku-5-5"))
    by_name = {"gpt-6-luna": primary, "claude-haiku-5-5": fallback}
    with (
        patch("models.build_chat_model", side_effect=lambda name, **_: by_name[name]),
        patch("chatbot.classify_query", return_value=("conversation", {"input": 0, "output": 0})),
        patch("chatbot.score_faithfulness_with_usage", return_value=(None, {"input": 0, "output": 0})),
    ):
        return chatbot.create_assistant(mode="classifier", model="gpt-6-luna", **kwargs), primary, fallback


def test_a_turn_survives_the_primary_failing_and_is_priced_at_the_fallbacks_rate():
    graph, _, _ = _graph(RuntimeError("quota"), fallback_model="claude-haiku-5-5")

    with (
        patch("chatbot.classify_query", return_value=("conversation", {"input": 0, "output": 0})),
        patch("chatbot.score_faithfulness_with_usage", return_value=(None, {"input": 0, "output": 0})),
    ):
        state = graph.invoke(_initial_state("Hello"))

    assert state["messages"][-1].content == "from fallback"
    assert state["token_usage"]["generate_response"]["model"] == "claude-haiku-5-5"
    dollars, _, _ = graph_cost(state["token_usage"], "gpt-6-luna")
    assert dollars == pytest.approx(estimate_cost("claude-haiku-5-5", 10, 3))


def test_without_a_fallback_the_primary_failure_surfaces():
    graph, _, _ = _graph(RuntimeError("quota"))

    with (
        patch("chatbot.classify_query", return_value=("conversation", {"input": 0, "output": 0})),
        patch("chatbot.score_faithfulness_with_usage", return_value=(None, {"input": 0, "output": 0})),
        pytest.raises(RuntimeError, match="quota"),
    ):
        graph.invoke(_initial_state("Hello"))


def test_the_evaluation_path_never_gets_a_fallback_even_when_one_is_configured(monkeypatch):
    monkeypatch.setenv("FALLBACK_CHAT_MODEL", "claude-haiku-5-5")
    primary = _Fake(error=RuntimeError("quota"))
    fallback = _Fake(_reply("from fallback", "claude-haiku-5-5"))
    by_name = {"gpt-6-luna": primary, "claude-haiku-5-5": fallback}

    with (
        patch("models.build_chat_model", side_effect=lambda name, **_: by_name[name]),
        patch("chatbot.classify_query", return_value=("conversation", {"input": 0, "output": 0})),
        patch("chatbot.score_faithfulness_with_usage", return_value=(None, {"input": 0, "output": 0})),
    ):
        graph = chatbot.create_assistant(mode="classifier", model="gpt-6-luna", use_fallback=False)
        with pytest.raises(RuntimeError, match="quota"):
            graph.invoke(_initial_state("Hello"))

    assert fallback.calls == 0


def test_the_configured_fallback_is_used_when_none_is_passed(monkeypatch):
    monkeypatch.setenv("FALLBACK_CHAT_MODEL", "claude-haiku-5-5")
    primary = _Fake(error=RuntimeError("quota"))
    fallback = _Fake(_reply("from fallback", "claude-haiku-5-5"))
    by_name = {"gpt-6-luna": primary, "claude-haiku-5-5": fallback}

    with (
        patch("models.build_chat_model", side_effect=lambda name, **_: by_name[name]),
        patch("chatbot.classify_query", return_value=("conversation", {"input": 0, "output": 0})),
        patch("chatbot.score_faithfulness_with_usage", return_value=(None, {"input": 0, "output": 0})),
    ):
        state = chatbot.create_assistant(mode="classifier", model="gpt-6-luna").invoke(
            _initial_state("Hello")
        )

    assert state["messages"][-1].content == "from fallback"


def test_a_claude_block_list_answer_reaches_the_history_as_plain_text():
    block_reply = AIMessage(
        content=[{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "Plain."}],
        usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        response_metadata={"model_name": "claude-haiku-5-5"},
    )
    claude = _Fake(block_reply)

    with (
        patch("models.build_chat_model", return_value=claude),
        patch("chatbot.classify_query", return_value=("conversation", {"input": 0, "output": 0})),
        patch("chatbot.score_faithfulness_with_usage", return_value=(None, {"input": 0, "output": 0})),
    ):
        state = chatbot.create_assistant(
            mode="classifier", model="claude-haiku-5-5", use_fallback=False
        ).invoke(_initial_state("Hello"))

    assert state["messages"][-1].content == "Plain."


# ---------------------------------------------------------------------------
# Which model answered: names, usage and cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-6-luna", "gpt-6-luna"),
        ("claude-haiku-5-5", "claude-haiku-5-5"),
        ("gpt-4o-mini-extra", "gpt-4o-mini"),
        ("gpt-4o-minimal", None),
        ("some-future-model", None),
        (None, None),
    ],
)
def test_a_reported_model_name_maps_to_its_pricing_key(reported, expected):
    assert canonical_model(reported) == expected


def test_usage_names_the_model_that_answered_when_the_response_says_so():
    response = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        response_metadata={"model_name": "gpt-4o-mini-2024-07-18"},
    )
    assert usage_from_response(response) == {"input": 12, "output": 3, "model": "gpt-4o-mini"}


def test_anthropic_metadata_uses_the_model_key():
    response = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        response_metadata={"model": "claude-haiku-5-5"},
    )
    assert usage_from_response(response)["model"] == "claude-haiku-5-5"


def test_claude_haiku_is_priced_at_its_published_rate():
    assert estimate_cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)


def test_graph_cost_prices_each_node_at_its_own_model_and_falls_back_to_the_default():
    usage = {
        "route_query": {"input": 1000, "output": 10, "model": "claude-haiku-5-5"},
        "generate_response": {"input": 2000, "output": 100},
        "check_faithfulness": {"input": 0, "output": 0},
    }

    dollars, total_in, total_out = graph_cost(usage, "gpt-6-luna")

    assert (total_in, total_out) == (3000, 110)
    assert dollars == pytest.approx(
        estimate_cost("claude-haiku-5-5", 1000, 10) + estimate_cost("gpt-6-luna", 2000, 100)
    )


def test_add_usage_remembers_which_model_answered_a_node():
    merged = chatbot._add_usage(
        {}, "generate_response", {"input": 5, "output": 1, "model": "claude-haiku-5-5"}
    )
    merged = chatbot._add_usage(
        merged, "generate_response", {"input": 5, "output": 1, "model": "claude-haiku-5-5"}
    )

    assert merged["generate_response"] == {"input": 10, "output": 2, "model": "claude-haiku-5-5"}


def test_add_usage_without_a_model_keeps_the_old_shape():
    assert chatbot._add_usage({}, "route_query", {"input": 5, "output": 1}) == {
        "route_query": {"input": 5, "output": 1}
    }


def test_a_response_with_no_model_is_stamped_with_the_name_of_the_model_that_gave_it():
    """A streamed Claude response reports no model; cost attribution needs one."""
    unnamed = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        response_metadata={},
    )
    runnable = models.with_fallback(
        _Fake(error=RuntimeError("down")),
        _Fake(unnamed),
        primary_name="gpt-6-luna",
        fallback_name="claude-haiku-5-5",
    )

    answer = runnable.invoke([HumanMessage(content="hi")])

    assert usage_from_response(answer)["model"] == "claude-haiku-5-5"


def test_a_model_the_response_already_names_is_not_overwritten():
    named = _reply("x", "gpt-6-luna")
    runnable = models.with_fallback(
        _Fake(named), _Fake(_reply("y", "claude-haiku-5-5")), primary_name="gpt-4o-mini"
    )

    assert usage_from_response(runnable.invoke([HumanMessage(content="hi")]))["model"] == "gpt-6-luna"
