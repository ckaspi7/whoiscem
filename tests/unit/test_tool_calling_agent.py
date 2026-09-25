"""The tool-calling graph (AGENT_MODE=tool_calling): bind tools, let the model
choose, loop back with results. Fully mocked — no real LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import chatbot
from tools.result import ToolResult

INITIAL_STATE = {
    "next_step": "",
    "search_query": "",
    "route": "",
    "tool_result": "",
    "context_used": "",
    "context_chunks": [],
    "tool_error": "",
    "node_latencies": {},
}


def _call(name: str, args: dict | None = None, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args or {}, "id": call_id}


def _mock_llm(*responses: AIMessage) -> MagicMock:
    """A ChatOpenAI stand-in whose bind_tools(...).invoke() replays `responses`
    in order — first any tool-calling rounds, then the final answer."""
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.side_effect = list(responses)
    return llm


def _run_graph(llm: MagicMock, question: str) -> dict:
    with patch("chatbot.ChatOpenAI", return_value=llm):
        graph = chatbot.create_assistant(mode="tool_calling")
    return graph.invoke({**INITIAL_STATE, "messages": [HumanMessage(content=question)]})


# ---------------------------------------------------------------------------
# _call_tool — the name-to-typed-result dispatcher
# ---------------------------------------------------------------------------


def test_call_tool_dispatches_resume_with_its_query_argument():
    with patch("chatbot.get_resume_info_result", return_value=ToolResult.success("resume content")) as fn:
        result = chatbot._call_tool("get_resume_info", {"query": "where does he work"})
    fn.assert_called_once_with("where does he work")
    assert result.content == "resume content"


def test_call_tool_dispatches_personal_with_its_info_type_argument():
    with patch("chatbot.get_personal_info_result", return_value=ToolResult.success("x")) as fn:
        chatbot._call_tool("get_personal_info", {"info_type": "hobbies"})
    fn.assert_called_once_with("hobbies")


def test_call_tool_dispatches_zero_argument_tools():
    with patch("chatbot.get_music_taste_result", return_value=ToolResult.success("x")) as fn:
        chatbot._call_tool("get_music_taste", {})
    fn.assert_called_once_with()


def test_call_tool_rejects_an_unknown_name():
    result = chatbot._call_tool("not_a_real_tool", {})
    assert result.ok is False
    assert "not_a_real_tool" in result.error


# ---------------------------------------------------------------------------
# The graph loop
# ---------------------------------------------------------------------------


def test_a_single_tool_call_is_executed_and_the_agent_answers_from_it():
    ai_calls_tool = AIMessage(content="", tool_calls=[_call("get_resume_info", {"query": "work"})])
    final = AIMessage(content="Cem works at TELUS.")
    llm = _mock_llm(ai_calls_tool, final)

    with patch("chatbot.get_resume_info_result", return_value=ToolResult.success("Cem works at TELUS.")):
        state = _run_graph(llm, "Where does Cem work?")

    assert state["route"] == "resume"
    assert state["context_used"] == "Cem works at TELUS."
    assert state["tool_error"] == ""
    assert state["messages"][-1].content == "Cem works at TELUS."


def test_no_tool_call_is_recorded_as_conversation():
    final = AIMessage(content="Hello! How can I help?")
    state = _run_graph(_mock_llm(final), "Hi!")

    assert state["route"] == "conversation"
    assert state["context_used"] == ""


def test_two_tools_in_one_round_both_execute_and_accumulate():
    """The multi-intent case: one AIMessage can carry more than one tool_call."""
    ai_calls_both = AIMessage(
        content="",
        tool_calls=[
            _call("get_resume_info", {"query": "resume"}, "c1"),
            _call("get_linkedin_info", {}, "c2"),
        ],
    )
    final = AIMessage(content="They agree.")
    llm = _mock_llm(ai_calls_both, final)

    with (
        patch("chatbot.get_resume_info_result", return_value=ToolResult.success("resume says X")),
        patch("chatbot.get_linkedin_info_result", return_value=ToolResult.success("linkedin says X")),
    ):
        state = _run_graph(llm, "Compare his resume to his LinkedIn")

    assert set(state["route"].split(",")) == {"resume", "linkedin"}
    assert "resume says X" in state["context_used"]
    assert "linkedin says X" in state["context_used"]
    assert len(state["context_chunks"]) == 2


def test_a_failed_tool_call_never_reaches_context_but_the_agent_still_sees_it():
    """The agent should see the failure (it can react to it); context_used and
    the faithfulness judge must not."""
    ai_calls_tool = AIMessage(content="", tool_calls=[_call("get_resume_info", {"query": "work"})])
    final = AIMessage(content="I'm having trouble accessing that right now.")
    llm = _mock_llm(ai_calls_tool, final)

    with patch("chatbot.get_resume_info_result", return_value=ToolResult.failure("connection refused")):
        state = _run_graph(llm, "Where does Cem work?")

    assert state["context_used"] == ""
    assert "connection refused" in state["tool_error"]
    # The ToolMessage the agent itself saw DOES carry the error — that is the
    # point of it being a real agent observing a real failure.
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    assert any("connection refused" in m.content for m in tool_messages)


def test_a_second_round_of_tool_calls_accumulates_rather_than_overwrites():
    """Genuine multi-hop: two separate rounds of tool calls in the same turn."""
    round1 = AIMessage(content="", tool_calls=[_call("get_resume_info", {"query": "work"}, "c1")])
    round2 = AIMessage(content="", tool_calls=[_call("get_music_taste", {}, "c2")])
    final = AIMessage(content="done")
    llm = _mock_llm(round1, round2, final)

    with (
        patch("chatbot.get_resume_info_result", return_value=ToolResult.success("resume info")),
        patch("chatbot.get_music_taste_result", return_value=ToolResult.success("music info")),
    ):
        state = _run_graph(llm, "irrelevant, the mock script drives this")

    assert set(state["route"].split(",")) == {"resume", "spotify"}
    assert "resume info" in state["context_used"]
    assert "music info" in state["context_used"]


def test_unknown_tool_name_is_reported_as_a_tool_error_not_a_crash():
    ai_calls_bogus = AIMessage(content="", tool_calls=[_call("not_a_real_tool", {})])
    final = AIMessage(content="Something went wrong.")
    state = _run_graph(_mock_llm(ai_calls_bogus, final), "trigger a bogus tool call")

    assert "not_a_real_tool" in state["tool_error"]
    assert state["context_used"] == ""


# ---------------------------------------------------------------------------
# trajectory — Phase 3.5's raw material: the actual (round, tool, args, ok)
# sequence, which the accumulated `route` string alone cannot reconstruct.
# ---------------------------------------------------------------------------


def test_trajectory_records_each_call_in_a_round_with_its_category():
    ai_calls_both = AIMessage(
        content="",
        tool_calls=[
            _call("get_resume_info", {"query": "resume"}, "c1"),
            _call("get_linkedin_info", {}, "c2"),
        ],
    )
    final = AIMessage(content="They agree.")
    llm = _mock_llm(ai_calls_both, final)

    with (
        patch("chatbot.get_resume_info_result", return_value=ToolResult.success("resume says X")),
        patch("chatbot.get_linkedin_info_result", return_value=ToolResult.success("linkedin says X")),
    ):
        state = _run_graph(llm, "Compare his resume to his LinkedIn")

    trajectory = state["trajectory"]
    assert len(trajectory) == 2
    assert {e["category"] for e in trajectory} == {"resume", "linkedin"}
    assert all(e["round"] == 0 for e in trajectory)
    assert all(e["ok"] for e in trajectory)


def test_trajectory_carries_the_actual_arguments_a_tool_was_called_with():
    ai_calls_tool = AIMessage(content="", tool_calls=[_call("get_resume_info", {"query": "NeoWise"})])
    final = AIMessage(content="NeoWise was a startup.")
    with patch("chatbot.get_resume_info_result", return_value=ToolResult.success("NeoWise info")):
        state = _run_graph(_mock_llm(ai_calls_tool, final), "What was NeoWise?")

    assert state["trajectory"][0]["args"] == {"query": "NeoWise"}
    assert state["trajectory"][0]["tool"] == "get_resume_info"


def test_trajectory_numbers_rounds_in_order_across_multi_hop_calls():
    round1 = AIMessage(content="", tool_calls=[_call("get_resume_info", {"query": "work"}, "c1")])
    round2 = AIMessage(content="", tool_calls=[_call("get_music_taste", {}, "c2")])
    final = AIMessage(content="done")
    llm = _mock_llm(round1, round2, final)

    with (
        patch("chatbot.get_resume_info_result", return_value=ToolResult.success("resume info")),
        patch("chatbot.get_music_taste_result", return_value=ToolResult.success("music info")),
    ):
        state = _run_graph(llm, "irrelevant, the mock script drives this")

    trajectory = state["trajectory"]
    assert [e["round"] for e in trajectory] == [0, 1]
    assert [e["category"] for e in trajectory] == ["resume", "spotify"]


def test_trajectory_records_a_failed_call_as_not_ok():
    ai_calls_tool = AIMessage(content="", tool_calls=[_call("get_resume_info", {"query": "work"})])
    final = AIMessage(content="I'm having trouble accessing that right now.")
    with patch("chatbot.get_resume_info_result", return_value=ToolResult.failure("connection refused")):
        state = _run_graph(_mock_llm(ai_calls_tool, final), "Where does Cem work?")

    assert state["trajectory"][0]["ok"] is False


# ---------------------------------------------------------------------------
# Phase 4.2 — stream_usage=True on every construction, streaming or not.
# Confirmed directly: ChatOpenAI(streaming=True) without it returns no real
# usage_metadata at all, even for a call that produced a real answer, and
# defaults to False if not set explicitly.
# ---------------------------------------------------------------------------


def test_every_llm_construction_requests_stream_usage_in_classifier_mode():
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="classifier")

    assert llm_cls.call_args_list, "expected llm and llm_fast to be constructed"
    assert all(c.kwargs.get("stream_usage") is True for c in llm_cls.call_args_list)


def test_every_llm_construction_requests_stream_usage_in_tool_calling_mode():
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="tool_calling")

    assert all(c.kwargs.get("stream_usage") is True for c in llm_cls.call_args_list)


# ---------------------------------------------------------------------------
# CHAT_MODEL — gpt-6-luna only supports tool calling at reasoning_effort="none"
# ---------------------------------------------------------------------------


def test_gpt_6_luna_sets_reasoning_effort_none_on_the_tool_bound_agent_only():
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="tool_calling", model="gpt-6-luna")

    calls = llm_cls.call_args_list
    with_reasoning = [c for c in calls if "reasoning_effort" in c.kwargs]
    assert len(with_reasoning) == 1, "expected exactly one construction (the tool-bound agent) to set it"
    assert with_reasoning[0].kwargs["reasoning_effort"] == "none"
    # llm and llm_fast never bind a tool, so the gotcha does not apply to them.
    other_calls = [c for c in calls if c is not with_reasoning[0]]
    assert len(other_calls) == 2, "expected llm and llm_fast to also be constructed"
    assert all("reasoning_effort" not in c.kwargs for c in other_calls)


def test_gpt_4o_mini_never_sets_reasoning_effort():
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="tool_calling", model="gpt-4o-mini")

    assert all("reasoning_effort" not in c.kwargs for c in llm_cls.call_args_list)


def test_classifier_mode_never_sets_reasoning_effort_even_on_luna():
    """No tool is ever bound to a model in classifier mode, so the gotcha
    that applies to _build_tool_calling_graph's agent is moot here."""
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="classifier", model="gpt-6-luna")

    assert all("reasoning_effort" not in c.kwargs for c in llm_cls.call_args_list)
    assert all(c.kwargs.get("model") == "gpt-6-luna" for c in llm_cls.call_args_list)


# ---------------------------------------------------------------------------
# CHAT_MODEL — gpt-6-luna also rejects any non-default temperature outright
# (a live 400: "'temperature' does not support 0.0 ... Only the default (1)
# value is supported"), not just 0. Confirmed by actually running it, not
# documented anywhere in advance.
# ---------------------------------------------------------------------------


def test_gpt_6_luna_never_sets_a_temperature_override_in_classifier_mode():
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="classifier", model="gpt-6-luna")

    assert llm_cls.call_args_list, "expected llm and llm_fast to be constructed"
    assert all("temperature" not in c.kwargs for c in llm_cls.call_args_list)


def test_gpt_6_luna_never_sets_a_temperature_override_in_tool_calling_mode():
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="tool_calling", model="gpt-6-luna")

    assert all("temperature" not in c.kwargs for c in llm_cls.call_args_list)


def test_gpt_4o_mini_still_gets_its_usual_temperature_overrides():
    """Regression check: the gpt-6-luna carve-out must not silently drop
    temperature for the model that actually supports and relies on it —
    llm at 0.7 for prose variety, llm_fast at 0 for deterministic routing."""
    with patch("chatbot.ChatOpenAI") as llm_cls:
        chatbot.create_assistant(mode="classifier", model="gpt-4o-mini")

    temperatures = sorted(c.kwargs.get("temperature") for c in llm_cls.call_args_list)
    assert temperatures == [0, 0.7]
