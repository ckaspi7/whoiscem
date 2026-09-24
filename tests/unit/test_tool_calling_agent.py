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
