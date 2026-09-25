"""Phase 3.3: the faithfulness score used to end at a text banner outside the
graph. Now, on the resume route only, a low score drives one bounded retry —
reformulate the query, re-run resume retrieval, regenerate — before falling
through to that same outer banner. Fully mocked: no real LLM or retrieval
calls, and no real judge call (score_faithfulness itself is patched directly;
its own correctness is covered by tests/unit/test_guardrails.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import chatbot
from retrieval.types import ScoredChunk

INITIAL_STATE = {
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
}

_CHUNK_1 = [ScoredChunk(chunk_id="1", text="Cem works at TELUS.", score=0.9)]
_CHUNK_2 = [ScoredChunk(chunk_id="2", text="Cem is an AI/ML Engineer at TELUS Communications.", score=0.9)]


def _mock_llm(*responses: AIMessage) -> MagicMock:
    """A ChatOpenAI stand-in whose .invoke() replays `responses` in order."""
    llm = MagicMock()
    llm.invoke.side_effect = list(responses)
    return llm


def _run_graph(question: str, route: str, scores: list, llm_responses: list[AIMessage], search_results=None):
    """Run the classifier graph for one turn with the LLM, router, faithfulness
    score, and resume search all under direct control."""
    with (
        patch("chatbot.ChatOpenAI", return_value=_mock_llm(*llm_responses)),
        patch("chatbot.classify_query", return_value=route),
        patch("chatbot.score_faithfulness", side_effect=scores),
        patch("chatbot.search_resume", side_effect=search_results or [_CHUNK_1]),
    ):
        graph = chatbot.create_assistant(mode="classifier")
        return graph.invoke({**INITIAL_STATE, "messages": [HumanMessage(content=question)]})


def test_a_high_score_ends_without_retrying():
    state = _run_graph(
        "Where does Cem work?",
        route="resume",
        scores=[5],
        llm_responses=[AIMessage(content="Cem works at TELUS.")],
    )

    assert state["retry_count"] == 0
    assert state["faithfulness_score"] == 5
    assert state["messages"][-1].content == "Cem works at TELUS."


def test_a_low_score_on_resume_triggers_exactly_one_retry():
    state = _run_graph(
        "Where does Cem work?",
        route="resume",
        scores=[2, 5],
        llm_responses=[
            AIMessage(content="a vague first answer"),
            AIMessage(content="Cem is an AI/ML Engineer at TELUS Communications."),
            AIMessage(content="a well-grounded second answer"),
        ],
        search_results=[_CHUNK_1, _CHUNK_2],
    )

    assert state["retry_count"] == 1
    assert state["faithfulness_score"] == 5
    assert state["messages"][-1].content == "a well-grounded second answer"
    assert state["context_chunks"] == [c.text for c in _CHUNK_2]


def test_retry_replaces_the_first_answer_rather_than_appending_a_second_message():
    state = _run_graph(
        "Where does Cem work?",
        route="resume",
        scores=[2, 5],
        llm_responses=[
            AIMessage(content="a vague first answer"),
            AIMessage(content="reformulated query"),
            AIMessage(content="a well-grounded second answer"),
        ],
        search_results=[_CHUNK_1, _CHUNK_2],
    )

    # One HumanMessage, one AIMessage — not two AIMessages from two rounds.
    assert len(state["messages"]) == 2
    assert state["messages"][-1].content == "a well-grounded second answer"


def test_reformulated_query_drives_the_retry_search():
    with (
        patch("chatbot.ChatOpenAI") as llm_cls,
        patch("chatbot.classify_query", return_value="resume"),
        patch("chatbot.score_faithfulness", side_effect=[2, 5]),
        patch("chatbot.search_resume", side_effect=[_CHUNK_1, _CHUNK_2]) as search,
    ):
        llm_cls.return_value.invoke.side_effect = [
            AIMessage(content="a vague first answer"),
            AIMessage(content="a broader resume search query"),
            AIMessage(content="a well-grounded second answer"),
        ]
        chatbot.create_assistant(mode="classifier").invoke(
            {**INITIAL_STATE, "messages": [HumanMessage(content="Where does Cem work?")]}
        )

    calls = [c.args[0] for c in search.call_args_list]
    assert calls == ["Where does Cem work?", "a broader resume search query"]


def test_still_low_after_one_retry_does_not_retry_again():
    state = _run_graph(
        "Where does Cem work?",
        route="resume",
        scores=[2, 2],
        llm_responses=[
            AIMessage(content="a vague first answer"),
            AIMessage(content="reformulated query"),
            AIMessage(content="still a vague second answer"),
        ],
        search_results=[_CHUNK_1, _CHUNK_2],
    )

    assert state["retry_count"] == 1
    assert state["faithfulness_score"] == 2
    assert state["messages"][-1].content == "still a vague second answer"


def test_a_low_score_on_a_non_resume_route_never_retries():
    """personal/spotify/linkedin are fixed lookups — the same content every
    time — so a retry there would spend two calls to reproduce one answer."""
    from tools.result import ToolResult

    with patch("chatbot.get_personal_info_result", return_value=ToolResult.success("Cem enjoys hiking.")):
        state = _run_graph(
            "What are Cem's hobbies?",
            route="personal",
            scores=[2],
            llm_responses=[AIMessage(content="an answer about hobbies")],
        )

    assert state["retry_count"] == 0
    assert state["messages"][-1].content == "an answer about hobbies"


def test_no_context_never_triggers_a_retry():
    """conversation has nothing retrieved to reformulate; score_faithfulness
    itself returns None for empty context (see test_guardrails.py) — here
    that contract is asserted at the graph level via _should_retry."""
    state = _run_graph(
        "Hello!",
        route="conversation",
        scores=[None],
        llm_responses=[AIMessage(content="Hi there!")],
    )

    assert state["retry_count"] == 0
    assert state["faithfulness_score"] is None
    assert state["messages"][-1].content == "Hi there!"
