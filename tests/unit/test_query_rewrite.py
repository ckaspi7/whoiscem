"""Condensing a follow-up into a standalone question."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from query_rewrite import MAX_HISTORY_TURNS, condense_query, reformulate_for_retry


def _llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=reply)
    return llm


def test_a_first_turn_is_returned_unchanged_without_calling_the_model():
    """No history means nothing to resolve, so the call is pure cost."""
    llm = _llm("should not be used")
    query, usage = condense_query("Where does Cem work?", [], llm)
    assert query == "Where does Cem work?"
    assert usage == {"input": 0, "output": 0}
    llm.invoke.assert_not_called()


def test_a_follow_up_is_rewritten_using_the_history():
    history = [
        HumanMessage(content="What startup did Cem co-found?"),
        AIMessage(content="NeoWise, a wearable thermal-device startup."),
    ]
    llm = _llm("How long was Cem at NeoWise?")
    query, _usage = condense_query("How long was that?", history, llm)
    assert query == "How long was Cem at NeoWise?"


def test_surrounding_quotes_are_stripped():
    history = [HumanMessage(content="Which university?")]
    query, _usage = condense_query("And there?", history, _llm('"What did Cem study at UBC?"'))
    assert query == "What did Cem study at UBC?"


def test_a_model_failure_falls_back_to_the_original():
    """A rewrite failure must not take the turn down with it."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("upstream down")
    history = [HumanMessage(content="Where does Cem work?")]
    query, usage = condense_query("And before that?", history, llm)
    assert query == "And before that?"
    assert usage == {"input": 0, "output": 0}


def test_an_empty_rewrite_falls_back_to_the_original():
    history = [HumanMessage(content="Where does Cem work?")]
    query, _usage = condense_query("And before that?", history, _llm("   "))
    assert query == "And before that?"


def test_an_answer_shaped_rewrite_is_rejected():
    """A model that answers instead of rewriting would poison routing and retrieval."""
    history = [HumanMessage(content="Where does Cem work?")]
    essay = "Cem works at TELUS Communications in Vancouver, where he " + ("builds systems. " * 40)
    query, _usage = condense_query("And before that?", history, _llm(essay))
    assert query == "And before that?"


def test_returns_the_real_token_usage():
    history = [HumanMessage(content="Where does Cem work?")]
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="How long was Cem at NeoWise?",
        usage_metadata={"input_tokens": 60, "output_tokens": 8, "total_tokens": 68},
    )
    _query, usage = condense_query("How long was that?", history, llm)
    assert usage == {"input": 60, "output": 8}


def test_only_recent_turns_are_sent():
    history = [HumanMessage(content=f"question {i}") for i in range(40)]
    llm = _llm("standalone question")
    condense_query("and then?", history, llm)

    prompt_value = llm.invoke.call_args.args[0]
    sent = "\n".join(m.content for m in prompt_value.messages)
    assert "question 39" in sent, "the most recent turn must be included"
    assert "question 0" not in sent, "old turns must be dropped"
    # Count the history lines only: the instructions themselves say "question".
    assert sent.count("User: question ") <= MAX_HISTORY_TURNS


def test_a_message_with_no_text_content_is_treated_as_no_history():
    """Multimodal LangChain messages can carry list content instead of a string;
    it should be skipped rather than crash the prompt formatter."""
    llm = _llm("unused")
    history = [AIMessage(content=[{"type": "text", "text": "part"}])]
    query, _usage = condense_query("Where does Cem work?", history, llm)
    assert query == "Where does Cem work?"
    llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# reformulate_for_retry — Phase 3.3's self-correction lever
# ---------------------------------------------------------------------------


def test_reformulate_for_retry_returns_the_rewritten_query():
    query, usage = reformulate_for_retry(
        "How long was Cem at NeoWise?",
        _llm("What was the duration of Cem's time at the NeoWise startup?"),
    )
    assert query == "What was the duration of Cem's time at the NeoWise startup?"
    assert usage == {"input": 0, "output": 0}  # _llm's bare mock has no usage_metadata configured


def test_reformulate_for_retry_strips_surrounding_quotes():
    query, _usage = reformulate_for_retry("How long?", _llm('"How long did the NeoWise role last?"'))
    assert query == "How long did the NeoWise role last?"


def test_reformulate_for_retry_falls_back_to_the_original_on_failure():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("upstream down")
    query, usage = reformulate_for_retry("How long was Cem at NeoWise?", llm)
    assert query == "How long was Cem at NeoWise?"
    assert usage == {"input": 0, "output": 0}


def test_reformulate_for_retry_falls_back_on_an_empty_rewrite():
    query, _usage = reformulate_for_retry("How long was Cem at NeoWise?", _llm("   "))
    assert query == "How long was Cem at NeoWise?"


def test_reformulate_for_retry_returns_the_real_token_usage():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="a broader query", usage_metadata={"input_tokens": 40, "output_tokens": 6, "total_tokens": 46}
    )
    _query, usage = reformulate_for_retry("original query", llm)
    assert usage == {"input": 40, "output": 6}


def test_a_raw_generator_can_no_longer_reach_history_at_all():
    """This used to be a defensive test: a live streaming generator briefly sat
    in message content mid-turn, and _format_history had to skip it rather than
    crash. GraphState no longer allows that state to exist — BaseMessage
    rejects non-string, non-list content at construction — so the bug this
    guarded against is now impossible by construction, not just handled."""
    with pytest.raises(Exception, match="valid string"):
        AIMessage(content=object())  # type: ignore[arg-type]
