"""Condensing a follow-up into a standalone question."""

from __future__ import annotations

from unittest.mock import MagicMock

from query_rewrite import MAX_HISTORY_TURNS, condense_query


def _llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=reply)
    return llm


def test_a_first_turn_is_returned_unchanged_without_calling_the_model():
    """No history means nothing to resolve, so the call is pure cost."""
    llm = _llm("should not be used")
    assert condense_query("Where does Cem work?", [], llm) == "Where does Cem work?"
    llm.invoke.assert_not_called()


def test_a_follow_up_is_rewritten_using_the_history():
    history = [
        {"role": "human", "content": "What startup did Cem co-found?"},
        {"role": "ai", "content": "NeoWise, a wearable thermal-device startup."},
    ]
    llm = _llm("How long was Cem at NeoWise?")
    assert condense_query("How long was that?", history, llm) == "How long was Cem at NeoWise?"


def test_surrounding_quotes_are_stripped():
    history = [{"role": "human", "content": "Which university?"}]
    assert condense_query("And there?", history, _llm('"What did Cem study at UBC?"')) == (
        "What did Cem study at UBC?"
    )


def test_a_model_failure_falls_back_to_the_original():
    """A rewrite failure must not take the turn down with it."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("upstream down")
    history = [{"role": "human", "content": "Where does Cem work?"}]
    assert condense_query("And before that?", history, llm) == "And before that?"


def test_an_empty_rewrite_falls_back_to_the_original():
    history = [{"role": "human", "content": "Where does Cem work?"}]
    assert condense_query("And before that?", history, _llm("   ")) == "And before that?"


def test_an_answer_shaped_rewrite_is_rejected():
    """A model that answers instead of rewriting would poison routing and retrieval."""
    history = [{"role": "human", "content": "Where does Cem work?"}]
    essay = "Cem works at TELUS Communications in Vancouver, where he " + ("builds systems. " * 40)
    assert condense_query("And before that?", history, _llm(essay)) == "And before that?"


def test_only_recent_turns_are_sent():
    history = [{"role": "human", "content": f"question {i}"} for i in range(40)]
    llm = _llm("standalone question")
    condense_query("and then?", history, llm)

    prompt_value = llm.invoke.call_args.args[0]
    sent = "\n".join(m.content for m in prompt_value.messages)
    assert "question 39" in sent, "the most recent turn must be included"
    assert "question 0" not in sent, "old turns must be dropped"
    # Count the history lines only: the instructions themselves say "question".
    assert sent.count("User: question ") <= MAX_HISTORY_TURNS


def test_history_without_usable_text_is_treated_as_no_history():
    """A streaming generator sits in message content mid-turn; it is not history."""
    llm = _llm("unused")
    history = [{"role": "ai", "content": object()}]
    assert condense_query("Where does Cem work?", history, llm) == "Where does Cem work?"
    llm.invoke.assert_not_called()
