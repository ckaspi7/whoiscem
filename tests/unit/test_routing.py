from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from router import classify_query


def _mock_llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=reply)
    return llm


@pytest.mark.parametrize(
    "query,llm_reply,expected",
    [
        ("What is your work experience?", "resume", "resume"),
        ("Where did you go to school?", "resume", "resume"),
        ("How old are you?", "personal", "personal"),
        ("What are your hobbies?", "personal", "personal"),
        ("What music do you listen to?", "spotify", "spotify"),
        ("Who is your favourite artist?", "spotify", "spotify"),
        ("Where do you work now?", "linkedin", "linkedin"),
        ("Tell me about your promotions.", "linkedin", "linkedin"),
        ("Hello!", "conversation", "conversation"),
        ("What can you do?", "conversation", "conversation"),
    ],
)
def test_routing_parametrized(query, llm_reply, expected):
    llm = _mock_llm(llm_reply)
    category, _usage = classify_query(query, llm)
    assert category == expected


def test_unknown_response_falls_back_to_conversation():
    llm = _mock_llm("weather")
    category, _usage = classify_query("What is the weather?", llm)
    assert category == "conversation"


def test_extra_whitespace_and_casing_normalised():
    llm = _mock_llm("  Resume  ")
    category, _usage = classify_query("Tell me about your CV.", llm)
    assert category == "resume"


def test_llm_called_with_the_query():
    llm = _mock_llm("personal")
    classify_query("How old are you?", llm)
    assert llm.invoke.call_count == 1


def test_returns_the_real_token_usage():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="resume", usage_metadata={"input_tokens": 50, "output_tokens": 2, "total_tokens": 52}
    )
    _category, usage = classify_query("What is your work experience?", llm)
    assert usage == {"input": 50, "output": 2}
