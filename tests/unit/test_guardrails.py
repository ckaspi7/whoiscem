from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_openai_response(score: int, reason: str = "test") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({"score": score, "reason": reason})))]
    )
    return client


def test_high_score_returns_answer_unchanged():
    with patch("guardrails.faithfulness_check._get_client", return_value=_mock_openai_response(5)):
        from guardrails.faithfulness_check import check_faithfulness
        result = check_faithfulness("Cem works at TELUS.", "Cem is an AI/ML Engineer at TELUS.")

    assert result == "Cem works at TELUS."


def test_mid_score_prepends_warning():
    with patch("guardrails.faithfulness_check._get_client", return_value=_mock_openai_response(3)):
        from guardrails.faithfulness_check import check_faithfulness
        result = check_faithfulness("Some answer.", "Some context.")

    assert "⚠️" in result
    assert "Some answer." in result


def test_low_score_replaces_answer():
    with patch("guardrails.faithfulness_check._get_client", return_value=_mock_openai_response(1)):
        from guardrails.faithfulness_check import check_faithfulness
        result = check_faithfulness("Made up answer.", "Unrelated context.")

    assert "reliable" in result.lower() or "don't have" in result.lower()
    assert "Made up answer." not in result


def test_malformed_json_fails_open():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not valid json {{}"))]
    )
    with patch("guardrails.faithfulness_check._get_client", return_value=client):
        from guardrails.faithfulness_check import check_faithfulness
        result = check_faithfulness("original answer", "some context")

    assert result == "original answer"


def test_empty_context_skips_check():
    with patch("guardrails.faithfulness_check._get_client") as mock_client:
        from guardrails.faithfulness_check import check_faithfulness
        result = check_faithfulness("answer text", "")

    mock_client.assert_not_called()
    assert result == "answer text"


def test_judge_uses_gpt4o_mini():
    client = _mock_openai_response(5)
    with patch("guardrails.faithfulness_check._get_client", return_value=client):
        from guardrails.faithfulness_check import check_faithfulness
        check_faithfulness("answer", "context")

    call_kwargs = client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("model") == "gpt-4o-mini"
