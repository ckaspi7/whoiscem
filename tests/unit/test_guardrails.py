from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


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


def test_judge_requests_json_object_response_format():
    """Phase 4.3: without this, a ```json fence around the reply breaks
    json.loads and the guard fails open silently — this is the fix, not just
    the fail-open path that made the symptom survivable."""
    client = _mock_openai_response(5)
    with patch("guardrails.faithfulness_check._get_client", return_value=client):
        from guardrails.faithfulness_check import check_faithfulness

        check_faithfulness("answer", "context")

    call_kwargs = client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"}


def test_a_failed_judge_call_is_logged(caplog):
    """Phase 4.3: a guardrail that can go silently inert is worse than none —
    this is what makes the fail-open path in test_malformed_json_fails_open
    visible instead of mute."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("connection reset")
    with patch("guardrails.faithfulness_check._get_client", return_value=client):
        from guardrails.faithfulness_check import check_faithfulness

        with caplog.at_level("WARNING"):
            check_faithfulness("answer", "context")

    assert "failing open" in caplog.text.lower()


# ---------------------------------------------------------------------------
# score_faithfulness_with_usage / apply_faithfulness_tiering (Phase 4.2) —
# the split that lets a caller which already scored an answer (chatbot.py's
# in-graph retry check) apply the same tiering without a second judge call,
# and lets a caller doing cost accounting see what the call actually cost.
# ---------------------------------------------------------------------------


def _mock_openai_response_with_usage(score: int, prompt_tokens: int, completion_tokens: int) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({"score": score, "reason": "test"})))],
        usage=MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    return client


def test_score_faithfulness_with_usage_returns_the_real_token_counts():
    client = _mock_openai_response_with_usage(5, prompt_tokens=142, completion_tokens=18)
    with patch("guardrails.faithfulness_check._get_client", return_value=client):
        from guardrails.faithfulness_check import score_faithfulness_with_usage

        score, usage = score_faithfulness_with_usage("answer", "context")

    assert score == 5
    assert usage == {"input": 142, "output": 18}


def test_score_faithfulness_with_usage_reports_zero_usage_on_empty_context():
    from guardrails.faithfulness_check import score_faithfulness_with_usage

    score, usage = score_faithfulness_with_usage("answer", "")
    assert score is None
    assert usage == {"input": 0, "output": 0}


def test_score_faithfulness_with_usage_reports_zero_usage_on_judge_failure():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    with patch("guardrails.faithfulness_check._get_client", return_value=client):
        from guardrails.faithfulness_check import score_faithfulness_with_usage

        score, usage = score_faithfulness_with_usage("answer", "context")

    assert score is None
    assert usage == {"input": 0, "output": 0}


def test_apply_faithfulness_tiering_matches_check_faithfulness_for_every_tier():
    from guardrails.faithfulness_check import apply_faithfulness_tiering

    assert apply_faithfulness_tiering("the answer", 5) == "the answer"
    assert apply_faithfulness_tiering("the answer", 4) == "the answer"
    assert apply_faithfulness_tiering("the answer", None) == "the answer"
    assert "⚠️" in apply_faithfulness_tiering("the answer", 3)
    assert "the answer" in apply_faithfulness_tiering("the answer", 2)
    assert apply_faithfulness_tiering("the answer", 1) != "the answer"
