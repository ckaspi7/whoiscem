from __future__ import annotations

from unittest.mock import MagicMock, patch


def _mock_hhem(raw_score: float) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = [raw_score]
    return model


def test_high_score_returns_answer_unchanged():
    with patch("guardrails.faithfulness_check._get_hhem", return_value=_mock_hhem(0.9)):
        from guardrails.faithfulness_check import check_faithfulness

        result = check_faithfulness("Cem works at TELUS.", "Cem is an AI/ML Engineer at TELUS.")

    assert result == "Cem works at TELUS."


def test_mid_score_prepends_warning():
    with patch("guardrails.faithfulness_check._get_hhem", return_value=_mock_hhem(0.25)):
        from guardrails.faithfulness_check import check_faithfulness

        result = check_faithfulness("Some answer.", "Some context.")

    assert "⚠️" in result
    assert "Some answer." in result


def test_low_score_replaces_answer():
    with patch("guardrails.faithfulness_check._get_hhem", return_value=_mock_hhem(0.02)):
        from guardrails.faithfulness_check import check_faithfulness

        result = check_faithfulness("Made up answer.", "Unrelated context.")

    assert "reliable" in result.lower() or "don't have" in result.lower()
    assert "Made up answer." not in result


def test_model_load_failure_fails_open():
    with patch("guardrails.faithfulness_check._get_hhem", side_effect=RuntimeError("model unavailable")):
        from guardrails.faithfulness_check import check_faithfulness

        result = check_faithfulness("original answer", "some context")

    assert result == "original answer"


def test_prediction_failure_fails_open():
    model = MagicMock()
    model.predict.side_effect = RuntimeError("boom")
    with patch("guardrails.faithfulness_check._get_hhem", return_value=model):
        from guardrails.faithfulness_check import check_faithfulness

        result = check_faithfulness("original answer", "some context")

    assert result == "original answer"


def test_empty_context_skips_check():
    with patch("guardrails.faithfulness_check._get_hhem") as mock_get_hhem:
        from guardrails.faithfulness_check import check_faithfulness

        result = check_faithfulness("answer text", "")

    mock_get_hhem.assert_not_called()
    assert result == "answer text"


def test_a_failed_check_is_logged(caplog):
    """Phase 4.3: a guardrail that can go silently inert is worse than none —
    this is what makes the fail-open path visible instead of mute."""
    with patch("guardrails.faithfulness_check._get_hhem", side_effect=RuntimeError("connection reset")):
        from guardrails.faithfulness_check import check_faithfulness

        with caplog.at_level("WARNING"):
            check_faithfulness("answer", "context")

    assert "failing open" in caplog.text.lower()


# ---------------------------------------------------------------------------
# _hhem_to_five_scale — the mapping from HHEM's continuous 0-1 score onto the
# 1-5 scale every other caller (tiering, chatbot.py's in-graph retry check,
# eval/run_eval.py) already gates on. Boundaries from
# guardrails/faithfulness_check.py's own module comment: refuse below 0.15,
# warn in [0.15, 0.40), pass at 0.40 and above.
# ---------------------------------------------------------------------------


def test_hhem_scale_boundaries():
    from guardrails.faithfulness_check import _hhem_to_five_scale

    assert _hhem_to_five_scale(0.0) == 1
    assert _hhem_to_five_scale(0.14) == 1
    assert _hhem_to_five_scale(0.15) == 3  # boundary is inclusive on the warn side
    assert _hhem_to_five_scale(0.39) == 3
    assert _hhem_to_five_scale(0.40) == 5  # boundary is inclusive on the pass side
    assert _hhem_to_five_scale(1.0) == 5


# ---------------------------------------------------------------------------
# score_faithfulness_with_usage / apply_faithfulness_tiering (Phase 4.2) —
# the split that lets a caller which already scored an answer (chatbot.py's
# in-graph retry check) apply the same tiering without a second model call,
# and lets a caller doing cost accounting see what the call actually cost.
# ---------------------------------------------------------------------------


def test_score_faithfulness_with_usage_reports_zero_usage_on_a_successful_call():
    """HHEM is a local model, not a priced API call — real cost accounting
    should show $0 for this line item now, not omit it or fake a token count."""
    with patch("guardrails.faithfulness_check._get_hhem", return_value=_mock_hhem(0.9)):
        from guardrails.faithfulness_check import score_faithfulness_with_usage

        score, usage = score_faithfulness_with_usage("answer", "context")

    assert score == 5
    assert usage == {"input": 0, "output": 0}


def test_score_faithfulness_with_usage_reports_zero_usage_on_empty_context():
    from guardrails.faithfulness_check import score_faithfulness_with_usage

    score, usage = score_faithfulness_with_usage("answer", "")
    assert score is None
    assert usage == {"input": 0, "output": 0}


def test_score_faithfulness_with_usage_reports_zero_usage_on_failure():
    with patch("guardrails.faithfulness_check._get_hhem", side_effect=RuntimeError("boom")):
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
