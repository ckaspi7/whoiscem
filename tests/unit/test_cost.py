from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cost import estimate_cost, usage_from_response


def test_gpt_4o_mini_pricing_matches_published_rates():
    # 1000 input + 1000 output tokens at $0.15/$0.60 per 1M = $0.00015 + $0.0006
    assert estimate_cost("gpt-4o-mini", 1000, 1000) == 0.00075


def test_gpt_6_luna_is_cheaper_than_gpt_4o_mini_at_equal_usage():
    mini = estimate_cost("gpt-4o-mini", 1000, 1000)
    luna = estimate_cost("gpt-6-luna", 1000, 1000)
    assert luna < mini


def test_zero_usage_costs_nothing():
    assert estimate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_an_unrecognised_model_falls_back_to_gpt_4o_mini_rates_not_zero():
    assert estimate_cost("some-future-model", 1000, 1000) == estimate_cost("gpt-4o-mini", 1000, 1000)


def test_input_and_output_tokens_are_priced_independently():
    input_only = estimate_cost("gpt-4o-mini", 1000, 0)
    output_only = estimate_cost("gpt-4o-mini", 0, 1000)
    assert input_only != output_only
    assert input_only + output_only == pytest.approx(estimate_cost("gpt-4o-mini", 1000, 1000))


# ---------------------------------------------------------------------------
# usage_from_response — the direct fix for the len(text) // 4 estimate, and
# its own negative control: a bare mock without usage_metadata configured
# must read as zero usage, not silently propagate a mock object into a sum.
# ---------------------------------------------------------------------------


def test_reads_real_usage_metadata():
    response = MagicMock(usage_metadata={"input_tokens": 142, "output_tokens": 18, "total_tokens": 160})
    assert usage_from_response(response) == {"input": 142, "output": 18}


def test_a_response_with_no_usage_metadata_reads_as_zero():
    response = MagicMock(usage_metadata=None)
    assert usage_from_response(response) == {"input": 0, "output": 0}


def test_a_bare_mock_with_usage_metadata_unconfigured_reads_as_zero_not_a_mock():
    """A MagicMock auto-creates `.usage_metadata` on access rather than
    raising — this is what a test double built for something else (only
    `.content` configured) looks like, and it must not be mistaken for real
    usage data that happens to be present."""
    response = MagicMock(content="some answer")
    usage = usage_from_response(response)
    assert usage == {"input": 0, "output": 0}
    assert isinstance(usage["input"], int)
