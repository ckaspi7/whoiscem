from __future__ import annotations

import pytest

from spend_tracker import SpendTracker


@pytest.fixture
def tracker_with_fake_redis(fake_redis):
    tracker = SpendTracker.__new__(SpendTracker)
    tracker._redis = fake_redis
    return tracker


@pytest.fixture
def tracker_no_redis():
    tracker = SpendTracker.__new__(SpendTracker)
    tracker._redis = None
    return tracker


def test_a_fresh_day_starts_at_zero(tracker_with_fake_redis):
    assert tracker_with_fake_redis.today_total() == 0.0


def test_add_returns_the_new_running_total(tracker_with_fake_redis):
    assert tracker_with_fake_redis.add(0.05) == pytest.approx(0.05)
    assert tracker_with_fake_redis.add(0.03) == pytest.approx(0.08)


def test_today_total_reflects_accumulated_spend(tracker_with_fake_redis):
    tracker_with_fake_redis.add(1.23)
    tracker_with_fake_redis.add(0.10)
    assert tracker_with_fake_redis.today_total() == pytest.approx(1.33)


def test_different_calendar_days_are_independent_keys(tracker_with_fake_redis):
    tracker_with_fake_redis.add(1.0)
    key_today = tracker_with_fake_redis._key()
    key_other_day = tracker_with_fake_redis._key("2020-01-01")
    assert key_today != key_other_day
    assert tracker_with_fake_redis._redis.get(key_other_day) is None


def test_zero_or_negative_amounts_are_not_recorded(tracker_with_fake_redis):
    tracker_with_fake_redis.add(0.0)
    tracker_with_fake_redis.add(-5.0)
    assert tracker_with_fake_redis.today_total() == 0.0


# ---------------------------------------------------------------------------
# No Redis available — must degrade, never raise or block a request
# ---------------------------------------------------------------------------


def test_no_redis_reads_as_zero_not_an_error(tracker_no_redis):
    assert tracker_no_redis.today_total() == 0.0


def test_no_redis_add_does_not_raise(tracker_no_redis):
    assert tracker_no_redis.add(1.0) == 0.0


def test_a_broken_redis_client_fails_open_on_read(tracker_with_fake_redis):
    tracker_with_fake_redis._redis.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ConnectionError("down")
    )
    assert tracker_with_fake_redis.today_total() == 0.0


def test_a_broken_redis_client_fails_open_on_write(tracker_with_fake_redis):
    tracker_with_fake_redis._redis.incrbyfloat = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ConnectionError("down")
    )
    assert tracker_with_fake_redis.add(1.0) == 0.0  # falls back to today_total(), still zero
