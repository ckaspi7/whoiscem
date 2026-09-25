from __future__ import annotations

from rate_limit import RateLimiter


def test_allows_up_to_the_limit():
    limiter = RateLimiter(limit=3, window_seconds=60)
    assert limiter.allow("client-a", now=0)
    assert limiter.allow("client-a", now=1)
    assert limiter.allow("client-a", now=2)


def test_rejects_once_the_limit_is_reached():
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("client-a", now=0)
    assert limiter.allow("client-a", now=1)
    assert not limiter.allow("client-a", now=2)


def test_a_rejected_call_does_not_itself_count_toward_the_window():
    """A rejected call recording a hit would make every subsequent call
    within the window also rejected, permanently locking the key out."""
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.allow("client-a", now=0)
    assert not limiter.allow("client-a", now=1)
    assert not limiter.allow("client-a", now=2)
    # Once the first hit ages out, a fresh call is allowed again — it would
    # not be if the two earlier rejections had each added a hit.
    assert limiter.allow("client-a", now=61)


def test_old_hits_age_out_of_the_window():
    limiter = RateLimiter(limit=1, window_seconds=10)
    assert limiter.allow("client-a", now=0)
    assert not limiter.allow("client-a", now=5)
    assert limiter.allow("client-a", now=11)  # the hit at t=0 is now outside the window


def test_different_keys_are_independent():
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.allow("client-a", now=0)
    assert limiter.allow("client-b", now=0)
    assert not limiter.allow("client-a", now=1)


def test_retry_after_is_zero_when_not_limited():
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.retry_after("client-a", now=0) == 0.0


def test_retry_after_reports_time_until_the_oldest_hit_expires():
    limiter = RateLimiter(limit=1, window_seconds=60)
    limiter.allow("client-a", now=0)
    assert limiter.retry_after("client-a", now=10) == 50.0
