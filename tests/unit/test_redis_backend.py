"""A Redis URL carries a password, so redis_backend must never write one to the logs."""

from __future__ import annotations

import logging

import pytest

from redis_backend import BACKEND_IN_PROCESS, connect, redact_url

SECRET = "s3cr3t-p4ss"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            f"rediss://default:{SECRET}@example-host.upstash.io:6379/0",
            "rediss://example-host.upstash.io:6379/0",
        ),
        ("redis://localhost:6379", "redis://localhost:6379"),
        (f"redis://:{SECRET}@cache.internal/2", "redis://cache.internal/2"),
        (f"redis://user:{SECRET}@[::1]:6380/1?ssl_cert_reqs=none", "redis://[::1]:6380/1"),
        (f"unix://:{SECRET}@/run/redis.sock?db=1", "unix:///run/redis.sock"),
    ],
)
def test_redact_url_keeps_the_address_and_drops_everything_else(url, expected):
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        f"user:{SECRET}@host:6379",  # no scheme: urlsplit would call "user" the scheme and keep the rest
        SECRET,
        f"http://u:{SECRET}@host",
        f"redis://u:{SECRET}@host:notaport",
    ],
)
def test_redact_url_never_echoes_what_it_cannot_recognise(url):
    assert SECRET not in redact_url(url)


def test_a_redis_that_cannot_be_reached_is_logged_without_its_password(caplog):
    # Port 1 on loopback refuses at once, so this runs the real redis-py failure path
    # (including its own error text) without a server.
    url = f"redis://default:{SECRET}@127.0.0.1:1/0"

    with caplog.at_level(logging.WARNING, logger="redis_backend"):
        client, backend = connect(url, purpose="session memory")

    assert backend == BACKEND_IN_PROCESS and client is not None
    assert "Redis unreachable" in caplog.text
    assert SECRET not in caplog.text
    assert "127.0.0.1:1" in caplog.text, "the address stays in the warning, so an outage is still diagnosable"
