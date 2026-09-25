"""Phase 4.4: nothing ever called logging.basicConfig, so every module's
`logger = logging.getLogger(__name__)` fed Python's default WARNING-only,
unstructured, uncorrelated stderr fallback. setup_logging fixes the handler;
set_session_id is the correlation id the plan calls out as existing and never
logged.
"""

from __future__ import annotations

import json
import logging

import pytest

import observability


@pytest.fixture(autouse=True)
def _reset_session_id():
    """The context var must not leak a session id from one test into the
    next — each test's expectations about what session_id appears (or does
    not) depend on starting from the same "nothing set yet" state."""
    observability._session_id.set("")
    yield
    observability._session_id.set("")


def _format(record: logging.LogRecord) -> dict:
    observability._SessionIdFilter().filter(record)
    return json.loads(observability._JsonFormatter().format(record))


def _record(msg: str = "hello", level: int = logging.INFO, **extra) -> logging.LogRecord:
    record = logging.LogRecord("some.logger", level, __file__, 1, msg, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formats_as_valid_json_with_the_core_fields():
    payload = _format(_record("hello world"))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "some.logger"
    assert "timestamp" in payload


def test_no_session_id_set_is_omitted_rather_than_empty_string():
    payload = _format(_record("hello"))
    assert "session_id" not in payload


def test_a_set_session_id_is_attached_as_the_correlation_field():
    observability.set_session_id("abc-123")
    payload = _format(_record("hello"))
    assert payload["session_id"] == "abc-123"


def test_extra_fields_are_carried_through_for_per_stage_timing():
    """This is what lets a single logger.info(..., extra={"node_latencies":
    ...}) call put per-stage retrieval timing into the actual logs, not just
    the interactive sidebar."""
    payload = _format(_record("turn completed", node_latencies={"route_query": 0.12}, route="resume"))
    assert payload["node_latencies"] == {"route_query": 0.12}
    assert payload["route"] == "resume"


def test_message_formatting_args_are_applied():
    record = logging.LogRecord("some.logger", logging.WARNING, __file__, 1, "failed: %s", ("boom",), None)
    payload = _format(record)
    assert payload["message"] == "failed: boom"


def test_exception_info_is_captured():
    try:
        raise ValueError("bad input")
    except ValueError:
        import sys

        record = logging.LogRecord("some.logger", logging.ERROR, __file__, 1, "it broke", (), sys.exc_info())
    payload = _format(record)
    assert "ValueError" in payload["exc_info"]
    assert "bad input" in payload["exc_info"]


def test_setup_logging_is_idempotent():
    observability._logging_configured = False
    root = logging.getLogger()
    before = len(root.handlers)

    observability.setup_logging()
    after_first = len(root.handlers)
    observability.setup_logging()
    after_second = len(root.handlers)

    assert after_first == before + 1
    assert after_second == after_first  # the second call added nothing more

    root.handlers = root.handlers[:before]  # leave the root logger as found
    observability._logging_configured = False
