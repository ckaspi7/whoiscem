"""ToolResult: the type that keeps a tool failure out of the model's context.

Every tool used to return f"Error: {e}" as an ordinary string on failure,
indistinguishable from real content, and chatbot.py fed that string to both the
model and the faithfulness judge as if it were retrieved evidence.
"""

from __future__ import annotations

from tools.result import ToolResult


def test_success_carries_content_and_no_error():
    result = ToolResult.success("Cem works at TELUS.")
    assert result.ok is True
    assert result.content == "Cem works at TELUS."
    assert result.error == ""


def test_failure_carries_an_error_and_no_content():
    result = ToolResult.failure("connection refused")
    assert result.ok is False
    assert result.content == ""
    assert result.error == "connection refused"


def test_as_context_returns_content_on_success():
    assert ToolResult.success("real content").as_context() == "real content"


def test_as_context_is_empty_on_failure():
    """The one property this whole type exists for: an error is never context."""
    assert ToolResult.failure("boom").as_context() == ""


def test_as_tool_string_returns_content_on_success():
    assert ToolResult.success("real content").as_tool_string("Error") == "real content"


def test_as_tool_string_prefixes_the_error_on_failure():
    result = ToolResult.failure("connection refused")
    assert result.as_tool_string("Error searching resume") == "Error searching resume: connection refused"
