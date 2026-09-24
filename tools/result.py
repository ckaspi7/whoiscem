"""A typed result for the four tools that feed the assistant.

Every tool returned ``f"Error: {e}"`` as its ordinary return value on failure,
indistinguishable from real content. ``chatbot.py`` then assigned that string to
both ``tool_result`` and ``context_used``, so a database lock or a Qdrant outage
became the model's evidence, and the faithfulness judge scored the answer
against a stack trace instead of skipping the check.

A caller that needs the LangChain ``@tool`` string interface still gets one
(``as_tool_string``); a caller that needs to tell success from failure — every
handler in the graph — gets ``ToolResult`` and can keep an error out of context.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call. ``content`` and ``error`` are mutually
    meaningful: only one is populated, matching ``ok``."""

    ok: bool
    content: str = ""
    error: str = ""

    @classmethod
    def success(cls, content: str) -> ToolResult:
        return cls(ok=True, content=content)

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, error=error)

    def as_context(self) -> str:
        """What may reach the model as retrieved evidence. Never the error."""
        return self.content if self.ok else ""

    def as_tool_string(self, error_prefix: str) -> str:
        """The single string a LangChain ``@tool`` function must return.

        Used only by the tool-calling interface (bound to an LLM, or invoked
        directly in a test) where the caller cannot see a typed result. Graph
        handlers should call the ``*_result()`` function instead and use
        ``as_context()``, not this.
        """
        return self.content if self.ok else f"{error_prefix}: {self.error}"
