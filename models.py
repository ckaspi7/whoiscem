"""Chat-model construction for both providers, and what makes them interchangeable.

The graphs, the router, the summariser and the evaluation harness all need "a chat
model" without caring whether OpenAI or Anthropic serves it. This module is the one
place that knows the difference:

- which provider a model name belongs to, and the settings each model requires;
- how to wrap a primary model with a second one that answers when the first fails;
- how to read a response as plain text, because Claude can return a list of content
  blocks (thinking plus text) where OpenAI returns a string.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

ANTHROPIC_PREFIX = "claude-"

# Models that reject any non-default sampling temperature outright, so they are built
# without one. gpt-6-luna answers 400 "Unsupported value: 'temperature' does not
# support 0.0 with this model"; Claude Haiku 5.5 accepts only 1 (or omission) per its
# migration guide. gpt-4o-mini has no such restriction.
FIXED_TEMPERATURE_MODELS = frozenset({"gpt-6-luna", "claude-haiku-5-5"})

# One Claude call's output ceiling. Thinking is off (below), so only the answer counts.
ANTHROPIC_MAX_TOKENS = 2048

# Thinking is switched off for Claude so it is on an equal footing with gpt-4o-mini and
# with gpt-6-luna at reasoning_effort "none": no hidden reasoning tokens to pay or wait
# for on routing, condensing and answering, and no thinking blocks to replay in a tool
# loop. Haiku 5.5 accepts {"type": "disabled"} (Models API capabilities) and does NOT
# accept the old manual "enabled" mode.
ANTHROPIC_THINKING = {"type": "disabled"}


def provider_of(model: str) -> str:
    return "anthropic" if model.startswith(ANTHROPIC_PREFIX) else "openai"


def build_chat_model(model: str, *, temperature: float = 0.0, streaming: bool = False) -> Any:
    """One LangChain chat model for `model`, with the settings that model requires.

    `temperature` is ignored for the models in FIXED_TEMPERATURE_MODELS. `streaming`
    only matters for usage reporting: without stream_usage a streaming OpenAI call
    returns no token counts at all, so it is always set.
    """
    if provider_of(model) == "anthropic":
        return _build_anthropic(model, streaming)

    kwargs: dict[str, Any] = {"model": model, "streaming": streaming, "stream_usage": True}
    if model not in FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature
    if model == "gpt-6-luna":
        # Chat Completions function calling only works at "none" (tool selection breaks
        # silently otherwise), and nothing here benefits from reasoning: routing and
        # condensing are short prompts and answering is grounded in supplied context.
        kwargs["reasoning_effort"] = "none"
    return ChatOpenAI(**kwargs)


def _build_anthropic(model: str, streaming: bool) -> Any:
    from langchain_anthropic import ChatAnthropic

    # The Anthropic SDK also honours ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN. A
    # gateway configured for some other tool would receive this project's key (or
    # authenticate this project's calls as that tool), so say so loudly.
    if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        logger.warning(
            "ANTHROPIC_BASE_URL or ANTHROPIC_AUTH_TOKEN is set in the environment: Claude calls "
            "will go to that endpoint or use that token, not just ANTHROPIC_API_KEY."
        )
    return ChatAnthropic(
        model_name=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        streaming=streaming,
        stream_usage=True,
        thinking=ANTHROPIC_THINKING,
    )


def with_fallback(
    primary: Runnable, fallback: Runnable | None, *, primary_name: str = "", fallback_name: str = ""
) -> Runnable:
    """`primary`, then `fallback` if the primary raises anything at all.

    Every switch is logged as a WARNING with the primary's error, because a silent
    fallback would hide a broken key or an exhausted balance behind answers that look
    fine. The evaluation harness builds without a fallback for the same reason: a run
    that quietly switched models would measure the wrong one.
    """
    if fallback is None:
        return primary

    def _call(messages: Any, config: Any = None) -> Any:
        try:
            return primary.invoke(messages, config)
        except Exception as exc:
            logger.warning(
                "%s failed (%s: %s); answering with %s",
                primary_name or "primary model",
                type(exc).__name__,
                str(exc)[:200],
                fallback_name or "the fallback model",
            )
            raise

    return RunnableLambda(_call).with_fallbacks([fallback])


def build_runnable(
    primary: str,
    fallback: str | None = None,
    *,
    temperature: float = 0.0,
    streaming: bool = False,
    tools: list | None = None,
) -> Runnable:
    """A chat model for `primary`, answered by `fallback` when it fails.

    Tools are bound to each model before they are wrapped: the wrapper that adds the
    fallback has no bind_tools of its own.
    """

    def one(name: str) -> Any:
        model = build_chat_model(name, temperature=temperature, streaming=streaming)
        return model.bind_tools(tools) if tools else model

    return with_fallback(
        one(primary),
        one(fallback) if fallback and fallback != primary else None,
        primary_name=primary,
        fallback_name=fallback or "",
    )


def message_text(message: BaseMessage) -> str:
    """The text of a response, whether `content` is a string or a list of blocks."""
    # `.text` is a str subclass that is also callable; reading it as a string avoids the
    # deprecated method call. Stand-ins without it fall back to a string `content`.
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return str(text)
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


def as_plain_message(message: BaseMessage) -> BaseMessage:
    """The same message with list-of-blocks `content` flattened to its text.

    A message that is asking for tool calls is left alone: its blocks are what the
    provider expects back on the next round. Everything the UI and the stored history
    see should be a string.
    """
    if isinstance(message, AIMessage) and isinstance(message.content, list) and not message.tool_calls:
        return message.model_copy(update={"content": message_text(message)})
    return message
