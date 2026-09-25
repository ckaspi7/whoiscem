"""Turn a follow-up into a question that stands on its own.

Retrieval ran on the bare last message, so "and before that?" was searched as
those three words — no employer, no dates, nothing to match. It is the classic
multi-turn RAG failure and it was live: follow-ups routed correctly 33% of the
time and never retrieved their reference.

Condensing happens before routing, so the classifier sees the real question too,
not just the router's guess at a pronoun.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from cost import usage_from_response

logger = logging.getLogger(__name__)

_NO_USAGE = {"input": 0, "output": 0}

MAX_HISTORY_TURNS = 6

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the user's latest message as a question that stands on its own,
using the conversation for any context it depends on.

Rules:
- Resolve pronouns and references: "there", "that", "he", "before that".
- Name Cem explicitly when the question is about him. A rewrite that drops the
  subject reads as a general-knowledge question and gets routed as one — "how
  long was NeoWise in operation" loses the fact that it is about his career.
- Keep the user's intent exactly. Do not answer, expand or narrow it.
- If the message is already self-contained, return it unchanged.
- Reply with the question only, no preamble.

Conversation so far:
{history}""",
        ),
        ("human", "{query}"),
    ]
)


_REFORMULATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The previous search for this question did not return a well-grounded
answer. Rewrite the search query to use different wording or a broader phrasing
that might match the resume's own terms more directly. Keep the same
underlying question — do not answer it, and do not narrow it.

Reply with the rewritten query only, no preamble.""",
        ),
        ("human", "{query}"),
    ]
)


def reformulate_for_retry(query: str, llm) -> tuple[str, dict[str, int]]:
    """Broaden or rephrase a query after a low-faithfulness resume answer.

    Never raises: a failed reformulation falls back to the original query,
    same discipline as condense_query. A no-op on this project's current
    corpus more often than not — `RETRIEVAL_STRATEGY=auto` already hands the
    model the whole resume once it fits in context, so a different query
    mostly just changes the rerank order rather than which chunks come back.
    It starts to matter for real once the corpus outgrows that ceiling.

    Returns (query, usage) — real token counts for this call (Phase 4.2),
    zero when the fallback path is taken since no call was made or none of it
    is usable.
    """
    try:
        response = llm.invoke(_REFORMULATE_PROMPT.invoke({"query": query}))
        rewritten = (response.content or "").strip().strip('"')
        usage = usage_from_response(response)
    except Exception as exc:
        logger.warning("Query reformulation failed, using the original: %s", exc)
        return query, dict(_NO_USAGE)
    return (rewritten or query), usage


def _format_history(messages: list) -> str:
    """Render recent turns for the prompt. Takes LangChain BaseMessage objects."""
    speaker = {"human": "User", "ai": "Assistant"}
    lines = []
    for message in messages[-MAX_HISTORY_TURNS:]:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            lines.append(f"{speaker.get(getattr(message, 'type', None), 'User')}: {content}")
    return "\n".join(lines)


def condense_query(query: str, history: list, llm) -> tuple[str, dict[str, int]]:
    """Rewrite `query` to stand alone. Returns it unchanged when it already does.

    Never raises: a condensation failure must degrade to the original question
    rather than take the turn down with it.

    Returns (query, usage) — real token counts for this call (Phase 4.2),
    zero on every path that never calls the model (no history, or a rejected
    rewrite).
    """
    formatted = _format_history(history)
    if not formatted:
        return query, dict(_NO_USAGE)

    try:
        response = llm.invoke(_CONDENSE_PROMPT.invoke({"history": formatted, "query": query}))
        rewritten = (response.content or "").strip().strip('"')
        usage = usage_from_response(response)
    except Exception as exc:
        logger.warning("Query condensation failed, using the original: %s", exc)
        return query, dict(_NO_USAGE)

    if not rewritten:
        return query, usage

    # A rewrite that balloons is usually the model answering rather than
    # rewriting, which would poison both routing and retrieval.
    if len(rewritten) > max(240, len(query) * 8):
        logger.warning("Condensation looks like an answer, not a question; using the original")
        return query, usage

    return rewritten, usage
