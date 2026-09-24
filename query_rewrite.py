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

logger = logging.getLogger(__name__)

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


def _format_history(messages: list) -> str:
    """Render recent turns for the prompt. Takes LangChain BaseMessage objects."""
    speaker = {"human": "User", "ai": "Assistant"}
    lines = []
    for message in messages[-MAX_HISTORY_TURNS:]:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            lines.append(f"{speaker.get(getattr(message, 'type', None), 'User')}: {content}")
    return "\n".join(lines)


def condense_query(query: str, history: list, llm) -> str:
    """Rewrite `query` to stand alone. Returns it unchanged when it already does.

    Never raises: a condensation failure must degrade to the original question
    rather than take the turn down with it.
    """
    formatted = _format_history(history)
    if not formatted:
        return query

    try:
        response = llm.invoke(_CONDENSE_PROMPT.invoke({"history": formatted, "query": query}))
        rewritten = (response.content or "").strip().strip('"')
    except Exception as exc:
        logger.warning("Query condensation failed, using the original: %s", exc)
        return query

    if not rewritten:
        return query

    # A rewrite that balloons is usually the model answering rather than
    # rewriting, which would poison both routing and retrieval.
    if len(rewritten) > max(240, len(query) * 8):
        logger.warning("Condensation looks like an answer, not a question; using the original")
        return query

    return rewritten
