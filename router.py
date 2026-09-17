from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

_VALID_TYPES = frozenset({"resume", "personal", "spotify", "linkedin", "conversation"})

_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user query into exactly one of these categories.

Take the first category that fits; they are listed in priority order.

- resume: ANY factual question about Cem's career, education, employers, job
  titles, dates, skills, technologies, projects or accomplishments. The resume
  is the source of record for all work history. "Where does he work", "what is
  his job title", "when did he join TELUS" and "what did he study" are resume.
- linkedin: ONLY the progression through internal titles at TELUS, or the text
  of his LinkedIn headline and About section. Use it when the question is
  specifically about promotions or moving between internal roles. General
  career questions are resume, not linkedin.
- personal: non-work background — hometown, languages spoken, hobbies, food,
  appearance, marital status.
- spotify: music taste, artists, tracks, genres.
- conversation: greetings, small talk, questions about you or your abilities,
  and anything that falls outside the categories above.

Reply with only the single category word.""",
        ),
        ("human", "{query}"),
    ]
)


def classify_query(query: str, llm) -> str:
    """Return one of: resume | personal | spotify | linkedin | conversation."""
    response = llm.invoke(_ROUTE_PROMPT.invoke({"query": query}))
    result = response.content.strip().lower()
    return result if result in _VALID_TYPES else "conversation"
