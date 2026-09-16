from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

_VALID_TYPES = frozenset({"resume", "personal", "spotify", "linkedin", "conversation"})

_ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user query into exactly one of these categories:
- resume: education, work experience, skills, projects
- personal: background, family, hobbies, languages, physical attributes
- spotify: music taste, favourite artists or songs
- linkedin: career development, promotions, job titles
- conversation: greetings, general chat, capabilities question

Reply with only the single category word."""),
    ("human", "{query}"),
])


def classify_query(query: str, llm) -> str:
    """Return one of: resume | personal | spotify | linkedin | conversation."""
    response = llm.invoke(_ROUTE_PROMPT.invoke({"query": query}))
    result = response.content.strip().lower()
    return result if result in _VALID_TYPES else "conversation"
