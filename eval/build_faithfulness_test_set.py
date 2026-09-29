"""Generate eval/faithfulness_test_set.json from the golden set.

Generated, not hand-authored — same discipline as build_golden_set.py. Rerun
this after the golden set or the underlying data (resume, seed_data) changes;
do not hand-edit the output.

For every answerable golden-set question outside the conversation route (which
has no real context to check faithfulness against), pairs the question's real,
production-retrieved context with two answers:

- The golden set's own hand-verified `ground_truth`, labeled faithful.
- An LLM-corrupted variant of it, labeled unfaithful: one additional specific
  detail (a date, a duration, a number, a qualifier) that sounds plausible but
  is not actually supported by the context. This mirrors a real bug found in
  production (guardrails/faithfulness_check.py let "...has beginner
  proficiency in Spanish" through when the context only said "Spanish") rather
  than testing against obvious, easy-to-catch fabrications.

personal/spotify/linkedin all return one fixed context blob regardless of the
question (see tools/personal_tool.py's get_personal_info_result() called with
no info_type, tools/spotify_tool.py, tools/linkedin_tool.py) - fetched once
per route and reused, not re-fetched per question.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI  # noqa: E402

from tools.linkedin_tool import get_linkedin_info_result  # noqa: E402
from tools.personal_tool import get_personal_info_result  # noqa: E402
from tools.resume_tool import get_resume_info_result  # noqa: E402
from tools.spotify_tool import get_music_taste_result  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
OUTPUT_PATH = Path(__file__).parent / "faithfulness_test_set.json"

# Golden-set entries whose ground_truth cannot be a faithful pairing no matter
# what context is fetched, found by actually running this script and checking
# why both the judge and HHEM scored several "faithful" pairs badly: p004/p006/
# p008 give a templated non-answer ("As recorded in his personal profile")
# instead of the real value, and p007 asks Cem's birth year, which
# tools/personal_tool.py's _SAFE_FIELDS deliberately excludes from context —
# pairing "1997." with a context that structurally cannot contain it is not a
# faithful example, it's a mislabeled one. Worth fixing golden_set.json's
# generator too, separately; excluded here so this test set isn't built on a
# false premise in the meantime.
_INVALID_GROUND_TRUTH_IDS = {"p004", "p006", "p007", "p008"}
_PLACEHOLDER_PHRASES = ("as recorded in", "as listed there", "drawn from his")

_CORRUPT_PROMPT = """You are creating test cases for a fact-checking system that catches AI \
hallucination in a RAG chatbot.

Given a QUESTION, its CONTEXT (the only true source of information), and a CORRECT ANSWER \
grounded in that context, rewrite the answer to add exactly ONE additional specific detail \
that sounds plausible and natural, but is NOT actually stated in or supported by the context \
(for example: a specific date, duration, number, reason, or qualifier). Keep the rest of the \
answer accurate and unchanged.

The goal is a realistic example of an AI assistant's subtle over-confidence -- not an obvious \
fabrication, not a wrong core fact, just one small unsupported elaboration added to an \
otherwise-correct answer.

QUESTION: {question}
CONTEXT: {context}
CORRECT ANSWER: {answer}

Return ONLY the rewritten answer text, nothing else -- no preamble, no quotes."""


def _route_context(route: str, question: str, cache: dict[str, str]) -> str | None:
    if route == "resume":
        result = get_resume_info_result(question)
        return result.content if result.ok else None
    if route not in cache:
        if route == "personal":
            result = get_personal_info_result()
        elif route == "spotify":
            result = get_music_taste_result()
        elif route == "linkedin":
            result = get_linkedin_info_result()
        else:
            return None
        cache[route] = result.content if result.ok else None
    return cache[route]


def main() -> None:
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    candidates = [
        q
        for q in golden
        if q.get("answerable")
        and q.get("ground_truth")
        and q.get("expected_route") != "conversation"
        and q["id"] not in _INVALID_GROUND_TRUTH_IDS
        and not any(p in q["ground_truth"].lower() for p in _PLACEHOLDER_PHRASES)
    ]
    print(f"{len(candidates)} candidate questions (of {len(golden)} total)")

    corrupter = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    route_context_cache: dict[str, str] = {}
    pairs = []

    for q in candidates:
        route = q["expected_route"]
        context = _route_context(route, q["question"], route_context_cache)
        if not context:
            print(f"  skip {q['id']}: no context available ({route})")
            continue

        answer = q["ground_truth"]
        corrupted = corrupter.invoke(
            _CORRUPT_PROMPT.format(question=q["question"], context=context, answer=answer)
        ).content.strip()

        pairs.append(
            {
                "id": f"{q['id']}-faithful",
                "source_id": q["id"],
                "route": route,
                "context": context,
                "answer": answer,
                "faithful": True,
            }
        )
        pairs.append(
            {
                "id": f"{q['id']}-unfaithful",
                "source_id": q["id"],
                "route": route,
                "context": context,
                "answer": corrupted,
                "faithful": False,
            }
        )
        print(f"  {q['id']} ({route}): ok")

    OUTPUT_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    half = len(pairs) // 2
    print(f"\nWrote {len(pairs)} pairs ({half} faithful / {half} unfaithful) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
