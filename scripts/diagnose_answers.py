"""Replay prompts through the real graph and show which layer refuses.

For each prompt this prints the route taken, how much context the model saw,
the raw HHEM score of the final answer, and a verdict naming the layer
responsible:

    degraded          a tool errored, so the model saw no evidence
    llm_refusal       the model itself said it lacks the information
    guardrail_refuse  HHEM tiering would replace the answer with a refusal
    guardrail_warn    HHEM tiering would prefix a low-confidence warning
    ok                none of the above

Run it once per Qdrant mode to separate an environment problem (retrieval
unreachable) from a product problem (routing, prompt, guardrail):

    python scripts/diagnose_answers.py
    $env:QDRANT_MODE = "embedded"; python scripts/diagnose_answers.py
    python scripts/diagnose_answers.py --prompts my_prompts.txt --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings  # noqa: E402  (after sys.path setup)
from guardrails.refusal import looks_like_refusal  # noqa: E402

# "expect" is a hand label of whether the knowledge base can answer the
# question today, so false refusals can be counted. Labels are grounded in
# data/resume.md and the key facts in chatbot._system_prompt.
STARTER_PROMPTS: list[dict] = [
    {"q": "Where does Cem work?", "expect": "answerable"},
    {"q": "What has he been listening to?", "expect": "answerable"},
    {"q": "What's on his LinkedIn?", "expect": "answerable"},
    {"q": "What did Cem study at university?", "expect": "answerable"},
    {"q": "What programming languages does Cem know?", "expect": "answerable"},
    {"q": "Has he worked with LangGraph?", "expect": "answerable"},
    {"q": "Tell me about the TELUS AI Financial Agent.", "expect": "answerable"},
    {"q": "Which cloud platforms is he experienced with?", "expect": "answerable"},
    {"q": "How many years of experience does he have?", "expect": "answerable"},
    {"q": "Does he have any startup experience?", "expect": "answerable"},
    {"q": "What is his experience evaluating LLM systems?", "expect": "answerable"},
    {"q": "Has he done any mentoring or teaching?", "expect": "answerable"},
    {"q": "Does he have experience with Terraform?", "expect": "answerable"},
    {"q": "What was his business impact at TELUS?", "expect": "answerable"},
    {"q": "Where is he based?", "expect": "answerable"},
    {"q": "What languages does he speak?", "expect": "answerable"},
    {
        "q": "And what was his minor?",
        "expect": "answerable",
        "history": [
            ("human", "Where did Cem study?"),
            ("ai", "Cem earned a BASc in Electrical Engineering from the University of British Columbia."),
        ],
    },
    {
        "q": "What tools did he use there?",
        "expect": "answerable",
        "history": [
            ("human", "Tell me about the eChange Bot."),
            ("ai", "The eChange Bot is a Google Chat copilot for network change management at TELUS."),
        ],
    },
    {"q": "Is he open to relocating?", "expect": "unanswerable"},
    {"q": "What are his salary expectations?", "expect": "unanswerable"},
    {"q": "What is his biggest weakness?", "expect": "unanswerable"},
    {"q": "What kind of role is he looking for next?", "expect": "unanswerable"},
    {"q": "How was this chatbot built?", "expect": "unanswerable"},
]

_REFUSING = {"degraded", "llm_refusal", "guardrail_refuse"}


def _initial_state(messages: list) -> dict:
    return {
        "messages": messages,
        "next_step": "",
        "search_query": "",
        "route": "",
        "tool_result": "",
        "context_used": "",
        "context_chunks": [],
        "tool_error": "",
        "node_latencies": {},
        "faithfulness_score": None,
        "retry_count": 0,
        "trajectory": [],
        "agent_rounds": 0,
        "token_usage": {},
    }


def _load_prompts(path: str | None, limit: int | None) -> list[dict]:
    if path:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        prompts = [
            {"q": line.strip(), "expect": "unknown"}
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        prompts = list(STARTER_PROMPTS)
    return prompts[:limit] if limit else prompts


def _probe_retrieval() -> str:
    from tools.resume_tool import search_resume

    started = time.perf_counter()
    try:
        chunks = search_resume("What did Cem study at university?")
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return f"FAILED after {elapsed:.1f}s: {type(exc).__name__}: {str(exc)[:240]}"
    return f"ok, {len(chunks)} chunk(s) in {time.perf_counter() - started:.1f}s"


def _hhem_score(answer: str, context: str) -> float | None:
    from guardrails.faithfulness_check import hhem_score

    try:
        return hhem_score(answer, context)
    except Exception as exc:
        print(f"  (HHEM unavailable: {exc})", file=sys.stderr)
        return None


def _verdict(tool_error: str, answer: str, hhem: float | None) -> str:
    if tool_error:
        return "degraded"
    if looks_like_refusal(answer):
        return "llm_refusal"
    if hhem is not None:
        from guardrails.faithfulness_check import _hhem_to_five_scale

        tier = _hhem_to_five_scale(hhem)
        if tier == 1:
            return "guardrail_refuse"
        if tier <= 3:
            return "guardrail_warn"
    return "ok"


def run_prompt(graph, item: dict) -> dict:
    messages = [
        (HumanMessage if role == "human" else AIMessage)(content=content)
        for role, content in item.get("history", [])
    ]
    messages.append(HumanMessage(content=item["q"]))

    started = time.perf_counter()
    state = graph.invoke(_initial_state(messages))
    elapsed = time.perf_counter() - started

    answer = state["messages"][-1].content
    context = state.get("context_used", "")
    tool_error = state.get("tool_error", "")
    hhem = _hhem_score(answer, context)
    return {
        "question": item["q"],
        "expect": item["expect"],
        "route": state.get("route", ""),
        "verdict": _verdict(tool_error, answer, hhem),
        "tool_error": tool_error,
        "context_chars": len(context),
        "hhem": None if hhem is None else round(hhem, 3),
        "graph_score": state.get("faithfulness_score"),
        "seconds": round(elapsed, 1),
        "answer": answer,
    }


def _print_row(index: int, row: dict) -> None:
    hhem = "  -  " if row["hhem"] is None else f"{row['hhem']:.2f}"
    print(
        f"{index:>2}  {row['expect']:<12} {row['route']:<14} {row['verdict']:<17} "
        f"{hhem:>5} {row['context_chars']:>6} {row['seconds']:>5.1f}  {row['question'][:52]}"
    )
    if row["verdict"] != "ok":
        detail = row["tool_error"] or row["answer"]
        print(f"      -> {' '.join(detail.split())[:150]}")


def _print_summary(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print("\nVerdicts: " + ", ".join(f"{name} {n}" for name, n in sorted(counts.items())))

    answerable = [r for r in rows if r["expect"] == "answerable"]
    unanswerable = [r for r in rows if r["expect"] == "unanswerable"]
    if answerable:
        refused = [r for r in answerable if r["verdict"] in _REFUSING]
        degraded = sum(1 for r in refused if r["verdict"] == "degraded")
        print(
            f"Answerable questions refused: {len(refused)} of {len(answerable)} "
            f"({100 * len(refused) / len(answerable):.0f}%), of which {degraded} were tool errors"
        )
    if unanswerable:
        declined = sum(1 for r in unanswerable if r["verdict"] in _REFUSING)
        print(
            f"Unanswerable questions declined: {declined} of {len(unanswerable)}; "
            f"answered anyway: {len(unanswerable) - declined}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prompts", help="text file, one question per line (replaces the starter set)")
    parser.add_argument("--limit", type=int, help="only run the first N prompts")
    parser.add_argument("--agent-mode", choices=["classifier", "tool_calling"])
    parser.add_argument("--output", help="also write the full results to this JSON file")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # answers contain dashes and emoji

    from chatbot import create_assistant

    settings = load_settings()
    agent_mode = args.agent_mode or settings.agent_mode
    print(f"QDRANT_MODE={settings.qdrant_mode}  AGENT_MODE={agent_mode}  CHAT_MODEL={settings.chat_model}")
    print(f"Retrieval probe: {_probe_retrieval()}\n")

    graph = create_assistant(mode=agent_mode)
    prompts = _load_prompts(args.prompts, args.limit)

    print(" #  expect       route          verdict           hhem    ctx  secs  question")
    rows = []
    for index, item in enumerate(prompts, start=1):
        row = run_prompt(graph, item)
        rows.append(row)
        _print_row(index, row)

    _print_summary(rows)
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
