from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from typing import Annotated, Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI

import query_rewrite
from config import load_settings
from cost import estimate_cost, usage_from_response
from guardrails.faithfulness_check import apply_faithfulness_tiering, score_faithfulness_with_usage
from memory.session_memory import SessionMemory
from observability import set_session_id, setup_logging, setup_tracing, tracing_status
from router import classify_query
from tools.linkedin_tool import get_linkedin_info, get_linkedin_info_result
from tools.personal_tool import get_personal_info, get_personal_info_result
from tools.result import ToolResult
from tools.resume_tool import format_chunks, get_resume_info, get_resume_info_result, search_resume
from tools.spotify_tool import get_music_taste, get_music_taste_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment & observability setup
# ---------------------------------------------------------------------------
load_dotenv()

_SECRET_KEYS = (
    "OPENAI_API_KEY",
    "PHOENIX_API_KEY",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "PHOENIX_PROJECT_NAME",
    "REDIS_URL",
    "QDRANT_MODE",
    "QDRANT_PATH",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "RESUME_PATH",
)

# Deliberately generous: legitimate questions about Cem are a sentence or
# two. This bounds cost per turn (Phase 4.3), not phrasing — the app is
# deployed publicly on the owner's own API key with no other rate limit yet.
MAX_INPUT_CHARS = 1000


def _apply_streamlit_secrets() -> None:
    """Let Streamlit Cloud secrets override .env values.

    Called from inside main(), after set_page_config: reading st.secrets counts
    as a Streamlit command, and set_page_config must be the first one. Missing
    secrets are normal outside Streamlit Cloud, so absence is not an error.
    """
    try:
        secrets = dict(st.secrets)
    except Exception as exc:  # no secrets.toml — the local and CI case
        logger.debug("No Streamlit secrets available: %s", exc)
        return

    for key in _SECRET_KEYS:
        val = secrets.get(key)
        if val:
            os.environ[key] = str(val)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    # add_messages appends new messages to existing ones (matching by id rather
    # than duplicating), which is what makes a checkpointer's incremental
    # updates and LangGraph's prebuilt ToolNode both work. Every message is a
    # real BaseMessage now: generate_response used to put a live streaming
    # generator directly into a message's content, which made state
    # unserializable (no checkpointing possible), made this node's own
    # measured latency read as ~0s (the real work happened later, wherever the
    # caller drained the generator), and needed isinstance(..., str) guards
    # scattered across three modules. A BaseMessage's content is validated at
    # construction — a generator cannot land here even by accident.
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: str
    # The last message rewritten to stand alone, which is what routing and
    # retrieval both act on. Identical to the last message on a first turn.
    search_query: str
    # The routing decision, kept apart from next_step, which is control flow and
    # is overwritten by each node. Without it the chosen route is unrecoverable
    # after a run — and routing accuracy is measured on exactly that.
    route: str
    tool_result: str
    context_used: str
    # Retrieved chunks as a list. Ranking metrics over one joined blob are
    # degenerate: precision@k has to see the chunks separately.
    context_chunks: list[str]
    # Set when the handler's tool call failed. Empty on success, and never
    # folded into tool_result or context_used — an outage must not become the
    # model's evidence, nor something the faithfulness judge scores against.
    tool_error: str
    node_latencies: dict[str, float]
    # The raw 1-5 judge score for the current answer, or None when nothing was
    # scored (no context, or the judge call failed) — classifier mode only,
    # set by check_faithfulness_node. Distinct from the user-facing tiering
    # guardrails.faithfulness_check.check_faithfulness still applies outside
    # the graph: this drives the self-correction retry below, not the banner.
    faithfulness_score: int | None
    # How many times this turn has already retried after a low resume-route
    # score. Bounded by _MAX_FAITHFULNESS_RETRIES so a persistently ungrounded
    # answer degrades to the existing disclaimer/refusal tiering rather than
    # looping.
    retry_count: int
    # tool_calling mode only: one entry per tool call across every round this
    # turn, in order — {"round", "tool", "category", "args", "ok"}. Phase 3.5's
    # trajectory eval needs the actual sequence, which the accumulated `route`
    # string alone cannot reconstruct (it loses round boundaries and args).
    trajectory: list[dict]
    # tool_calling mode only: how many rounds of execute_tools have run this
    # turn, stamped onto each round's trajectory entries.
    agent_rounds: int
    # Real per-node token usage (Phase 4.2), keyed by node name — every node
    # that calls a model reads its own response's usage_metadata and merges it
    # in via _add_usage. Not LangChain's get_openai_callback: confirmed
    # directly that it does not propagate through LangGraph's node execution
    # (a real graph turn showed 0 tokens captured that way despite a real
    # model call happening), so each call site captures its own instead.
    # check_faithfulness stays a separate key because it is always
    # gpt-4o-mini regardless of CHAT_MODEL — see main()'s cost accounting.
    token_usage: dict[str, dict[str, int]]


# ---------------------------------------------------------------------------
# LangGraph factory
# ---------------------------------------------------------------------------
def _timed(name: str, fn, state: GraphState) -> GraphState:
    start = time.perf_counter()
    result = fn(state)
    elapsed = time.perf_counter() - start
    latencies = dict(result.get("node_latencies", {}))
    latencies[name] = round(elapsed, 3)
    result["node_latencies"] = latencies
    return result


def _add_usage(token_usage: dict[str, dict[str, int]] | None, node: str, usage: dict[str, int]) -> dict:
    """Merge one call's real usage into a node-keyed running total (Phase 4.2).

    Per node, not one grand total: check_faithfulness stays pinned to
    gpt-4o-mini regardless of CHAT_MODEL, so it must be priced separately from
    everything else in state — see main()'s cost accounting below.
    """
    merged = dict(token_usage or {})
    existing = merged.get(node, {"input": 0, "output": 0})
    merged[node] = {
        "input": existing["input"] + usage["input"],
        "output": existing["output"] + usage["output"],
    }
    return merged


def _system_prompt(prior_context: str) -> str:
    # Built outside the f-string: a backslash in an f-string expression is a
    # syntax error before Python 3.12, and this project targets 3.11.
    prior_block = ""
    if prior_context:
        prior_block = "Prior conversation context (returning visitor):\n" + prior_context

    return f"""You are a helpful personal assistant chatbot for Cem Kaspi.
You have access to Cem's resume, personal information, Spotify listening history, and LinkedIn profile.
Use the available tools to retrieve the most relevant information to answer queries about Cem.
If you did not receive any relevant information from the tools, say so honestly.
Do not make up or invent information. Be helpful, friendly, and professional with a touch of humour.

Key facts about Cem:
- AI/ML Engineer at TELUS Communications Inc. in Vancouver, Canada
- Originally from Istanbul, Turkey
- Speaks Turkish, English, and beginner Spanish
- Born in 1997

{prior_block}"""


def create_assistant(prior_context: str = "", mode: str | None = None, model: str | None = None) -> Any:
    """Build the compiled graph. ``mode`` defaults to ``settings.agent_mode``,
    ``model`` to ``settings.chat_model``.

    Two independent graph shapes, chosen by configuration rather than one
    replacing the other on faith: "classifier" is the original design (an LLM
    picks one of five fixed labels, a hardcoded switch calls exactly one
    tool); "tool_calling" binds the tools to the LLM directly and lets it
    choose, call zero-to-many of them, and loop back with the results before
    answering. Both are measured (see eval/results/) so the choice of default
    is evidence, not preference. Same discipline for ``model``: see README for
    the gpt-4o-mini/gpt-6-luna comparison behind CHAT_MODEL's default.
    """
    settings = load_settings()
    chat_model = model or settings.chat_model

    # gpt-6-luna is a reasoning model: it rejects any non-default temperature
    # outright — a live call returns 400 "Unsupported value: 'temperature'
    # does not support 0.0 with this model. Only the default (1) value is
    # supported," not just for 0. gpt-4o-mini has no such restriction, so the
    # override is only skipped for the model that cannot accept it.
    fixed_temperature = chat_model == "gpt-6-luna"

    def _llm(temperature: float, streaming: bool, **extra: Any) -> ChatOpenAI:
        # stream_usage defaults to False on ChatOpenAI — a streaming call
        # without it returns no usage_metadata at all (confirmed directly: a
        # real generate_response call produced a real answer but usage_from_
        # response read 0/0). Harmless to set unconditionally: it only affects
        # the streaming code path, and llm_fast (streaming=False) ignores it.
        kwargs: dict[str, Any] = {
            "model": chat_model,
            "streaming": streaming,
            "stream_usage": True,
            **extra,
        }
        if not fixed_temperature:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)

    llm = _llm(0.7, streaming=True)
    llm_fast = _llm(0, streaming=False)
    system_prompt = _system_prompt(prior_context)
    mode = mode or settings.agent_mode

    if mode == "tool_calling":
        # Not llm: that call decides *and* eventually writes the final answer
        # in the same loop, and llm's temperature=0.7 was chosen for prose
        # variety, not for tool-selection — measured directly, it made which
        # tool got called nondeterministic (routing moved 93.2% -> 91.5%
        # between two identical runs). The classifier avoids this by using a
        # separate temp=0 model for its own decision point; this does the
        # same, while keeping streaming=True so the final round still types
        # live in the UI. gpt-6-luna cannot be pinned to temp=0 at all (see
        # fixed_temperature above) — whatever determinism its tool selection
        # has comes from the model itself, not from this project's usual fix,
        # and is measured rather than assumed in the README comparison.
        extra = {"reasoning_effort": "none"} if chat_model == "gpt-6-luna" else {}
        # gpt-6-luna only supports function/tool calling via Chat Completions
        # at reasoning_effort="none" — tool selection breaks silently
        # otherwise. Irrelevant to llm/llm_fast above: neither ever has a tool
        # bound to it (routing and condensation are plain prompt completions,
        # not function calls). A top-level kwarg, not model_kwargs:
        # langchain_openai warns that reasoning_effort has its own
        # constructor parameter and should not be nested.
        llm_agent = _llm(0, streaming=True, **extra)
        return _build_tool_calling_graph(llm_agent, system_prompt)
    return _build_classifier_graph(llm, llm_fast, system_prompt)


# Self-correction (Phase 3.3) is scoped to the resume route only. personal,
# spotify, and linkedin are fixed lookups — the same info_type, or no argument
# at all, every time — so retrying returns byte-identical content; retrying
# them would spend a judge call and a generation call to reproduce the exact
# answer already given. conversation has no retrieved context at all, so
# there is nothing a retry could reformulate. resume is the only route with a
# free-text query a differently-worded retry could actually change.
_RETRIABLE_ROUTE = "resume"
# Bounds the loop to one retry per turn regardless of how the second attempt
# scores — an answer that is still ungrounded after a reformulated search is
# a case for the existing disclaimer/refusal tiering, not another round trip.
_MAX_FAITHFULNESS_RETRIES = 1


def _build_classifier_graph(llm: ChatOpenAI, llm_fast: ChatOpenAI, system_prompt: str) -> Any:
    """The original design: classify, then a hardcoded switch calls one tool."""

    def condense_query(state: GraphState) -> GraphState:
        def _run(state):
            messages = state["messages"]
            query = messages[-1].content
            # Only pays for a model call when there is history to resolve
            # against, so a first turn costs nothing.
            rewritten, usage = query_rewrite.condense_query(query, messages[:-1], llm_fast)
            token_usage = _add_usage(state.get("token_usage"), "condense_query", usage)
            return {**state, "search_query": rewritten, "token_usage": token_usage}

        return _timed("condense_query", _run, state)

    def route_query(state: GraphState) -> GraphState:
        def _run(state):
            query = state.get("search_query") or state["messages"][-1].content
            qtype, usage = classify_query(query, llm_fast)
            token_usage = _add_usage(state.get("token_usage"), "route_query", usage)
            return {**state, "next_step": qtype, "route": qtype, "token_usage": token_usage}

        return _timed("route_query", _run, state)

    def handle_resume(state: GraphState) -> GraphState:
        def _run(state):
            query = state.get("search_query") or state["messages"][-1].content
            try:
                chunks = search_resume(query)
                result = ToolResult.success(format_chunks(chunks))
            except Exception as e:
                logger.warning("Resume retrieval failed: %s", e)
                chunks, result = [], ToolResult.failure(str(e))
            return {
                **state,
                "tool_result": result.as_context(),
                "context_used": result.as_context(),
                "context_chunks": [c.text for c in chunks],
                "tool_error": result.error,
                "next_step": "generate_response",
            }

        return _timed("handle_resume", _run, state)

    def handle_personal(state: GraphState) -> GraphState:
        def _run(state):
            result = get_personal_info_result()
            return {
                **state,
                "tool_result": result.as_context(),
                "context_used": result.as_context(),
                "context_chunks": [result.content] if result.ok else [],
                "tool_error": result.error,
                "next_step": "generate_response",
            }

        return _timed("handle_personal", _run, state)

    def handle_spotify(state: GraphState) -> GraphState:
        def _run(state):
            result = get_music_taste_result()
            return {
                **state,
                "tool_result": result.as_context(),
                "context_used": result.as_context(),
                "context_chunks": [result.content] if result.ok else [],
                "tool_error": result.error,
                "next_step": "generate_response",
            }

        return _timed("handle_spotify", _run, state)

    def handle_linkedin(state: GraphState) -> GraphState:
        def _run(state):
            result = get_linkedin_info_result()
            return {
                **state,
                "tool_result": result.as_context(),
                "context_used": result.as_context(),
                "context_chunks": [result.content] if result.ok else [],
                "tool_error": result.error,
                "next_step": "generate_response",
            }

        return _timed("handle_linkedin", _run, state)

    def handle_conversation(state: GraphState) -> GraphState:
        def _run(state):
            return {
                **state,
                "tool_result": "",
                "context_used": "",
                "context_chunks": [],
                "tool_error": "",
                "next_step": "generate_response",
            }

        return _timed("handle_conversation", _run, state)

    def generate_response(state: GraphState) -> GraphState:
        def _run(state):
            messages = state["messages"]
            tool_result = state.get("tool_result", "")
            context_used = state.get("context_used", "")

            lc_messages = [SystemMessage(content=system_prompt)]
            if tool_result:
                lc_messages.append(SystemMessage(content=f"Relevant information:\n{tool_result}"))
            # messages are already real HumanMessage/AIMessage objects — no
            # reconstruction needed, unlike when this read dicts.
            lc_messages.extend(messages)

            # A blocking call, not .stream(): the node's own return value must
            # be a plain, serializable message. The caller still gets live
            # token-by-token output via graph.stream(..., stream_mode=
            # "messages"), which surfaces this same call's streaming callbacks
            # regardless of whether the node awaits .invoke() or .stream().
            response = llm.invoke(lc_messages)
            if state.get("retry_count", 0) > 0:
                # This call is regenerating the same turn's answer after a
                # self-correction retry — it must replace the first attempt,
                # not sit beside it as a second assistant message. add_messages
                # matches by id, so reusing the prior answer's id is what makes
                # this a replacement instead of an append.
                response.id = messages[-1].id
            gen_usage = usage_from_response(response)
            token_usage = _add_usage(state.get("token_usage"), "generate_response", gen_usage)
            # A single-element list, not the full history: add_messages
            # appends it. Returning the whole list back here would ask the
            # reducer to "add" every prior message a second time — harmless
            # (matched by id, so no duplication) but pointless.
            return {
                **state,
                "messages": [response],
                "next_step": "end",
                "context_used": context_used,
                "token_usage": token_usage,
            }

        return _timed("generate_response", _run, state)

    def check_faithfulness_node(state: GraphState) -> GraphState:
        def _run(state):
            answer = state["messages"][-1].content
            score, usage = score_faithfulness_with_usage(answer, state.get("context_used", ""))
            token_usage = _add_usage(state.get("token_usage"), "check_faithfulness", usage)
            return {**state, "faithfulness_score": score, "token_usage": token_usage}

        return _timed("check_faithfulness", _run, state)

    def reformulate_and_retry(state: GraphState) -> GraphState:
        def _run(state):
            query = state.get("search_query", "")
            new_query, usage = query_rewrite.reformulate_for_retry(query, llm_fast)
            token_usage = _add_usage(state.get("token_usage"), "reformulate_and_retry", usage)
            try:
                chunks = search_resume(new_query)
                result = ToolResult.success(format_chunks(chunks))
            except Exception as e:
                logger.warning("Resume retry retrieval failed: %s", e)
                chunks, result = [], ToolResult.failure(str(e))
            return {
                **state,
                "search_query": new_query,
                "tool_result": result.as_context(),
                "context_used": result.as_context(),
                "context_chunks": [c.text for c in chunks],
                "tool_error": result.error,
                "retry_count": state.get("retry_count", 0) + 1,
                "token_usage": token_usage,
            }

        return _timed("reformulate_and_retry", _run, state)

    def _should_retry(state: GraphState) -> str:
        score = state.get("faithfulness_score")
        already_retried = state.get("retry_count", 0) >= _MAX_FAITHFULNESS_RETRIES
        is_retriable_route = state.get("route") == _RETRIABLE_ROUTE
        if score is not None and score < 4 and is_retriable_route and not already_retried:
            return "retry"
        return "finish"

    workflow = StateGraph(GraphState)
    workflow.add_node("condense_query", condense_query)
    workflow.add_node("route_query", route_query)
    workflow.add_node("handle_resume", handle_resume)
    workflow.add_node("handle_personal", handle_personal)
    workflow.add_node("handle_spotify", handle_spotify)
    workflow.add_node("handle_linkedin", handle_linkedin)
    workflow.add_node("handle_conversation", handle_conversation)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("check_faithfulness", check_faithfulness_node)
    workflow.add_node("reformulate_and_retry", reformulate_and_retry)

    workflow.add_conditional_edges(
        "route_query",
        lambda x: x["next_step"],
        {
            "resume": "handle_resume",
            "personal": "handle_personal",
            "spotify": "handle_spotify",
            "linkedin": "handle_linkedin",
            "conversation": "handle_conversation",
        },
    )
    for node in (
        "handle_resume",
        "handle_personal",
        "handle_spotify",
        "handle_linkedin",
        "handle_conversation",
    ):
        workflow.add_edge(node, "generate_response")

    # Self-correction (Phase 3.3): the faithfulness score used to end at a text
    # banner applied outside the graph. Now it drives one bounded retry —
    # reformulate the query, re-run resume retrieval, regenerate — before
    # falling through to that same outer banner. Scoped to the resume route
    # only: see _RETRIABLE_ROUTE above for why the other routes have no
    # retrieval lever a retry could actually change.
    workflow.add_edge("generate_response", "check_faithfulness")
    workflow.add_conditional_edges(
        "check_faithfulness",
        _should_retry,
        {"retry": "reformulate_and_retry", "finish": "__end__"},
    )
    workflow.add_edge("reformulate_and_retry", "generate_response")

    workflow.add_edge("condense_query", "route_query")
    workflow.set_entry_point("condense_query")
    return workflow.compile()


# Maps a tool's registered name to the golden set's route category, so
# routing/tool-selection accuracy is scored identically for both graph modes —
# eval/run_eval.py never needs to know which mode produced a given result.
_TOOL_TO_CATEGORY: dict[str, str] = {
    "get_resume_info": "resume",
    "get_personal_info": "personal",
    "get_music_taste": "spotify",
    "get_linkedin_info": "linkedin",
}


def _call_tool(name: str, args: dict) -> ToolResult:
    """Dispatch a tool call by name to its typed result function.

    Not the @tool-decorated function itself: those return a plain string
    (content on success, an error-prefixed string on failure) for the LLM
    tool-calling contract. The typed function is what lets execute_tools keep
    a failed call's error text out of context_used while still handing the
    agent the same string a human calling the tool would see.
    """
    if name == "get_resume_info":
        return get_resume_info_result(args.get("query", ""))
    if name == "get_personal_info":
        return get_personal_info_result(args.get("info_type", ""))
    if name == "get_music_taste":
        return get_music_taste_result()
    if name == "get_linkedin_info":
        return get_linkedin_info_result()
    return ToolResult.failure(f"Unknown tool: {name}")


# A measured addition, not a guess written in advance: a first pass at this
# graph — no addendum, just the shared system prompt — routed at 81.4% against
# the classifier's 98.3% on the same golden set. Nearly every miss had the same
# shape: the agent declined ("I don't have information about that") instead of
# calling a tool that would have answered it, for anything not already in the
# prompt's few "key facts" lines. Not hallucination — it never invented an
# answer — but premature refusal, treating four bullet points as the ceiling of
# what it knows rather than as tone-setting context. This addendum is the fix,
# and its effect is measured in eval/results/ under agent_mode=tool_calling,
# same as everything else in this file.
_TOOL_CALLING_ADDENDUM = """
The "Key facts" above are for tone and identity only — never treat their
absence as evidence you lack information. For any question about Cem's career,
education, background, music taste, or LinkedIn history, call the relevant
tool before answering, even if you suspect you already know. Only say you
don't have something after checking, and never guess or invent a detail no
tool returned."""


def _build_tool_calling_graph(llm: ChatOpenAI, system_prompt: str) -> Any:
    """Bind the tools to the LLM and let it choose — a real agent, not a switch.

    No condense_query node: unlike the classifier, which only ever sees one
    isolated string (the query) with no memory of its own, this agent sees the
    full conversation history natively when deciding which tool to call and
    with what arguments — it can resolve "and before that?" while constructing
    the tool call itself. Whether that actually holds up is exactly what
    eval/run_eval.py measures per mode, rather than being assumed.

    One AIMessage can carry more than one tool_call — the model may decide to
    call get_resume_info and get_linkedin_info in the same turn for "compare
    his resume to his LinkedIn" — and execute_tools' loop already handles
    that, which is most of Phase 3.2's multi-intent case arriving as a side
    effect of building this properly rather than needing separate work.
    """
    tools = [get_resume_info, get_personal_info, get_music_taste, get_linkedin_info]
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = system_prompt + _TOOL_CALLING_ADDENDUM

    def agent(state: GraphState) -> GraphState:
        def _run(state):
            lc_messages = [SystemMessage(content=system_prompt), *state["messages"]]
            response = llm_with_tools.invoke(lc_messages)
            token_usage = _add_usage(state.get("token_usage"), "agent", usage_from_response(response))
            updates = {**state, "messages": [response], "token_usage": token_usage}
            # Only when nothing has called a tool yet this turn: a later round
            # with no further tool_calls is the agent's final answer after an
            # earlier round already set a real route, and must not overwrite it.
            if not getattr(response, "tool_calls", None) and not state.get("route"):
                updates["route"] = "conversation"
            return updates

        return _timed("agent", _run, state)

    def execute_tools(state: GraphState) -> GraphState:
        def _run(state):
            last = state["messages"][-1]
            tool_messages: list[ToolMessage] = []
            categories, context_parts, chunk_parts, error_parts = [], [], [], []
            # One entry per call, this round's index carried on each — trajectory
            # eval (Phase 3.5) needs the actual sequence of (round, tool, args),
            # not just which categories got touched somewhere in the turn. The
            # accumulated `route` string already loses "which round" and "with
            # what arguments"; this doesn't.
            round_num = state.get("agent_rounds", 0)
            trajectory_entries: list[dict] = []

            for call in last.tool_calls:
                name, args = call["name"], call.get("args") or {}
                try:
                    result = _call_tool(name, args)
                except Exception as e:
                    logger.warning("Tool %s failed: %s", name, e)
                    result = ToolResult.failure(str(e))

                category = _TOOL_TO_CATEGORY.get(name, name)
                categories.append(category)
                trajectory_entries.append(
                    {"round": round_num, "tool": name, "category": category, "args": args, "ok": result.ok}
                )
                if result.ok:
                    context_parts.append(result.content)
                    chunk_parts.append(result.content)
                else:
                    error_parts.append(f"{name}: {result.error}")
                tool_messages.append(
                    ToolMessage(
                        content=result.as_tool_string(f"Error calling {name}"),
                        tool_call_id=call["id"],
                    )
                )

            # Accumulated, not overwritten: a second round of tool calls in the
            # same turn (genuine multi-hop) must not lose the first round's
            # context, route, or errors.
            return {
                **state,
                "messages": tool_messages,
                "route": ",".join(filter(None, [state.get("route", ""), *categories])),
                "context_used": "\n\n".join(filter(None, [state.get("context_used", ""), *context_parts])),
                "context_chunks": (state.get("context_chunks") or []) + chunk_parts,
                "trajectory": (state.get("trajectory") or []) + trajectory_entries,
                "agent_rounds": round_num + 1,
                "tool_error": "; ".join(filter(None, [state.get("tool_error", ""), *error_parts])),
            }

        return _timed("execute_tools", _run, state)

    def should_continue(state: GraphState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    workflow = StateGraph(GraphState)
    workflow.add_node("agent", agent)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_conditional_edges("agent", should_continue, {"tools": "execute_tools", "end": "__end__"})
    workflow.add_edge("execute_tools", "agent")
    workflow.set_entry_point("agent")
    return workflow.compile()


# ---------------------------------------------------------------------------
# Token cost accounting (Phase 4.2)
# ---------------------------------------------------------------------------
def _accumulate_real_cost(*priced_usages: tuple[str, dict[str, int]]) -> None:
    """Add real per-call token usage to the running session total.

    Replaces estimating tokens as len(text) // 4 and pricing all of it at the
    output rate — which counted only the final answer's characters, missing
    every other call a turn makes (router, judge, summariser) and mispricing
    the one call it did count. `priced_usages` is (model, usage) pairs, not a
    single model, because the judge and summariser stay pinned to gpt-4o-mini
    regardless of CHAT_MODEL while the graph's own calls use whichever model
    is actually configured — each needs its own rate, not one applied to all.
    """
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
    for model, usage in priced_usages:
        input_tokens, output_tokens = usage.get("input", 0), usage.get("output", 0)
        st.session_state.total_tokens += input_tokens + output_tokens
        st.session_state.total_cost += estimate_cost(model, input_tokens, output_tokens)


# ---------------------------------------------------------------------------
# Session ID management (for Redis memory) — Phase 4.5
# ---------------------------------------------------------------------------
# A process-lifetime fallback: with no SESSION_SECRET configured, signed
# links still work (sharing a session's own URL is a deliberate feature, not
# the vulnerability), but stop verifying after a restart — the same
# degrade-rather-than-refuse pattern Redis absence already uses elsewhere in
# this project. Set SESSION_SECRET for a deployment that should survive one.
_EPHEMERAL_SESSION_SECRET = secrets.token_hex(32)


def _session_secret() -> bytes:
    return (os.environ.get("SESSION_SECRET") or _EPHEMERAL_SESSION_SECRET).encode()


def _sign_session_id(sid: str) -> str:
    mac = hmac.new(_session_secret(), sid.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{sid}.{mac}"


def _verify_session_id(signed: str) -> str | None:
    """The raw sid if `signed` carries a valid signature, else None.

    Without this, chatbot.py read `sid` straight from the URL query param
    with no validation and used it directly as the Redis key — editing
    `?sid=` to any string, guessed or not, read that "session" (an empty one
    for a fresh guess, but a real IDOR the moment two visitors' guesses
    collide, or a leaked link is reused). A signature does not stop someone
    who has a *legitimately issued* link from opening it — sharing a session
    via URL is a documented feature, not this bug — it stops a fabricated or
    edited sid from ever being accepted as one the server actually issued.
    """
    sid, _, mac = signed.rpartition(".")
    if not sid or not mac:
        return None
    expected = hmac.new(_session_secret(), sid.encode(), hashlib.sha256).hexdigest()[:16]
    return sid if hmac.compare_digest(mac, expected) else None


def _get_or_create_session_id() -> str:
    raw = st.query_params.get("sid")
    sid = _verify_session_id(raw) if raw else None
    if sid is None:
        sid = str(uuid.uuid4())

    signed = _sign_session_id(sid)
    if st.query_params.get("sid") != signed:
        st.query_params["sid"] = signed

    if "session_id" not in st.session_state:
        st.session_state.session_id = sid
    return st.session_state.session_id


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HowToCem",
        page_icon="😎",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _apply_streamlit_secrets()
    setup_logging()
    # After secrets (which may carry the Phoenix credentials) and before
    # create_assistant: the instrumentor patches LangChain's callback manager,
    # so objects built earlier would never be traced.
    setup_tracing()

    st.markdown(
        """
        <style>
        .app-title { font-size: 2.5rem; font-weight: bold; color: #1c1c1c;
                     text-align: center; margin-bottom: 10px; }
        .app-subtitle { font-size: 1rem; color: #555; text-align: center; margin-bottom: 20px; }
        </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">HowToCem</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">👋 Ask me anything about Cem — resume, career, '
        "music taste, or personal background.</div>",
        unsafe_allow_html=True,
    )

    # --- Session state init ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_latencies" not in st.session_state:
        st.session_state.last_latencies = {}
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0

    session_id = _get_or_create_session_id()
    set_session_id(session_id)
    memory = SessionMemory()
    prior_context = memory.load_summary(session_id)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Session Info")
        cost = st.session_state.total_cost
        tokens = st.session_state.total_tokens
        st.metric(
            "Session cost",
            f"${cost:.5f}",
            help=(
                "Real token usage from the API's own usage object, priced at each model's "
                "published rate — not an estimate from text length. Excludes embedding calls "
                "(a few tokens per query, ~$0.02/1M — negligible next to either chat model)."
            ),
        )
        st.metric("Tokens used", f"{tokens:,}")

        if st.session_state.last_latencies:
            with st.expander("⏱ Last query latency"):
                for node, ms in st.session_state.last_latencies.items():
                    st.text(f"{node}: {ms:.3f}s")
            total_lat = sum(st.session_state.last_latencies.values())
            st.caption(f"Total: {total_lat:.3f}s")

        if prior_context:
            with st.expander("🧠 Prior session context"):
                st.caption(prior_context)

        st.divider()
        st.markdown("### Backends")
        st.caption(f"Vector store: Qdrant ({load_settings().qdrant_mode})")
        st.caption(f"Session memory: {memory.backend}")
        st.caption(f"Tracing: {tracing_status()}")

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_latencies = {}
            st.session_state.total_tokens = 0
            st.session_state.total_cost = 0.0
            st.rerun()

    # --- Build graph ---
    if "assistant_graph" not in st.session_state:
        with st.spinner("Initialising retrieval index..."):
            st.session_state.assistant_graph = create_assistant(prior_context)

    graph = st.session_state.assistant_graph

    # --- Display history ---
    for msg in st.session_state.messages:
        avatar = "😎" if msg.type == "ai" else "🧐"
        with st.chat_message(msg.type, avatar=avatar):
            if isinstance(msg.content, str):
                st.markdown(msg.content)

    # --- Chat input ---
    if prompt := st.chat_input("Ask me something about Cem..."):
        if len(prompt) > MAX_INPUT_CHARS:
            # Rejected before it ever reaches session state or the graph: an
            # unbounded input is an unbounded cost, and this app is deployed
            # publicly on the owner's own API key with no other rate limit
            # (Phase 4.1's daily spend cap is the other half of that, not yet
            # built). Nothing is sent to the model, and the oversized text
            # never enters conversation history — it would otherwise still
            # cost tokens on every future turn as context, even though this
            # turn itself made no API call.
            st.error(
                f"That message is {len(prompt):,} characters — please keep it under "
                f"{MAX_INPUT_CHARS:,}. Nothing was sent to the model."
            )
            return

        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human", avatar="🧐"):
            st.markdown(prompt)

        chat_model = load_settings().chat_model
        with st.chat_message("ai", avatar="😎"):
            placeholder = st.empty()
            state: GraphState = {
                "messages": list(st.session_state.messages),
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

            # Two stream modes in one pass: "messages" gives live token deltas
            # for the placeholder below — LangGraph surfaces a streaming-
            # enabled model's callbacks regardless of which method the node
            # used. "values" gives the full state after each step; the last
            # one is the graph's result, same as graph.invoke() would return,
            # with no separate call. Node names differ by mode:
            # generate_response (classifier) vs agent (tool_calling, possibly
            # invoked more than once per turn — a round that decides to call a
            # tool typically emits little or no text, so its chunks simply add
            # nothing rather than needing to be filtered out separately).
            full_response = ""
            result_state: GraphState | None = None
            with st.spinner("Thinking..."):
                for mode, payload in graph.stream(state, stream_mode=["messages", "values"]):
                    if mode == "messages":
                        chunk, meta = payload
                        if meta.get("langgraph_node") in ("generate_response", "agent") and chunk.content:
                            full_response += chunk.content
                            placeholder.markdown(full_response + "▌")
                    elif mode == "values":
                        result_state = payload

            assert result_state is not None  # "values" mode always yields at least once
            placeholder.markdown(full_response)

            st.session_state.last_latencies = result_state.get("node_latencies", {})
            context_used = result_state.get("context_used", "")
            if result_state.get("tool_error"):
                # Never shown to the visitor — showErrorDetails is off for that
                # reason — but a deployed instance's logs should not stay silent
                # about a tool that is actually failing.
                logger.warning(
                    "Tool failure on route=%s: %s", result_state.get("route"), result_state["tool_error"]
                )

            # The streamed tokens are for the live placeholder only; the
            # authoritative text is whatever the node actually returned.
            full_response = result_state["messages"][-1].content

            # Faithfulness check. classifier mode already scored this exact
            # answer in-graph (for the self-correction retry decision) — reuse
            # it instead of paying for a second judge call on the same text.
            # tool_calling mode never scores in-graph, so this is where it
            # happens for that mode.
            faithfulness_score = result_state.get("faithfulness_score")
            judge_usage = (result_state.get("token_usage") or {}).get("check_faithfulness")
            if judge_usage is None:
                faithfulness_score, judge_usage = score_faithfulness_with_usage(full_response, context_used)
            final_response = apply_faithfulness_tiering(full_response, faithfulness_score)
            if final_response != full_response:
                placeholder.markdown(final_response)

            result_state["messages"][-1].content = final_response
            st.session_state.messages = result_state["messages"]

            # Update session memory summary in background
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            summary, summary_usage = memory.build_summary(
                [m for m in st.session_state.messages if isinstance(m.content, str)],
                openai_client,
            )
            if summary:
                memory.save_summary(session_id, summary)

            # Real usage from every call this turn: every graph node captures
            # its own response's usage_metadata directly (see _add_usage and
            # cost.usage_from_response) rather than a callback, priced at
            # whichever CHAT_MODEL is actually configured — except
            # check_faithfulness and the summariser, which stay pinned to
            # gpt-4o-mini regardless of it (see cost.py and
            # guardrails/faithfulness_check.py for why).
            graph_usage = {
                node: usage
                for node, usage in (result_state.get("token_usage") or {}).items()
                if node != "check_faithfulness"
            }
            graph_input = sum(u["input"] for u in graph_usage.values())
            graph_output = sum(u["output"] for u in graph_usage.values())
            _accumulate_real_cost(
                (chat_model, {"input": graph_input, "output": graph_output}),
                ("gpt-4o-mini", judge_usage),
                ("gpt-4o-mini", summary_usage),
            )

            # Structured, correlated by session_id (see observability.py):
            # per-stage retrieval timing that the sidebar already shows
            # interactively is also worth having in a deployed instance's
            # actual logs, not just its UI.
            logger.info(
                "turn completed",
                extra={
                    "route": result_state.get("route"),
                    "node_latencies": result_state.get("node_latencies", {}),
                    "faithfulness_score": faithfulness_score,
                    "self_correction_retried": result_state.get("retry_count", 0) > 0,
                },
            )

        st.rerun()


if __name__ == "__main__":
    main()
