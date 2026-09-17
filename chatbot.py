from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from openai import OpenAI

from config import load_settings
from guardrails.faithfulness_check import check_faithfulness
from memory.session_memory import SessionMemory
from observability import setup_tracing, tracing_status
from router import classify_query
from tools.linkedin_tool import get_linkedin_info
from tools.personal_tool import get_personal_info
from tools.resume_tool import get_resume_info
from tools.spotify_tool import get_music_taste

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
# Constants
# ---------------------------------------------------------------------------
GPT_4O_MINI_INPUT_COST_PER_1K = 0.000150  # $ per 1k input tokens
GPT_4O_MINI_OUTPUT_COST_PER_1K = 0.000600  # $ per 1k output tokens


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    messages: list[dict[str, str]]
    next_step: str
    tool_result: str
    context_used: str
    node_latencies: dict[str, float]


# ---------------------------------------------------------------------------
# LangGraph factory
# ---------------------------------------------------------------------------
def create_assistant(prior_context: str = "") -> Any:
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=True)
    llm_fast = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=False)

    # Built outside the f-string: a backslash in an f-string expression is a
    # syntax error before Python 3.12, and this project targets 3.11.
    prior_block = ""
    if prior_context:
        prior_block = "Prior conversation context (returning visitor):\n" + prior_context

    system_prompt = f"""You are a helpful personal assistant chatbot for Cem Kaspi.
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

    def timed(name: str, fn, state: GraphState) -> GraphState:
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        latencies = dict(result.get("node_latencies", {}))
        latencies[name] = round(elapsed, 3)
        result["node_latencies"] = latencies
        return result

    def route_query(state: GraphState) -> GraphState:
        def _run(state):
            query = state["messages"][-1]["content"]
            qtype = classify_query(query, llm_fast)
            return {**state, "next_step": qtype}

        return timed("route_query", _run, state)

    def handle_resume(state: GraphState) -> GraphState:
        def _run(state):
            result = get_resume_info.invoke(state["messages"][-1]["content"])
            return {**state, "tool_result": result, "context_used": result, "next_step": "generate_response"}

        return timed("handle_resume", _run, state)

    def handle_personal(state: GraphState) -> GraphState:
        def _run(state):
            result = get_personal_info.invoke("")
            return {**state, "tool_result": result, "context_used": result, "next_step": "generate_response"}

        return timed("handle_personal", _run, state)

    def handle_spotify(state: GraphState) -> GraphState:
        def _run(state):
            result = get_music_taste.invoke({})
            return {**state, "tool_result": result, "context_used": result, "next_step": "generate_response"}

        return timed("handle_spotify", _run, state)

    def handle_linkedin(state: GraphState) -> GraphState:
        def _run(state):
            result = get_linkedin_info.invoke({})
            return {**state, "tool_result": result, "context_used": result, "next_step": "generate_response"}

        return timed("handle_linkedin", _run, state)

    def handle_conversation(state: GraphState) -> GraphState:
        def _run(state):
            return {**state, "tool_result": "", "context_used": "", "next_step": "generate_response"}

        return timed("handle_conversation", _run, state)

    def generate_response(state: GraphState) -> GraphState:
        def _run(state):
            messages = state["messages"]
            tool_result = state.get("tool_result", "")
            context_used = state.get("context_used", "")

            lc_messages = [SystemMessage(content=system_prompt)]
            if tool_result:
                lc_messages.append(SystemMessage(content=f"Relevant information:\n{tool_result}"))
            for msg in messages:
                if msg["role"] == "human":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai" and isinstance(msg["content"], str):
                    lc_messages.append(AIMessage(content=msg["content"]))

            response = llm.stream(lc_messages)
            updated = list(messages) + [{"role": "ai", "content": response}]
            return {**state, "messages": updated, "next_step": "end", "context_used": context_used}

        return timed("generate_response", _run, state)

    workflow = StateGraph(GraphState)
    workflow.add_node("route_query", route_query)
    workflow.add_node("handle_resume", handle_resume)
    workflow.add_node("handle_personal", handle_personal)
    workflow.add_node("handle_spotify", handle_spotify)
    workflow.add_node("handle_linkedin", handle_linkedin)
    workflow.add_node("handle_conversation", handle_conversation)
    workflow.add_node("generate_response", generate_response)

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

    workflow.set_entry_point("route_query")
    return workflow.compile()


# ---------------------------------------------------------------------------
# Token cost accounting
# ---------------------------------------------------------------------------
def _estimate_cost(text: str) -> float:
    tokens = len(text) // 4
    return tokens * GPT_4O_MINI_OUTPUT_COST_PER_1K / 1000


def _accumulate_cost(chunk_text: str) -> None:
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
    st.session_state.total_tokens += len(chunk_text) // 4
    st.session_state.total_cost = st.session_state.total_tokens * GPT_4O_MINI_OUTPUT_COST_PER_1K / 1000


# ---------------------------------------------------------------------------
# Session ID management (for Redis memory)
# ---------------------------------------------------------------------------
def _get_or_create_session_id() -> str:
    params = st.query_params
    sid = params.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        st.query_params["sid"] = sid
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
    memory = SessionMemory()
    prior_context = memory.load_summary(session_id)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Session Info")
        cost = st.session_state.total_cost
        tokens = st.session_state.total_tokens
        st.metric("Session cost", f"${cost:.5f}", help="Estimated based on gpt-4o-mini token pricing")
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
        avatar = "😎" if msg["role"] == "ai" else "🧐"
        with st.chat_message(msg["role"], avatar=avatar):
            if isinstance(msg["content"], str):
                st.markdown(msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask me something about Cem..."):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human", avatar="🧐"):
            st.markdown(prompt)

        with st.chat_message("ai", avatar="😎"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                state: GraphState = {
                    "messages": list(st.session_state.messages),
                    "next_step": "",
                    "tool_result": "",
                    "context_used": "",
                    "node_latencies": {},
                }
                result_state = graph.invoke(state)

            st.session_state.last_latencies = result_state.get("node_latencies", {})
            context_used = result_state.get("context_used", "")
            raw_response = result_state["messages"][-1]["content"]

            # Stream the response
            full_response = ""
            if isinstance(raw_response, str):
                full_response = raw_response
            else:
                for chunk in raw_response:
                    if hasattr(chunk, "content"):
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            # Faithfulness check
            final_response = check_faithfulness(full_response, context_used)
            if final_response != full_response:
                placeholder.markdown(final_response)

            _accumulate_cost(final_response)

            result_state["messages"][-1]["content"] = final_response
            st.session_state.messages = result_state["messages"]

            # Update Redis memory summary in background
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            summary = memory.build_summary(
                [m for m in st.session_state.messages if isinstance(m["content"], str)],
                openai_client,
            )
            if summary:
                memory.save_summary(session_id, summary)

        st.rerun()


if __name__ == "__main__":
    main()
