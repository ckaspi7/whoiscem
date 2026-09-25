"""GraphState lives in its own module, deliberately separate from chatbot.py.

Found by actually adding a `pages/` directory (Phase 5's eval dashboard) and
running the test suite, not documented anywhere in advance: the instant a
`pages/` directory exists as a sibling of chatbot.py, Streamlit switches to
its multi-page-app execution mode for the entry-point script — and its page
runner does not give the executing script the same module identity its
single-page runner does. `chatbot.py` uses `from __future__ import
annotations` file-wide, which defers every annotation to a string re-resolved
later; LangGraph's own `StateGraph(GraphState)` calls `get_type_hints(...,
include_extras=True)` to read them, and under MPA mode that resolution
landed in a namespace without `Annotated`/`BaseMessage`/`add_messages` in
it — `NameError: name 'Annotated' is not defined`, breaking `create_assistant()`
and therefore the entire app, for a reason that has nothing to do with the
dashboard page's own content (an empty `pages/` directory reproduces it).

A TypedDict in its own, normally-imported module sidesteps the question
entirely: Python's ordinary import machinery always registers a module under
its real name in `sys.modules`, unlike whatever identity Streamlit's MPA
script-runner gives the entry-point script — so `get_type_hints` resolves
this module's own globals correctly regardless of how any caller executes.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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
