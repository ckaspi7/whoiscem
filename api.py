"""Headless FastAPI service layer (Phase 4.1).

Unlocks three things the Streamlit app alone cannot provide: a real
``/healthz`` a load balancer or CI can poll, headless ``/chat`` access for
automated trajectory evaluation (Phase 3.5) and load testing without driving
a browser, and the layer where auth, rate limiting, and a daily spend cap
actually belong. Building those into chatbot.py's Streamlit UI code would
mean every future headless caller — this service, a future CLI, a load
test — has to reimplement them; enforcing them once, at the boundary every
caller shares, is the point of a service layer.

Deliberately not yet wired as what the deployed Streamlit app calls instead
of the graph directly ("Streamlit becomes a client", per the plan this
follows) — that changes the production deployment topology (where does this
run relative to Streamlit Cloud? how do the two talk to each other?), which
is a real infrastructure decision, not an implementation detail, and stays a
Phase 4.6 conversation rather than being decided here. Usable standalone
right now:

    uvicorn api:app --reload
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
      -d '{"message": "Where does Cem work?"}'
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from chatbot import MAX_INPUT_CHARS
from config import load_settings
from cost import estimate_cost
from observability import set_session_id, setup_logging, setup_tracing
from rate_limit import RateLimiter
from retrieval.backends import create_qdrant_client
from spend_tracker import SpendTracker

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_PER_MINUTE = 20
DEFAULT_DAILY_SPEND_CAP_USD = 5.0

_rate_limiter = RateLimiter(
    limit=int(os.environ.get("RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE)),
    window_seconds=60,
)


def _daily_spend_cap_usd() -> float:
    return float(os.environ.get("DAILY_SPEND_CAP_USD", DEFAULT_DAILY_SPEND_CAP_USD))


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_tracing()
    app.state.spend_tracker = SpendTracker()
    # Built once at startup, not per request: create_assistant() triggers the
    # retrieval index's first-run setup, which is expensive and must not
    # repeat on every call. Imported here (not at module level) so importing
    # api.py itself — for tests that never start the lifespan — stays cheap
    # and does not require chatbot.py's own heavy imports (Streamlit et al.)
    # or a real OPENAI_API_KEY just to collect tests.
    from chatbot import create_assistant

    app.state.graph = create_assistant()
    app.state.chat_model = load_settings().chat_model
    yield


app = FastAPI(title="whoiscem headless API", lifespan=lifespan)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    route: str
    faithfulness_score: int | None
    cost_usd: float


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = os.environ.get("API_KEY", "")
    if not configured:
        # No key configured: auth is off. Loud, not silent — this is meant
        # for local dev, and a deployment that forgets to set API_KEY is
        # exactly the "no rate limit, no auth" cost-DoS gap the plan calls
        # out for the app as it stood before this file existed.
        logger.warning("API_KEY is not set — /chat is running with no authentication.")
        return
    if x_api_key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _client_key(request: Request, x_api_key: str | None) -> str:
    """Rate limit by API key when one is configured, so a client keeps its
    own bucket even if its IP changes; by client IP otherwise."""
    if x_api_key:
        return f"key:{x_api_key}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def _initial_state(messages: list) -> dict[str, Any]:
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


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Real dependency checks, not just "the process is up" — a load
    balancer or CI polling this should see whether the app can actually
    answer a question, not just whether Python is running."""
    checks: dict[str, Any] = {"openai_api_key": bool(os.environ.get("OPENAI_API_KEY"))}

    try:
        create_qdrant_client(load_settings()).get_collections()
        checks["qdrant"] = True
    except Exception as exc:
        checks["qdrant"] = False
        logger.warning("Health check: Qdrant unreachable: %s", exc)

    settings = load_settings()
    if not settings.redis_url:
        checks["redis"] = "not configured (session memory runs in-process)"
    else:
        try:
            import redis as redis_lib

            redis_lib.from_url(settings.redis_url, socket_connect_timeout=2).ping()
            checks["redis"] = "reachable"
        except Exception:
            checks["redis"] = "unreachable (in-process fallback active)"

    # Redis is informational only, same as everywhere else in this project —
    # its absence degrades a feature, it does not make the app unhealthy.
    healthy = bool(checks["openai_api_key"]) and bool(checks["qdrant"])
    return JSONResponse(status_code=200 if healthy else 503, content={"healthy": healthy, "checks": checks})


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
    _auth: None = Depends(_require_api_key),
) -> ChatResponse:
    key = _client_key(request, x_api_key)
    if not _rate_limiter.allow(key):
        retry_after = _rate_limiter.retry_after(key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded, retry in {retry_after:.0f}s",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    spend_tracker: SpendTracker = request.app.state.spend_tracker
    spent_today = spend_tracker.today_total()
    cap = _daily_spend_cap_usd()
    if spent_today >= cap:
        raise HTTPException(
            status_code=402,
            detail=f"Daily spend cap of ${cap:.2f} reached (${spent_today:.4f} spent today)",
        )

    # A fresh, disposable id per call: this endpoint is stateless by design
    # (see the module docstring on multi-turn scope) — `history` carries
    # context explicitly, the way eval/run_eval.py's golden set does, rather
    # than this service adopting Streamlit's session-memory model.
    set_session_id(f"api-{int(time.time() * 1000)}")

    messages = [
        (HumanMessage if turn.role == "human" else AIMessage)(content=turn.content) for turn in body.history
    ]
    messages.append(HumanMessage(content=body.message))

    result_state = request.app.state.graph.invoke(_initial_state(messages))

    token_usage = result_state.get("token_usage") or {}
    judge_usage = token_usage.get("check_faithfulness", {"input": 0, "output": 0})
    graph_input = sum(u["input"] for k, u in token_usage.items() if k != "check_faithfulness")
    graph_output = sum(u["output"] for k, u in token_usage.items() if k != "check_faithfulness")

    cost_usd = estimate_cost(request.app.state.chat_model, graph_input, graph_output) + estimate_cost(
        "gpt-4o-mini", judge_usage["input"], judge_usage["output"]
    )
    spend_tracker.add(cost_usd)

    return ChatResponse(
        response=result_state["messages"][-1].content,
        route=result_state.get("route", ""),
        faithfulness_score=result_state.get("faithfulness_score"),
        cost_usd=round(cost_usd, 6),
    )
