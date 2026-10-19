---
paths:
  - "api.py"
  - "rate_limit.py"
  - "spend_tracker.py"
  - "cost.py"
  - "Dockerfile.api"
  - "cloudbuild.yaml"
  - "tests/**/test_api*.py"
  - "tests/**/test_rate_limit*.py"
---

# Headless API (Phase 4.1)

`api.py` — a second, independent surface onto the same graph (`uvicorn
api:app`), built for headless access (load testing, Phase 3.5's trajectory
eval) and, since the standalone-frontend work, a real public backend that
only ever expects to be called through its frontend's own same-origin
server-side proxy rather than directly by a browser — still not what the
deployed Streamlit app calls. `lifespan()` builds the graph once at startup
via `chatbot.create_assistant()`, not per request. A `CORSMiddleware` scoped
to `FRONTEND_ORIGIN` wraps the whole app, but it is defense-in-depth, not the
real gate: the intended caller is server-to-server (the proxy), which CORS
does not govern at all, so this only stops a stray direct browser call.
`POST /chat` applies, in order: `_require_api_key` (a `Depends`, checked
against `API_KEY` — unset logs a warning and allows the request through
rather than failing closed), `RateLimiter.allow()` (`rate_limit.py`,
in-process sliding window, keyed by the real visitor IP — `X-Forwarded-For`
first, then the connecting socket — and never by the API key, since this
deployment has exactly one shared key and keying by it would merge every
visitor into a single bucket), then `SpendTracker.today_total()` against
`DAILY_SPEND_CAP_USD` (returns 402 once reached) before invoking the graph at
all. The graph's raw answer is then scored and tiered via
`guardrails.faithfulness_check` — reusing `state["faithfulness_score"]` when
classifier mode's resume-route retry already computed one for this exact
answer, scoring fresh otherwise — so a `/chat` caller gets the same
unchanged/warned/refused guarantee a Streamlit visitor already gets, not a
raw, unguarded answer. Real per-node `token_usage` from the resulting
state — the same mechanism `chatbot.py`'s sidebar uses — prices the call via
`cost.estimate_cost` and records it via `SpendTracker.add()`, which is
Redis-backed like `SessionMemory` (unlike `RateLimiter`, which has no
Redis-backed mode — see LIMITATIONS.md for why that asymmetry is deliberate).
`GET /healthz` checks Qdrant reachability and `OPENAI_API_KEY` presence for
real (not just "the process is up"); Redis is reported but never fails the
check, matching Redis's optional status everywhere else in this project.

**The deployed Streamlit app does not go through this file.** `chatbot.py`
still calls `create_assistant()`/`graph.invoke()` directly and has its own,
separate cost meter — the two do not share a spend cap. See
LIMITATIONS.md's "The Headless API's Protections Do Not Yet Cover the
Deployed Streamlit App" before assuming otherwise.
