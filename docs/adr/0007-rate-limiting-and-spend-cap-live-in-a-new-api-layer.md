# 0007 — Rate limiting and the spend cap live in a new API layer, not in Streamlit directly

**Status:** Accepted — with a scope boundary that is deliberately not yet
closed. Read the Consequences section before assuming this is fully solved.

## Context

The app is deployed publicly on its owner's own OpenAI API key. Every turn
makes several model calls (routing or agent decision, generation, the
faithfulness judge, the session summariser), and until this decision there
was no rate limit and no daily spend cap anywhere in the codebase — an
unbounded number of visitors, or one visitor in a tight loop, could run up
an unbounded bill. The only existing protection was `chatbot.MAX_INPUT_CHARS`
(Phase 4.3), which bounds one request's cost, not how many requests happen.

## Decision

Build the auth, rate limiting, and spend cap into a new, independent FastAPI
service (`api.py`), rather than adding them as Streamlit-specific code inside
`chatbot.py:main()`.

`api.py` exposes `POST /chat` and a real `GET /healthz`, with API-key auth
(optional — unset logs a loud warning rather than failing closed, for local
dev), an in-process sliding-window rate limiter (`rate_limit.py`, keyed by
API key or client IP), and a daily spend cap (`spend_tracker.py`, Redis-backed
like session memory) checked *before* the graph runs at all, using the same
real per-call token usage `cost.py` already computes for the Streamlit
sidebar. Building this at a shared boundary means every future headless
caller — this service, a future CLI, a load-testing script, Phase 3.5's
trajectory eval — inherits the same protections for free, instead of each
one reimplementing them, which is what embedding this logic inside
`chatbot.py`'s UI code specifically would have meant.

## Consequences

**This is the most important consequence to state plainly: the currently
deployed Streamlit app does not go through `api.py` and inherits none of
these protections yet.** The plan's own framing is "Streamlit becomes a
client" of this service — that rewiring is a real infrastructure decision
(where does this run relative to the app's actual deployment? how do the two
processes talk to each other?), not an implementation detail, and is
deliberately left to whoever makes that deployment decision rather than
assumed here. Until that happens:

- The deployed Streamlit app's only cost protection remains
  `MAX_INPUT_CHARS`. The original unbounded-request-volume risk this
  decision exists to close is closed for API traffic, not yet for the app a
  visitor would actually reach.
- `api.py`'s `SpendTracker` and the Streamlit sidebar's cost meter are two
  separate, unrelated running totals, even pointed at the same Redis
  instance — neither one's cap constrains the other's spend.
- `RateLimiter` has no Redis-backed mode, unlike `SpendTracker`, deliberately:
  a portfolio deployment's actual scale is one instance, and distributed
  correctness for a scale this project does not operate at would be effort
  spent on the wrong thing. `SpendTracker` is Redis-backed because an
  in-process-only spend cap resets on every restart, which undermines the
  one thing it exists to guarantee.
