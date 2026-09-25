# Limitations & Design Decisions

## Spotify Data

**Status:** Cached snapshot, refreshed manually. The tool reports the snapshot's
real age in the text it hands the model — it previously said "refreshed monthly"
regardless of how old the data actually was.

**Why not live?** Spotify's OAuth flow requires a redirect URI that resolves to a real server.
Stateless Streamlit Cloud deployments cannot receive OAuth callbacks, so the live token flow
cannot complete in the deployed environment.

**What's actually there:** `data/cache/spotify_cache.json` contains a real export from
Cem's Spotify account, generated locally using `scripts/refresh_cache.py`, which implements
the full OAuth flow (see that file for details). The cache includes a `cached_at` timestamp
that is displayed in the UI so users always know how fresh the data is.

**To update:** Run `python scripts/refresh_cache.py` locally, then commit the updated
`data/cache/spotify_cache.json`. The deployed app picks it up on next restart.

---

## LinkedIn Data

**Status:** Manually curated snapshot, refreshed when career information changes,
with its real age reported to the model.

It is reconciled against `data/resume.md`, because the two disagreed: the cache
had the Mercedes-Benz co-op as an 8-month internship rather than two co-op terms
spanning January 2018 to August 2019, and NeoWise running to January 2022 rather
than May 2021. The same question routed to different tools returned different
answers, which no amount of retrieval tuning can fix. The cache carries one fact
the resume does not — the three internal TELUS titles — which is the reason this
route exists.

**Why not live?** LinkedIn's official API removed personal profile access for most developers
in 2023. Unofficial scraping libraries violate LinkedIn's ToS and are fragile.

**What's actually there:** `data/cache/linkedin_cache.json` is manually maintained to reflect
Cem's current career history, with a `cached_at` timestamp displayed in the UI.

**To update:** Edit `data/cache/linkedin_cache.json` directly and run
`python scripts/refresh_cache.py` to bump the timestamp.

---

## Vector Store Persistence

The Qdrant vector store is populated from the resume PDF (`RESUME_PATH`) the first
time a query needs it, and rebuilt automatically whenever the collection is missing.

Which Qdrant it talks to is configuration, not code: `QDRANT_MODE` selects an
embedded on-disk store (the default — no server), a Qdrant reached over HTTP, or
Qdrant Cloud. Embedded mode takes an exclusive lock on its directory, so exactly
one process may use a given `QDRANT_PATH` at a time.

**Not yet done:** the deployed Streamlit Cloud app has no persistent volume, so
embedded storage there is rebuilt on every cold start. Pointing it at Qdrant Cloud
(`QDRANT_MODE=cloud`) is the fix and is not yet in place.

---

## Session Memory Persistence

With `REDIS_URL` set, rolling summaries live in Redis with a 30-day TTL. Without
it, they fall back to an in-process store: the feature still works, but summaries
are lost on restart and are not shared between processes. The sidebar shows which
backend is live so the deployed behaviour is never a guess.

---

## Session Links Are Signed, Not Authenticated (Phase 4.5)

`?sid=` used to be read straight off the URL and used directly as the Redis key,
with no check that the server had ever issued it — editing the query param to
any string, guessed or fabricated, was accepted outright as a previously-issued
session. It is now signed (`chatbot._sign_session_id`/`_verify_session_id`, HMAC-
SHA256): a fabricated or edited `sid` fails verification and a fresh session is
issued instead, closing that specific IDOR.

What this does not do: prevent someone who has a *legitimately issued, full*
signed link from opening that session. Sharing a session via URL is a documented
feature, not the bug being fixed, and the signature cannot distinguish "the
owner shared this on purpose" from "this leaked." That would need binding a
session to an authenticated identity, which this project deliberately has none
of — no signup, no login, matching its zero-account-friction design.

With no `SESSION_SECRET` configured, signing falls back to a secret generated
once per process start: signed links keep working for that process's lifetime
and stop verifying after a restart (a visitor silently gets a fresh session,
losing their rolling summary — the same category of degradation as Redis being
absent, not a crash). Set `SESSION_SECRET` for any deployment expected to survive
a restart with existing links still valid.

---

## Tracing

**Status: not wired yet.** The README no longer lists tracing as a live
capability, because it is not one.

The project is standardising on Arize Phoenix, instrumented through
OpenInference over OpenTelemetry, so the trace backend is a configuration choice
rather than a hard dependency — the same shape as the vector store and session
memory. Phoenix runs locally with no account, so tracing will not become another
thing a reviewer has to sign up for in order to run this.

LangSmith was removed rather than kept as a second target. It was configured
with a placeholder key and returned 403 on every request, which is worse than
no tracing: it looks configured. One platform, built properly, beats two
half-wired.

---

## Cost Accounting (Phase 4.2 — fixed, with two real gotchas found by running it)

**Status: real per-call usage, not a length estimate.** Every node that calls a
model (`condense_query`, `route_query`/`agent`, `generate_response`, a
self-correction retry, the faithfulness judge, the session summariser) now
reads its own response's real token counts and prices them at whichever model
actually made the call — `cost.py` holds the published per-model rates.
Embeddings are the one deliberate exception: `text-embedding-3-small` costs
~$0.02/1M tokens on a handful of tokens per query, several orders of magnitude
below either chat model, and often skipped entirely by `RetrievalCache` — not
worth a real code path.

Two things were not visible until this was actually built and run, not
assumed from documentation:

- **`langchain_community`'s `get_openai_callback()` does not propagate through
  LangGraph's node execution.** The obvious, minimally-invasive design —
  wrap the whole `graph.stream()` call in one `with get_openai_callback() as
  cb:` block — silently reported 0 tokens for every node's call despite a real
  model call happening and a real answer coming back. Confirmed directly with
  a live call, not inferred from a changelog. The fix: every node reads its
  own response's `usage_metadata` right where it makes the call
  (`cost.usage_from_response`, merged into state by node via `_add_usage`),
  the same pattern `check_faithfulness_node` already used for its own
  (non-LangChain) call.
- **`ChatOpenAI(streaming=True)` returns no usage data on `.invoke()` unless
  `stream_usage=True` is also set** — it defaults to `False`. Without it,
  `generate_response`'s own call (the actual answer) read as 0 input/0 output
  tokens despite a real, priced completion coming back — the node's own
  underlying HTTP request never asked OpenAI's streaming endpoint to include
  a usage object, which the API only attaches when explicitly requested.
  `create_assistant`'s `_llm` helper sets `stream_usage=True` unconditionally
  now, harmless on the one non-streaming construction (`llm_fast`).

The sidebar's help text says what it excludes rather than implying completeness
it doesn't have.

---

## Guardrails: Three of Four Phase 4.3 Items Done

The judge call now sets `response_format={"type": "json_object"}` — previously
absent, so a ```` ```json ```` fence around the reply broke `json.loads` and the
guard disabled itself with no signal anywhere.
`tests/unit/test_guardrails.py::test_malformed_json_fails_open` still asserts
the fail-open *behaviour* is correct (the app must not break), but a failure
there is now logged (`guardrails.faithfulness_check`, level WARNING) instead
of vanishing — a safety gate that can go silently inert is worse than no
gate, and this is what makes that visible instead of mute.

An input length cap (`chatbot.MAX_INPUT_CHARS`, 1000) now rejects an
oversized message before it reaches session state or the graph — no API call
at all, and the oversized text never enters conversation history, where it
would otherwise still cost tokens as context on every future turn even though
the turn that sent it made no call.

**Deliberately not done: a heuristic prompt-injection filter.** The golden
set's adversarial cases (`a001`–`a004`: "ignore your previous instructions",
"developer mode", a false-premise correction attempt, a request to fabricate
a reference letter) already measure at 100% under the model's own judgment
via the system prompt — there is no measured gap a keyword filter would
close. What a naive filter would add is real, unmeasured false-positive
risk: a legitimate question that happens to contain a flagged word ("Did Cem
ever have to *ignore* a flaky test in CI?") gets refused for something it
never did. Adding a filter to say one exists, without a measured case it
improves, is exactly the kind of unmeasured change this project's own
discipline argues against.

**Not done: moving the disclaimer/refusal banner into the graph.** Phase
3.3's score already lives in the graph and drives a real retry; the *tiering*
that turns a low score into a banner still runs in `chatbot.py:main()` after
`graph.stream()` completes, so a user still watches a possibly-ungrounded
answer type out live before it is replaced. Fixing this means giving up live
token-by-token display in favour of buffering the full answer until it is
checked — a real UX trade-off (responsiveness vs. never showing an answer
that gets pulled back), not just a refactor, and not made yet.

---

## Personal Information

The database behind the `personal` route stores no contact details, no date of
birth and no family information. The tool reads an explicit column allowlist
rather than `SELECT *`, so a database that still carries older columns cannot
leak them into a prompt, a log or a trace. `tests/unit/test_no_pii_committed.py`
fails the build if personal data becomes tracked by git, including inside a PDF.

---

## Resume Source

`data/resume.md` is a redacted copy: the phone number and email address are
removed. `RESUME_PATH` overrides it, so a local unredacted file can be indexed
without touching the repository.

---

## Chunking — Currently Degenerate

**Status: measured, unfixed, and the reason the evaluation work comes first.**

`SemanticChunker` with percentile thresholding splits the resume into **three
chunks** of roughly 2,200 / 1,000 / 4,100 characters. The pipeline then takes
the top 20 dense results, the top 20 BM25 results, fuses them, and reranks to
the top 3 — of a corpus of 3. Every query returns the entire document.

So the hybrid retrieval this project is built around is, on this corpus, not
retrieving anything: RRF has nothing to fuse and the cross-encoder has nothing
to discriminate between. BM25 is worse than inert — a term appearing in all
three chunks scores zero IDF, so sparse search regularly returns nothing at all.

Switching the source from PDF to Markdown was expected to help and did not: it
moved the count from two chunks to three. The cause is the chunker's
configuration on a short document, not the extraction format.

This is left in place on purpose. It is the baseline the evaluation harness will
measure, so that section-aware chunking can be reported as a before-and-after
rather than asserted.

---

## LLM-Judge Calibration

RAGAS faithfulness is not comparable across routes, and the aggregate should
not be read as a hallucination rate.

A three-case probe isolates why. Given prose context, the judge scores a correct
answer 1.0 and a deliberately wrong one 0.0 — it works. Given the same fact as a
numbered list (`Top Artists:
1. The Weeknd`), it scores the correct answer 0.0,
because deriving "top artist" from list position is not textual entailment. The
spotify tool emits ranked lists, so its faithfulness (0.42) measures output
format rather than truthfulness.

The `conversation` route scores 0.000 for a different reason: it retrieves
nothing, so no answer on it can be grounded in anything. That is a real property
worth surfacing, not an artifact — it is why a question misrouted onto that path
is the worst failure mode available, and why routing accuracy is tracked
separately.

Results therefore report faithfulness per route as well as in aggregate.
Reformatting tool output as prose would likely raise the number; it is left
alone until it can be run as a measured experiment rather than a metric-driven
edit.

---

## Rank Metrics Are Not Comparable Across `AGENT_MODE`

MRR and RAGAS context precision both look sharply better under
`AGENT_MODE=tool_calling` than under `classifier` (0.566→1.000, 0.646→0.806).
Neither is retrieval improving. `get_resume_info_result` pre-merges every
retrieved chunk into one block with `format_chunks` before it reaches state,
so `tool_calling`'s `context_chunks` is one item per successful tool call, not
several individually-ranked ones — a single block is trivially "rank 1" if the
reference is found in it at all. `eval/compare.py` treats a rank-sensitive
metric the same way whether `agent_mode` changed or the chunker did: reported,
never gated, in either direction.

The same substitution also means `tool_calling`'s routing accuracy is not
fully deterministic between identical runs unless the agent's own
tool-selection call uses `temperature=0` — an early version bound tools to the
same `temperature=0.7` model used for the classifier's final prose, and
routing moved from 93.2% to 91.5% across two runs of unchanged code purely
from sampling. `create_assistant` uses a separate, deterministic, still-
streaming model for the agent's decisions for exactly this reason.

---

## Personal Database Schema Drift

A database created before the schema was slimmed does not gain the `birth_year`
column, since `CREATE TABLE IF NOT EXISTS` does not migrate. The setup script
warns when it detects this. The symptom in evaluation is a question like "what
year was Cem born" answered correctly from a hardcoded fact in the system prompt
while scoring 0.0 faithfulness — correctly, because the answer is not grounded
in anything retrieved. Delete `data/user_data.db` and re-run
`scripts/setup_user_data_db.py` to fix.

This also exposes a design issue worth naming: facts hardcoded in the system
prompt are invisible to the faithfulness judge, which only sees retrieved
context. Any answer drawn from them scores as unsupported.

---

## Cross-Encoder Reranker

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downloads ~90 MB on first run
and runs fully locally with no API cost. Subsequent runs use the cached model.
Cold-start on a fresh container adds ~15 seconds to first query latency.

---

## Evaluation Is Rate-Limited, Not Just Slow

Measured, not assumed: OpenAI's Tier 1 caps `gpt-4o-mini` at 10,000 requests per day,
shared across every caller on the account — local development, CI's `eval-gate`, and the
deployed app all draw from one pool. Reaching Tier 2 needs $50 of lifetime spend and a
7-day-old account; deliberately spending toward that would contradict this project's own
free-tier constraint, so the actual fix is pacing, not upgrading.

The cap is a **rolling 24-hour window, not a fixed daily reset** — confirmed from the
account's own `x-ratelimit-reset-requests` response header, not assumed from the "per
day" name. Usage ages out gradually rather than zeroing at midnight, so "it'll reset
overnight" is approximately, not exactly, right. A single minimal request shows the real
number where guessing does not:

```bash
curl -s -D - -o /dev/null https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  | grep -i ratelimit
```

A single question through the graph costs roughly 2 requests (route + generate). A full
RAGAS pass costs more than "49 questions × 4 metrics" suggests, and more than this project
first estimated: `faithfulness` alone decomposes each answer into claims and verifies each
one as a separate call, and a measured full run — not a guess — costs roughly **2,700
requests**, not the 500-1,000 originally estimated. Two such runs back to back still fit
comfortably inside a 10,000 budget on their own; what exhausted the cap outright once,
during active Phase 3 development, was several full runs compounding within one rolling
window. `eval/compare.py`'s gate then failed on a run that had produced no output at all.

RAGAS's own concurrency separately trips the *per-minute* RPM/TPM caps (500 requests/min,
200K tokens/min) near the end of a scoring pass most of the time — a different, far more
benign thing: the retry-after is milliseconds, RAGAS's own job runner recovers on its own
without help, and it says nothing about the day's budget. Only a `requests per day (RPD)`
429 is the wall worth stopping for.

`python eval/run_eval.py --no-ragas` (routing and retrieval only, ~120 requests) is the
correct tool for iterative verification. Full RAGAS runs are for confirming a baseline
immediately before a commit, not for every intermediate check.

---

## Self-Correction Retries Only Have One Lever

Phase 3.3's retry (reformulate the resume query, re-search, regenerate on a
low faithfulness score) cannot fix an answer that is bad for a reason other
than "the resume search was worded badly." Measured directly, not assumed:
the one RAGAS-scorable question that triggered a retry in a full run was a
multi-intent question needing `linkedin` data — classifier mode never fetches
a second tool's data — so reformulating the resume query a second time cannot
supply a source it never queried in the first place. The retry correctly
identified a bad answer and correctly tried the only lever available to it;
that lever was not the fix this case needed. This is `classifier` mode's
already-documented multi-intent limitation compounding with itself, not a bug
in the retry logic — see the README's self-correction section for the actual
before/after numbers this produced.

The retry is scoped to the `resume` route specifically, and this is
structural, not an oversight: `personal`, `spotify`, and `linkedin` are fixed
lookups (the same `info_type`, or no argument at all, every time), so a retry
there would spend a judge call and a generation call to reproduce byte-
identical content. `conversation` has no retrieved context to reformulate at
all.

---

## `CHAT_MODEL=gpt-6-luna` Only Works Correctly Under `AGENT_MODE=classifier`

Measured, not assumed (`eval/results/v10-gpt4o-mini-baseline.json` /
`v11-gpt6-luna-comparison.json`): under `classifier` mode, gpt-6-luna matches or beats
gpt-4o-mini on every gated metric — see the README comparison. Under `tool_calling` mode,
the identical prompt and addendum that scores well under `classifier` misroutes basic
career questions to `get_linkedin_info` instead of `get_resume_info` (40% routing on a
5-question sample, reproduced identically twice). Not investigated further: `tool_calling`
is not the default `AGENT_MODE`, and re-tuning `_TOOL_CALLING_ADDENDUM` specifically for
one model's quirks would be prompt-overfitting to that model — the same mistake this
project already avoided once with the golden set itself.

`CHAT_MODEL` therefore stays at `gpt-4o-mini` by default: it is orthogonal to
`AGENT_MODE`, so one global default cannot express "good under classifier, bad under
tool_calling." Set `CHAT_MODEL=gpt-6-luna` explicitly for a `classifier`-mode deployment;
leave it unset if `AGENT_MODE=tool_calling`.

Two real API constraints, found by running it rather than documented anywhere in advance:

- Tool/function calling only works at `reasoning_effort="none"` — silently breaks
  otherwise. `create_assistant` sets it automatically, only for the tool-bound agent.
- Any non-default `temperature` is rejected outright (a live 400: `'temperature' does not
  support 0.0 with this model. Only the default (1) value is supported`), including the
  `temperature=0` this project otherwise relies on for deterministic routing and tool
  selection. `create_assistant` omits the override for this model instead of crashing —
  its routing determinism at temperature=1 is therefore measured over one run, not yet
  confirmed stable across repeats the way gpt-4o-mini's temperature-driven nondeterminism
  was originally caught (93.2%→91.5% between two identical runs).
