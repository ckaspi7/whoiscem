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

## Cost Accounting

**Status: the number in the sidebar is wrong, and knowingly so.**

`_accumulate_cost` estimates tokens as `len(text) // 4` and applies the *output*
price to all of them. Four of the five model calls per turn — the router, the
faithfulness judge, the summariser and the embeddings — are not counted at all.
`tiktoken` is a dependency and unused; the API already returns exact usage.

It is displayed to five decimal places, which implies a precision it does not
have. Replacing the estimate with reported usage is scheduled; until then, read
the figure as a lower bound of the wrong quantity.

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

A single question through the graph costs roughly 2 requests (route + generate). A full
RAGAS pass costs far more than "49 questions × 4 metrics" suggests: `faithfulness` alone
decomposes each answer into claims and verifies each one as a separate call, so one full
evaluation run is plausibly 500-1,000+ requests. Running several of those back to back —
which happened once, during active Phase 3 development — exhausted the daily cap outright;
`eval/compare.py`'s gate then failed on a run that had produced no output at all, and a
local run showed climbing per-question latency before hitting a wall.

`python eval/run_eval.py --no-ragas` (routing and retrieval only, ~120 requests) is the
correct tool for iterative verification. Full RAGAS runs are for confirming a baseline
immediately before a commit, not for every intermediate check.
