# 0008 — A local classifier replaces the LLM judge for the faithfulness guardrail

**Status:** Accepted — measured, including a correction to the first measurement
before trusting it.

## Context

A real, reported bug motivated re-examining the faithfulness guardrail, not a
hunch: a visitor watched a real answer stream in ("...has beginner proficiency
in Spanish" — a detail nowhere in the source data), then get retracted by the
post-hoc faithfulness check once it finally ran. The streaming/reveal timing
was fixed separately, but investigating it surfaced two things worth fixing
about the guardrail itself: `guardrails/faithfulness_check.py`'s gpt-4o-mini
judge was the single largest latency contributor in a real traced turn (57%
of total time, more than generation), and re-running that exact case showed
the judge scoring it 4/5 — passing — the answer that started this
investigation was, and had been, slipping through in production.

## Decision

Replace the gpt-4o-mini LLM-as-judge with Vectara's HHEM-2.1-Open, a small
local classifier trained specifically to score whether a hypothesis is
consistent with a premise, run the same way this project already runs the
retrieval cross-encoder (ADR-0003): loaded once, locally, no per-call API cost.

This was measured twice, not once, and the first measurement was not trusted
until re-checked at scale:

- **A 14-case hand-built pilot** (short, synthetic one-line contexts) showed
  100% accuracy and ~10x lower latency than the judge. Encouraging, but small
  and built by the same person investigating it — exactly the conditions
  under which a result should be treated as a lead, not a conclusion.
- **A 32-case set generated from the real golden set** (`eval/
  build_faithfulness_test_set.py`, real production context, real
  per-route retrieval) told a more honest story: **87.5% accuracy vs the
  judge's 62.5%**, and only **1.9x faster (520ms vs 989ms mean)**, not 10x —
  real multi-paragraph context costs the classifier more compute than the
  pilot's synthetic one-liners did, while the judge's cost is mostly a fixed
  API round-trip that doesn't shrink the same way. Building this set also
  found two real bugs in it before trusting any number from it: four
  golden-set answers turned out to be templated non-answers or to reference a
  field (birth year) that `tools/personal_tool.py`'s `_SAFE_FIELDS`
  deliberately excludes from context — pairing either with "faithful" would
  have mislabeled the ground truth, not measured either model.

87.5%, not 100%, is the number this decision is actually based on — still a
clear, real improvement (roughly a third the miss rate of what's live today),
not the clean sweep the small pilot suggested.

## Consequences

- **`trust_remote_code=True` is a real trust boundary, pinned deliberately.**
  Loading HHEM runs Vectara's own Python code, not just weights.
  `guardrails/faithfulness_check.py` pins an exact reviewed revision hash
  rather than always resolving to whatever is currently on the model's
  default branch, so an upstream change to their repo can never silently
  change what this app executes.
- **The continuous 0–1 score is mapped onto the existing 1–5 scale**
  (`_hhem_to_five_scale`) so every existing caller — `apply_faithfulness_tiering`,
  the in-graph self-correction check (ADR-0005), `eval/run_eval.py` — needed
  no changes at all. Only the binary faithful/unfaithful split (threshold
  0.20) was actually measured; the three-way refuse/warn/pass banding is a
  reasoned margin either side of that boundary, not independently validated
  as its own split. Worth revisiting if the warn tier fires far more or less
  than the ~10-15% of turns the measured overlap zone would predict.
- **No resume-route pairs were measured.** This machine's network cannot
  reach the Qdrant Cloud cluster the deployed app uses (a corporate-proxy
  block, documented elsewhere, unrelated to this decision) — re-run
  `eval/build_faithfulness_test_set.py` once that's not a constraint, or from
  an environment that can reach it, before treating personal/spotify/linkedin
  coverage as representative of resume's longer, denser prose.
- **The same discipline ADR-0006 already established still holds, now
  running on a different mechanism**: the judge fails open (a model load or
  prediction failure returns `None`, "nothing to act on," never a low score),
  a tool error still can never reach it as evidence. Fail-open logs a warning
  now for a different reason (`transformers` import or `.predict()` raising,
  not an OpenAI API exception) but the contract callers see is identical.
- **Real cost accounting (Phase 4.2) now shows $0 for this line item, honestly**,
  not as an omission — `score_faithfulness_with_usage` still returns a usage
  dict for every existing caller's sake, it's just always zero.
- The 2026 hallucination-detection literature's own caution applies here too:
  small classifiers can perform strongly in-domain and still fail to
  generalize outside the data they were checked against. 87.5% on 32 real
  but self-generated pairs is a measured result, not a guarantee — the honest
  position is "meaningfully better, not perfect," and that is the position
  actually being shipped.
