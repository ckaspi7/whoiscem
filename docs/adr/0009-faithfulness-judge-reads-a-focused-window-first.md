# 0009 — The faithfulness judge can read a focused window of context first

**Status:** Accepted as an option, **not enabled**: the default stays `full`. The
measurement overturned the hypothesis that motivated it. What the windowed mode
buys is speed and fewer false warnings, not accuracy, and it costs detection.

## Context

Locally, correct resume answers were arriving with the "not fully confident"
banner and taking 20-40 s. Diagnosing it (`scripts/diagnose_answers.py`)
separated two causes. The first was environmental and is fixed elsewhere: this
machine's proxy answers 403 for Qdrant Cloud, and a retrieval failure used to
surface as the model's ordinary "I don't have that information". The second
was the guardrail itself: on the resume route the judge scores against the
whole resume (7,610 characters, over 2,000 tokens, because the `auto` retrieval
strategy returns every chunk), and that is slow, 9-13 s per call.

ADR-0008 had flagged the gap that made this invisible: its 87.5% was measured on
32 pairs from the linkedin, personal and spotify routes (contexts of 333-1,346
characters) and **no resume-route pairs at all**, because retrieval was
unreachable when the set was built. The disk fallback for resume retrieval
removed that constraint, so `eval/build_faithfulness_test_set.py` gained an
incremental mode (existing pairs are kept; only missing questions are added)
and now covers the resume route: 23 questions, each paired with its
hand-verified answer (faithful) and an LLM-corrupted variant that adds one
plausible invented detail (unfaithful).

The first hypothesis came from three examples: scored against only the top few
chunks, verbatim-correct answers jumped from 0.19 and 0.34 to 0.97 and 0.83, so
a focused context would fix the judge's accuracy. Three examples chosen because
they had been flagged is not evidence.

## Decision

`hhem_score` takes a `mode`. `focus_then_full` reads the leading ~2,000
characters of context first (retrieval returns chunks most-relevant-first, so
that is the best evidence) and only a score below the pass line pays for a
second look at the whole context; the higher of the two scores is kept. Contexts
shorter than the window, which is every context on the other three routes, are
scored exactly as before. `DEFAULT_CONTEXT_MODE` stays `full`, so nothing about
what production scores changes; see "Why it is not the default" below.

## What was measured

`eval/ablate_judge_context.py`, 46 resume-route pairs (23 faithful, 23 invented
detail), the production tiers (pass at 0.40, refuse below 0.15):

| Strategy | Faithful: clean / warned / refused | Invented detail: missed / warned / refused | Time, faithful answer |
|---|---|---|---|
| `full` (before) | 21 / 2 / 0 | 5 / 8 / 10 | 13.3 s |
| `focus` at 2,000 chars | 20 / 0 / 3 | 5 / 6 / 12 | 1.6 s |
| `focus_then_full` at 1,500 | 23 / 0 / 0 | 7 / 8 / 8 | 3.8 s |
| **`focus_then_full` at 2,000** | **23 / 0 / 0** | **7 / 8 / 8** | **3.1 s** |
| `focus_then_full` at 3,000 | 23 / 0 / 0 | 7 / 9 / 7 | 3.2 s |

Accuracy at the pass line is 85% for `full` and for `focus_then_full`, 83% for
`focus`. The three other routes, whose contexts are all shorter than the
window, are identical under every strategy (the no-regression check).

## What the measurement said

- **It did not improve accuracy.** Accuracy is unchanged (85% either way) and
  detection got slightly worse: two more invented details pass clean (5 to 7 of
  23), where the whole-resume score would have at least attached a warning.
  The two windows make *different* errors rather than one being better: the
  window alone refuses three faithful answers whose evidence sits outside it
  (escalation recovers all three), while taking the higher of two noisy scores
  also lifts a couple of the unfaithful ones.
- **What it does buy** is time and a quieter UI. A faithful answer is judged in
  3.1 s instead of 13.3 s (an invented detail still pays for both looks,
  about 12 s), the two spurious warnings on correct answers disappear, and the
  in-graph self-correction retry (ADR-0005), which fires on any score below the
  pass line, fires less often on answers that were already right.
- **Why it is not the default.** A spurious banner is a nuisance; an invented
  detail about the owner's career passing with no banner at all is the failure
  the guardrail exists to catch, and it is the one this mode makes more likely.
  The offsetting benefit is two pairs out of 23, which is within noise, and the
  latency it saves (13 s to 3 s per faithful answer) was measured on one
  development laptop, not on a deployment. A planned change of generator model
  will also change which answers are faithful at all, so the pairs measured here
  describe the current model.
- **Turning it on** is one constant, `DEFAULT_CONTEXT_MODE = "focus_then_full"`.
  Re-run `eval/ablate_judge_context.py` first, on more pairs (50 or more per
  class) and ideally on unfaithful answers the production model actually
  produced, and on the replacement generator once there is one.

## Consequences

- **The evidence is small and synthetic.** 23 pairs per class; a two-pair
  difference is within noise. The unfaithful answers are LLM-corrupted ground
  truth, not production failures. Treat the table as a direction, not a rate.
- **Three golden-set sources were ruled out** once resume pairs existed (a003,
  m001, m002): their ground truths describe desired behaviour or need Spotify
  and LinkedIn context, so no judging strategy could rate them faithful against
  the resume alone. Scored 0/3 under every strategy, correctly. The same class
  of mislabel ADR-0008 found for p004-p008.
- **The three-way tiering is still not independently validated.** On these pairs
  the 0.40 pass line behaves sensibly; the warn band's value as its own signal
  is unproven. Under `full` it held 8 invented details against 2 faithful
  answers, so it was informative, and `focus_then_full` keeps the same 8.
- **`eval/ablate_faithfulness.py` is stale.** Its "judge" column now calls the
  same HHEM scorer, so it no longer compares two different methods. Left as the
  record of ADR-0008's comparison.
- **Not tried:** claim-by-claim checking, which could catch invented details a
  whole-answer score cannot, and speeding the model itself (quantisation).
