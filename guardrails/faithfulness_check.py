from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_hhem = None  # lazy-loaded local classifier; see _get_hhem()
_LOW_SCORE_RESPONSE = "I don't have reliable information about that in my knowledge base."
_WARN_PREFIX = "⚠️ *Note: I'm not fully confident in this answer — please verify independently.*\n\n"
_NO_USAGE = {"input": 0, "output": 0}

# vectara/hallucination_evaluation_model (HHEM-2.1-Open) replaced an
# LLM-as-judge (gpt-4o-mini) here after eval/ablate_faithfulness.py measured
# it more accurate (87.5% vs 62.5% on 32 real production-context pairs) and
# faster (1.9x — smaller than a first hand-built pilot suggested, since real
# multi-paragraph context costs HHEM more compute than that pilot's one-line
# examples did) with no per-call API cost. See eval/results/ablation-
# faithfulness.json and docs/adr/0008-local-classifier-replaces-llm-judge.md.
#
# trust_remote_code=True means this also runs Vectara's own Python code, not
# just weights — pinned to a specific reviewed revision so an upstream change
# to their repo can never silently change what runs here.
_HHEM_MODEL_ID = "vectara/hallucination_evaluation_model"
_HHEM_REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"

# Where a continuous 0-1 consistency score lands on the 1-5 scale every
# existing caller (apply_faithfulness_tiering, chatbot.py's in-graph retry
# check, eval/run_eval.py) already gates on — kept as 1-5 specifically so
# nothing downstream needed to change. Only the binary faithful/unfaithful
# split at 0.20 was actually measured; these two boundaries are a reasoned
# margin either side of it, not an independently validated 3-way split —
# revisit if the warn tier turns out to fire far more or less than expected.
_HHEM_REFUSE_BELOW = 0.15
_HHEM_PASS_ABOVE = 0.40


def _get_hhem():
    """Lazily load and cache the local hallucination classifier.

    Committed to the module global only after a successful load — same
    discipline as tools/resume_tool.py's retrieval singletons — so a
    transient failure (HF Hub briefly unreachable) is retried on the next
    call rather than permanently cached as broken.
    """
    global _hhem
    if _hhem is None:
        from transformers import AutoModelForSequenceClassification

        _hhem = AutoModelForSequenceClassification.from_pretrained(
            _HHEM_MODEL_ID, revision=_HHEM_REVISION, trust_remote_code=True
        )
    return _hhem


def _hhem_to_five_scale(raw_score: float) -> int:
    if raw_score < _HHEM_REFUSE_BELOW:
        return 1
    if raw_score < _HHEM_PASS_ABOVE:
        return 3
    return 5


def _judge(answer: str, context: str) -> tuple[int | None, dict[str, int]]:
    """The real check: score plus token usage (always zero now — HHEM is a
    local model, not a priced API call; the return shape is unchanged so
    every existing cost-accounting caller keeps working unmodified).

    None means "nothing to act on" — either there is no context to check
    against, or the model itself failed to load or predict — never a low
    score. A caller driving a retry must treat None as "don't retry", not as
    "score is bad". A failure is logged rather than silently swallowed
    (Phase 4.3): a guardrail that can go quietly inert is worse than no
    guardrail, and this is the one place that ever finds out it just did.
    """
    if not context.strip():
        return None, dict(_NO_USAGE)

    try:
        raw_score = float(_get_hhem().predict([(context, answer)])[0])
        return _hhem_to_five_scale(raw_score), dict(_NO_USAGE)
    except Exception as exc:
        logger.warning("Faithfulness check failed, failing open (no check applied): %s", exc)
        return None, dict(_NO_USAGE)


def score_faithfulness(answer: str, context: str) -> int | None:
    """Score how grounded `answer` is in `context`, 1 (hallucinated) to 5 (fully grounded).

    See `_judge` for what None means. Use `score_faithfulness_with_usage` when
    the caller needs real cost accounting for this call, not just the score.
    """
    score, _ = _judge(answer, context)
    return score


def score_faithfulness_with_usage(answer: str, context: str) -> tuple[int | None, dict[str, int]]:
    """Same as `score_faithfulness`, plus token usage for cost accounting
    (Phase 4.2) — always zero now that this runs a local model rather than a
    priced API call, which honest cost accounting should show as zero, not
    omit. Kept as a tuple so no caller's shape assumptions had to change."""
    return _judge(answer, context)


def apply_faithfulness_tiering(answer: str, score: int | None) -> str:
    """Turn a raw score into the user-facing text: unchanged, warned, or refused.

    Split out from `check_faithfulness` so a caller that already has a score
    (chatbot.py's in-graph check, which computed one for the retry decision)
    can apply the same tiering without paying for a second judge call on the
    same answer.
    """
    if score is None or score >= 4:
        return answer
    if score >= 2:
        return _WARN_PREFIX + answer
    return _LOW_SCORE_RESPONSE


def check_faithfulness(answer: str, context: str) -> str:
    """Score answer faithfulness against retrieved context, then gate or warn.

    Score 4-5: return answer unchanged.
    Score 2-3: prepend a visible warning badge.
    Score 1:   replace with a safe refusal.
    No score (no context, or the judge call failed): return answer unchanged.
    """
    return apply_faithfulness_tiering(answer, score_faithfulness(answer, context))
