"""Eval trends dashboard (Phase 5) — a second Streamlit page, auto-discovered
by Streamlit's multi-page convention (a `pages/` directory next to the
entry-point script). Reads only committed `eval/results/*.json` files: no
Qdrant, no Redis, no OPENAI_API_KEY. That is deliberate, not incidental — it
means this page cannot be affected by which vector-store or session-memory
backend `chatbot.py` is pointed at, and needs no changes at all when Phase
4.6 moves the deployed app to Qdrant Cloud.

"Mainline" vs "comparison" is derived from each run's own recorded
agent_mode/chat_model against config.py's actual current defaults, not a
hardcoded file list — a run matching the shipping configuration (or
predating those fields, for anything before Phase 3.1) is mainline;
anything else is a deliberate one-axis comparison, shown separately so it is
never blended into a trend line as if it were a regression. This is what
keeps the dashboard correct as new eval runs are committed later without
editing this file, including if the defaults themselves ever change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DEFAULT_AGENT_MODE, DEFAULT_CHAT_MODEL

RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval" / "results"

st.set_page_config(page_title="Eval Dashboard", page_icon=":material/analytics:", layout="wide")

# Human-readable labels and one-line definitions for the raw metric keys these
# result files store. This page's audience isn't only the person who wrote
# eval/run_eval.py, so "context_precision" gets spelled out rather than assumed
# — see the "What these metrics mean" expander below, where METRIC_GLOSSARY is
# rendered as a key-value table.
METRIC_LABELS: dict[str, str] = {
    "routing_accuracy": "Routing accuracy",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_recall": "Context recall",
    "context_precision": "Context precision",
}
METRIC_GLOSSARY: dict[str, str] = {
    "Routing accuracy": (
        "Share of questions sent to the correct tool — resume, personal info, Spotify, "
        "LinkedIn, or plain conversation."
    ),
    "Faithfulness": (
        "Does the answer stick to what was actually retrieved, or invent details? "
        "An LLM judge scores this 0-1 (RAGAS)."
    ),
    "Answer relevancy": (
        "Does the answer actually address the question asked? Scored 0-1 by an LLM judge (RAGAS)."
    ),
    "Context recall": (
        "Of the facts needed to answer, how many did retrieval actually surface? Scored 0-1 (RAGAS)."
    ),
    "Context precision": ("Of what retrieval surfaced, how much was actually relevant? Scored 0-1 (RAGAS)."),
}


@st.cache_data
def load_results() -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["_file"] = path.name
        runs.append(data)
    return runs


def is_ablation(run: dict) -> bool:
    """Ablation studies measure retrieval strategies directly, not through
    the graph — no routing, no per-question rows, a different shape entirely."""
    return "routing" not in run


def is_mainline(run: dict) -> bool:
    agent_mode = run.get("agent_mode")
    chat_model = run.get("chat_model")
    agent_ok = agent_mode is None or agent_mode == DEFAULT_AGENT_MODE
    model_ok = chat_model is None or chat_model == DEFAULT_CHAT_MODEL
    return agent_ok and model_ok


st.title("Eval dashboard", icon=":material/analytics:")
st.caption(
    "Every number below is read directly from a committed file in `eval/results/` — "
    "nothing here is recomputed, estimated, or remembered. See "
    "[CHANGELOG.md](https://github.com/ckaspi7/whoiscem/blob/main/CHANGELOG.md) "
    "for the narrative behind each point."
)
with st.expander("What these metrics mean"):
    st.table(METRIC_GLOSSARY, border="horizontal")

runs = load_results()
if not runs:
    st.warning(f"No result files found in `{RESULTS_DIR}`.")
    st.stop()

question_runs = [r for r in runs if not is_ablation(r)]
ablation_runs = [r for r in runs if is_ablation(r)]

# ---------------------------------------------------------------------------
# Mainline trend
# ---------------------------------------------------------------------------
st.header("Mainline progression")
st.caption(
    f"Runs matching the current shipping defaults (`AGENT_MODE={DEFAULT_AGENT_MODE}`, "
    f"`CHAT_MODEL={DEFAULT_CHAT_MODEL}`), or predating those settings, in commit order. "
    "Deliberately excludes the architecture/model comparisons below — blending them in would "
    "make an intentional trade-off (e.g. tool_calling's lower routing accuracy) look like a "
    "regression in the same trend a real one would show up in."
)

mainline = sorted(
    (r for r in question_runs if is_mainline(r) and r.get("scores")),
    key=lambda r: r.get("run_at", ""),
)

if mainline:
    trend_rows = [
        {
            "run": r["_file"],
            "run_at": (r.get("run_at") or "")[:10],
            "routing_accuracy": (r.get("routing") or {}).get("accuracy"),
            **(r.get("scores") or {}),
        }
        for r in mainline
    ]
    trend_df = pd.DataFrame(trend_rows).set_index("run")
    candidate_columns = [
        "routing_accuracy",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]
    chart_columns = [c for c in candidate_columns if c in trend_df.columns]
    display_df = trend_df.rename(columns=METRIC_LABELS)
    chart_labels = [METRIC_LABELS[c] for c in chart_columns]
    st.line_chart(display_df[chart_labels], x_label="Evaluation run (commit order)", y_label="Score")
    st.dataframe(display_df)
else:
    st.info("No mainline runs with RAGAS scores committed yet.")

# ---------------------------------------------------------------------------
# Per-category routing — latest mainline run
# ---------------------------------------------------------------------------
if mainline:
    latest = mainline[-1]
    st.header(f"Per-category routing — latest mainline run (`{latest['_file']}`)", icon=":material/route:")
    st.caption("Share of questions correctly routed to the right tool, broken down by question category.")
    by_category = (latest.get("routing") or {}).get("by_category") or {}
    if by_category:
        cat_df = pd.DataFrame(
            [
                {"category": k, "accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
                for k, v in sorted(by_category.items())
            ]
        ).set_index("category")
        st.bar_chart(cat_df["accuracy"], x_label="Category", y_label="Accuracy")
        st.dataframe(cat_df.rename(columns={"accuracy": "Accuracy", "correct": "Correct", "total": "Total"}))

# ---------------------------------------------------------------------------
# Deliberate comparisons — never blended into the trend above
# ---------------------------------------------------------------------------
st.header("Deliberate comparisons", icon=":material/compare_arrows:")
st.caption(
    "Each of these changes exactly one axis against a mainline baseline from the same day — "
    "paired, not chronological, the same way the README presents them. **agent_mode** swaps the "
    "LangGraph architecture (a classifier that calls one fixed tool vs. a tool-calling agent); "
    "**chat_model** swaps the underlying LLM. Both default to the shipping configuration "
    f"(`{DEFAULT_AGENT_MODE}` / `{DEFAULT_CHAT_MODEL}`) unless named in a comparison below."
)

comparisons = sorted((r for r in question_runs if not is_mainline(r)), key=lambda r: r.get("run_at", ""))

if not comparisons:
    st.caption("None committed yet.")

for run in comparisons:
    axis = []
    if run.get("agent_mode") and run["agent_mode"] != DEFAULT_AGENT_MODE:
        axis.append(f"agent_mode={run['agent_mode']}")
    if run.get("chat_model") and run["chat_model"] != DEFAULT_CHAT_MODEL:
        axis.append(f"chat_model={run['chat_model']}")
    label = ", ".join(axis) or "non-mainline configuration"

    # No icon= on this expander (unlike the rest of this page's headers):
    # st.expander's icon parameter isn't picked up by AppTest's .expander
    # introspection on this pinned Streamlit version, and
    # test_dashboard_separates_comparisons_from_the_mainline_trend reads
    # this exact label through app.expander.
    with st.expander(f"{run['_file']} — {label}"):
        scores = run.get("scores") or {}
        cols = st.columns(max(1, 1 + len(scores)))
        accuracy = (run.get("routing") or {}).get("accuracy")
        cols[0].metric(
            "Routing accuracy",
            f"{accuracy:.1%}" if accuracy is not None else "n/a",
            help=METRIC_GLOSSARY["Routing accuracy"],
        )
        for i, (metric, value) in enumerate(scores.items(), start=1):
            metric_label = METRIC_LABELS.get(metric, metric)
            cols[i].metric(
                metric_label,
                f"{value:.3f}" if value is not None else "n/a",
                help=METRIC_GLOSSARY.get(metric_label),
            )
        if not scores:
            st.caption("No RAGAS scores in this run (a `--no-ragas` pass).")

        trajectory = run.get("trajectory") or {}
        if trajectory.get("measured"):
            st.caption(
                f"Trajectory: {trajectory['multi_intent_complete']}/{trajectory['multi_intent_total']} "
                f"multi-intent complete, {trajectory['avg_tool_calls']} avg tool calls/turn."
            )

        self_correction = run.get("self_correction") or {}
        if self_correction.get("retried"):
            st.caption(f"Self-correction retried {self_correction['retried']} question(s).")

# ---------------------------------------------------------------------------
# Retrieval ablation — the cost/latency Pareto
# ---------------------------------------------------------------------------
if ablation_runs:
    st.header("Retrieval ablation", icon=":material/tune:")
    st.caption(
        "Which parts of the hybrid pipeline earn their place, measured with routing excluded "
        "so a variant is never penalised for a question the router misdirected. See the README "
        "for the full finding: plain dense search wins outright on this corpus."
    )
    ablation = ablation_runs[-1]
    variants_df = pd.DataFrame(ablation["variants"]).set_index("variant")
    st.caption(
        "Each point is one retrieval strategy; bigger points returned more context text per query "
        "(mean characters retrieved)."
    )
    st.scatter_chart(
        variants_df,
        x="mean_latency_ms",
        y="recall_at_k",
        size="mean_context_chars",
        x_label="Mean latency (ms)",
        y_label="Recall@k",
    )
    st.dataframe(
        variants_df.rename(
            columns={
                "recall_at_k": "Recall@k",
                "mrr": "MRR",
                "mean_context_chars": "Mean context (chars)",
                "mean_latency_ms": "Mean latency (ms)",
            }
        )
    )
