"""pages/1_Eval_Dashboard.py (Phase 5). Reads only committed eval/results/
files — no Qdrant, no Redis, no OPENAI_API_KEY, so no needs_openai guard and
no mocking: this is exactly what the page does in production.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[2] / "pages" / "1_📊_Eval_Dashboard.py")


def test_dashboard_boots_without_error_against_the_real_committed_results():
    app = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not app.exception, [e.message for e in app.exception]


def test_dashboard_shows_the_mainline_trend_and_at_least_one_chart():
    app = AppTest.from_file(PAGE, default_timeout=30).run()

    headers = [h.value for h in app.header]
    assert any("Mainline progression" in h for h in headers)
    assert len(app.dataframe) >= 1, "expected at least the mainline trend table"


def test_dashboard_separates_comparisons_from_the_mainline_trend():
    app = AppTest.from_file(PAGE, default_timeout=30).run()

    headers = [h.value for h in app.header]
    assert any("Deliberate comparisons" in h for h in headers)
    # v9/v11/v13 are real committed comparison runs — the dashboard must
    # render each as its own expander, not fold them into the trend chart.
    expander_labels = [e.label for e in app.expander]
    assert any("v9-tool-calling" in label for label in expander_labels)
    assert any("v11-gpt6-luna-comparison" in label for label in expander_labels)


def test_dashboard_shows_the_retrieval_ablation():
    app = AppTest.from_file(PAGE, default_timeout=30).run()

    headers = [h.value for h in app.header]
    assert any("Retrieval ablation" in h for h in headers)
