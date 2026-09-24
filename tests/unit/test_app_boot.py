"""The app must import and boot on the Python version CI runs.

Nothing else in the suite imports chatbot.py, which is how a syntax error that
only bites on Python 3.11 survived in the module for a full release.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

# Streamlit resolves a relative AppTest path against the calling file, not the
# working directory, so point at the app explicitly.
APP = str(Path(__file__).resolve().parents[2] / "chatbot.py")

needs_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")


def test_chatbot_module_imports():
    import chatbot

    assert callable(chatbot.create_assistant)
    assert callable(chatbot.main)


@needs_openai
def test_app_boots_with_no_services_running():
    """Boots against the default backends: embedded Qdrant, in-process memory.

    No Qdrant, no Redis, no Docker — if this needs a service to render, the
    zero-infrastructure quickstart in the README is not true.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=120).run()

    assert not app.exception, [e.message for e in app.exception]
    assert len(app.chat_input) == 1

    captions = [c.value for c in app.sidebar.caption]
    assert any("Vector store" in c for c in captions), captions
    assert any("Session memory" in c for c in captions), captions


@needs_openai
def test_a_failed_tool_call_never_reaches_the_model_as_context():
    """The actual graph wiring, not just the tool functions in isolation.

    classify_query is mocked to force the resume route deterministically and
    for free; search_resume is mocked to fail, the way an unreachable Qdrant
    would. The only real network call left is the final generation.
    """
    import chatbot

    with (
        patch("chatbot.classify_query", return_value="resume"),
        patch("chatbot.search_resume", side_effect=ConnectionError("qdrant unreachable")),
    ):
        graph = chatbot.create_assistant()
        state = graph.invoke(
            {
                "messages": [HumanMessage(content="Where does Cem work?")],
                "next_step": "",
                "search_query": "",
                "route": "",
                "tool_result": "",
                "context_used": "",
                "context_chunks": [],
                "tool_error": "",
                "node_latencies": {},
            }
        )

    assert state["route"] == "resume"
    assert "qdrant unreachable" in state["tool_error"]
    # The two fields that reach the prompt and the faithfulness judge: an
    # error must never appear in either, regardless of how the tool failed.
    assert state["tool_result"] == ""
    assert state["context_used"] == ""

    # generate_response now calls .invoke(), so the final message is already a
    # plain string, not a generator to drain.
    answer = state["messages"][-1].content
    assert "qdrant" not in answer.lower()
    assert "connectionerror" not in answer.lower()
