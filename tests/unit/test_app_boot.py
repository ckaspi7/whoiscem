"""The app must import and boot on the Python version CI runs.

Nothing else in the suite imports chatbot.py, which is how a syntax error that
only bites on Python 3.11 survived in the module for a full release.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
