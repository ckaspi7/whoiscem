"""Regression tests for the retrieval singleton poisoning bug.

``_init_retrieval()`` used to commit ``_store`` / ``_bm25`` to the module-level
globals before the network calls that build them had succeeded. A transient
failure — a Qdrant container not yet accepting connections, which is exactly
what a fresh CI service container is on its first request — left the module
permanently convinced retrieval was already set up. Every later call in the
same process reused that broken state instead of retrying: not an error,
just silently wrong for the rest of the process's life.

This is also, concretely, the best explanation found for a real CI failure: a
scheduled full evaluation run passed "Run full evaluation" but failed
"Compare against the committed baseline" — the run completed and produced
numbers, but far enough below tolerance to suggest every resume-routed
question failed for the same reason, all at once, which is what a poisoned
singleton looks like from the outside.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import tools.resume_tool as resume_tool


@pytest.fixture(autouse=True)
def _reset_retrieval_singletons():
    """The module caches its singletons at call time; tests must not leak them."""
    resume_tool._store = None
    resume_tool._bm25 = None
    resume_tool._reranker = None
    resume_tool._corpus_chars = 0
    yield
    resume_tool._store = None
    resume_tool._bm25 = None
    resume_tool._reranker = None
    resume_tool._corpus_chars = 0


def test_a_failed_store_setup_is_retried_not_cached(tmp_path):
    """The bug's first flavour: _store used to be assigned before needs_rebuild ran."""
    resume_file = tmp_path / "resume.md"
    resume_file.write_text("# Someone\n\n## Summary\n\nAn engineer.\n", encoding="utf-8")
    fake_settings = MagicMock(resume_path=str(resume_file))

    broken_store = MagicMock()
    broken_store.needs_rebuild.side_effect = ConnectionError("qdrant not ready")

    with (
        patch("tools.resume_tool.load_settings", return_value=fake_settings),
        patch("tools.resume_tool.QdrantVectorStore", return_value=broken_store),
        pytest.raises(ConnectionError),
    ):
        resume_tool._init_retrieval()

    # The bug: this used to be non-None here, so a second call would never retry.
    assert resume_tool._store is None, "a failed setup must not be cached"

    # A second, now-healthy attempt must actually retry, not reuse stale state.
    healthy_store = MagicMock()
    healthy_store.needs_rebuild.return_value = False
    healthy_store.get_all_chunks.return_value = []

    with (
        patch("tools.resume_tool.load_settings", return_value=fake_settings),
        patch("tools.resume_tool.QdrantVectorStore", return_value=healthy_store),
        patch("tools.resume_tool.CrossEncoderReranker", return_value=MagicMock()),
    ):
        store, _bm25, _reranker = resume_tool._init_retrieval()

    assert store is healthy_store
    healthy_store.needs_rebuild.assert_called_once()


def test_a_failed_bm25_build_is_retried_not_cached():
    """The bug's second flavour: _bm25 = BM25Index() ran before the network call
    that fills it, so a failure there left _bm25 non-None but never built —
    every later query silently returned no results, with no error at all."""
    healthy_store = MagicMock()
    healthy_store.needs_rebuild.return_value = False
    healthy_store.get_all_chunks.side_effect = [ConnectionError("timed out"), []]
    resume_tool._store = healthy_store  # already "built" from a prior call

    with pytest.raises(ConnectionError):
        resume_tool._init_retrieval()

    assert resume_tool._bm25 is None, "a failed bm25 build must not be cached"

    with patch("tools.resume_tool.CrossEncoderReranker", return_value=MagicMock()):
        _store, bm25, _reranker = resume_tool._init_retrieval()

    assert bm25 is not None
    assert healthy_store.get_all_chunks.call_count == 2, "the retry must hit the store again"


def test_a_successful_setup_is_reused_not_rebuilt():
    """The other half of the contract: success should still be cached."""
    healthy_store = MagicMock()
    healthy_store.needs_rebuild.return_value = False
    healthy_store.get_all_chunks.return_value = []

    with (
        patch("tools.resume_tool.load_settings", return_value=MagicMock(resume_path=__file__)),
        patch("tools.resume_tool.QdrantVectorStore", return_value=healthy_store),
        patch("tools.resume_tool.CrossEncoderReranker", return_value=MagicMock()),
    ):
        resume_tool._init_retrieval()
        resume_tool._init_retrieval()

    healthy_store.needs_rebuild.assert_called_once()
    healthy_store.get_all_chunks.assert_called_once()


def test_get_resume_info_result_never_puts_the_error_in_content():
    with patch("tools.resume_tool.search_resume", side_effect=RuntimeError("qdrant down")):
        result = resume_tool.get_resume_info_result("anything")

    assert result.ok is False
    assert result.as_context() == ""
    assert "qdrant down" in result.error
