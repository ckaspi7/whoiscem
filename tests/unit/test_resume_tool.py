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


def _clear_singletons() -> None:
    resume_tool._store = None
    resume_tool._bm25 = None
    resume_tool._reranker = None
    resume_tool._cache = None
    resume_tool._corpus_chars = 0
    resume_tool._corpus_fp = ""
    resume_tool._qdrant_failed_at = None
    resume_tool._qdrant_last_error = None
    resume_tool._disk_store = None
    resume_tool._disk_store_fp = ""


@pytest.fixture(autouse=True)
def _reset_retrieval_singletons():
    """The module caches its singletons at call time; tests must not leak them."""
    _clear_singletons()
    yield
    _clear_singletons()


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


# ---------------------------------------------------------------------------
# search_resume caches its full result, not just the embedding
# ---------------------------------------------------------------------------


def test_search_resume_does_not_recompute_an_identical_query(fake_redis):
    from retrieval.cache import RetrievalCache
    from retrieval.types import ScoredChunk

    resume_tool._store = MagicMock()
    resume_tool._store.dense_search.return_value = [
        ScoredChunk(chunk_id="1", text="Cem works at TELUS.", score=0.9)
    ]
    resume_tool._bm25 = MagicMock()
    resume_tool._reranker = MagicMock()
    resume_tool._corpus_fp = "test-fingerprint"

    cache = RetrievalCache.__new__(RetrievalCache)
    cache._redis = fake_redis
    cache._backend = "in-process"
    resume_tool._cache = cache

    fake_settings = MagicMock(retrieval_top_n=5, retrieval_strategy="dense")
    with patch("tools.resume_tool.load_settings", return_value=fake_settings):
        first = resume_tool.search_resume("Where does Cem work?")
        second = resume_tool.search_resume("Where does Cem work?")

    assert first == second
    resume_tool._store.dense_search.assert_called_once()


def test_search_resume_treats_a_different_query_as_a_different_cache_entry(fake_redis):
    from retrieval.cache import RetrievalCache
    from retrieval.types import ScoredChunk

    resume_tool._store = MagicMock()
    resume_tool._store.dense_search.side_effect = [
        [ScoredChunk(chunk_id="1", text="answer one", score=0.9)],
        [ScoredChunk(chunk_id="2", text="answer two", score=0.8)],
    ]
    resume_tool._bm25 = MagicMock()
    resume_tool._reranker = MagicMock()
    resume_tool._corpus_fp = "test-fingerprint"

    cache = RetrievalCache.__new__(RetrievalCache)
    cache._redis = fake_redis
    cache._backend = "in-process"
    resume_tool._cache = cache

    fake_settings = MagicMock(retrieval_top_n=5, retrieval_strategy="dense")
    with patch("tools.resume_tool.load_settings", return_value=fake_settings):
        first = resume_tool.search_resume("Where does Cem work?")
        second = resume_tool.search_resume("What did Cem study?")

    assert first != second
    assert resume_tool._store.dense_search.call_count == 2


# ---------------------------------------------------------------------------
# Disk fallback: an unreachable vector store degrades retrieval, it does not
# turn every resume question into "I don't have that information".
# ---------------------------------------------------------------------------

_RESUME_MD = """# Someone

## Education

BASc, Electrical Engineering, University of British Columbia, 2015 - 2021.
Minor in Entrepreneurship. Dean's List 2019-2021.

## Experience

### TELUS Communications - AI/ML Engineer

Built a production RAG agent for annual-report questions and its evaluation suite.
"""


def _disk_fallback_setup(tmp_path):
    resume_file = tmp_path / "resume.md"
    resume_file.write_text(_RESUME_MD, encoding="utf-8")
    settings = MagicMock(resume_path=str(resume_file), retrieval_top_n=5, retrieval_strategy="auto")
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, candidates, top_n=3: candidates
    return settings, reranker


def test_an_unreachable_vector_store_is_answered_from_the_resume_on_disk(tmp_path):
    settings, reranker = _disk_fallback_setup(tmp_path)

    with (
        patch("tools.resume_tool.load_settings", return_value=settings),
        patch("tools.resume_tool._init_retrieval", side_effect=ConnectionError("403 Forbidden")),
        patch("tools.resume_tool.CrossEncoderReranker", return_value=reranker),
    ):
        chunks = resume_tool.search_resume("What did Cem study?")

    assert any("Electrical Engineering" in chunk.text for chunk in chunks)
    assert resume_tool._store is None, "the fallback must never be committed as the store"


def test_the_vector_store_is_skipped_during_the_cooldown_and_retried_after_it(tmp_path):
    settings, reranker = _disk_fallback_setup(tmp_path)
    init = MagicMock(side_effect=ConnectionError("403 Forbidden"))

    with (
        patch("tools.resume_tool.load_settings", return_value=settings),
        patch("tools.resume_tool._init_retrieval", init),
        patch("tools.resume_tool.CrossEncoderReranker", return_value=reranker),
    ):
        resume_tool.search_resume("What did Cem study?")
        resume_tool.search_resume("What did Cem study again?")
        assert init.call_count == 1, "an outage must cost one connection attempt per interval"

        resume_tool._qdrant_failed_at -= resume_tool.QDRANT_RETRY_SECONDS + 1
        resume_tool.search_resume("What did Cem study once more?")
        assert init.call_count == 2, "once the cooldown passes, the store is tried again"


def test_a_recovered_vector_store_is_used_again(tmp_path):
    import time

    from retrieval.types import ScoredChunk

    settings, _ = _disk_fallback_setup(tmp_path)
    settings.retrieval_strategy = "dense"
    resume_tool._store = MagicMock()
    resume_tool._store.dense_search.return_value = [
        ScoredChunk(chunk_id="1", text="from the store", score=0.9)
    ]
    resume_tool._bm25 = MagicMock()
    resume_tool._reranker = MagicMock()
    resume_tool._qdrant_failed_at = time.monotonic() - resume_tool.QDRANT_RETRY_SECONDS - 1
    resume_tool._qdrant_last_error = ConnectionError("403 Forbidden")

    with patch("tools.resume_tool.load_settings", return_value=settings):
        chunks = resume_tool.search_resume("What did Cem study?")

    assert [chunk.text for chunk in chunks] == ["from the store"]
    assert resume_tool._qdrant_failed_at is None
    assert resume_tool._qdrant_last_error is None


def test_the_original_store_error_surfaces_when_the_disk_fallback_cannot_work(tmp_path):
    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    settings = MagicMock(resume_path=str(pdf), retrieval_top_n=5, retrieval_strategy="auto")

    with (
        patch("tools.resume_tool.load_settings", return_value=settings),
        patch("tools.resume_tool._init_retrieval", side_effect=ConnectionError("403 Forbidden")),
        pytest.raises(ConnectionError, match="403 Forbidden"),
    ):
        resume_tool.search_resume("What did Cem study?")


def test_the_typed_wrapper_succeeds_from_disk_when_the_store_is_down(tmp_path):
    settings, reranker = _disk_fallback_setup(tmp_path)

    with (
        patch("tools.resume_tool.load_settings", return_value=settings),
        patch("tools.resume_tool._init_retrieval", side_effect=ConnectionError("403 Forbidden")),
        patch("tools.resume_tool.CrossEncoderReranker", return_value=reranker),
    ):
        result = resume_tool.get_resume_info_result("What did Cem study?")

    assert result.ok is True
    assert "Electrical Engineering" in result.as_context()
