from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.types import ScoredChunk


def _chunk(cid: str, text: str) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, text=text, score=0.5)


@pytest.fixture
def mock_cross_encoder():
    with patch("retrieval.reranker._CrossEncoder") as MockCE:
        instance = MagicMock()
        MockCE.return_value = instance
        yield instance


def test_reranker_returns_top_n(mock_cross_encoder):
    from retrieval.reranker import CrossEncoderReranker

    mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    candidates = [_chunk(str(i), f"chunk {i}") for i in range(10)]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("test query", candidates, top_n=3)

    assert len(result) == 3


def test_reranker_ordering_highest_score_first(mock_cross_encoder):
    from retrieval.reranker import CrossEncoderReranker

    mock_cross_encoder.predict.return_value = [0.2, 0.9, 0.5]
    candidates = [_chunk("a", "low"), _chunk("b", "high"), _chunk("c", "mid")]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", candidates, top_n=3)

    assert result[0].chunk_id == "b"
    assert result[1].chunk_id == "c"
    assert result[2].chunk_id == "a"


def test_reranker_single_candidate(mock_cross_encoder):
    from retrieval.reranker import CrossEncoderReranker

    mock_cross_encoder.predict.return_value = [0.75]
    candidates = [_chunk("only", "sole chunk")]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", candidates, top_n=3)

    assert len(result) == 1
    assert result[0].chunk_id == "only"


def test_reranker_empty_input(mock_cross_encoder):
    from retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", [], top_n=3)

    assert result == []
    mock_cross_encoder.predict.assert_not_called()


def test_reranker_score_stored_in_result(mock_cross_encoder):
    from retrieval.reranker import CrossEncoderReranker

    mock_cross_encoder.predict.return_value = [0.88]
    candidates = [_chunk("x", "text")]

    reranker = CrossEncoderReranker()
    result = reranker.rerank("q", candidates, top_n=1)

    assert abs(result[0].score - 0.88) < 1e-6
