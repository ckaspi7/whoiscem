from __future__ import annotations

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.types import ScoredChunk


def _chunk(cid: str, text: str = "", score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, text=text or cid, score=score)


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_only_dense():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    result = reciprocal_rank_fusion(dense, [])
    ids = [r.chunk_id for r in result]
    assert ids == ["a", "b", "c"]


def test_rrf_only_sparse():
    sparse = [_chunk("x"), _chunk("y")]
    result = reciprocal_rank_fusion([], sparse)
    ids = [r.chunk_id for r in result]
    assert ids == ["x", "y"]


def test_rrf_deduplicates_overlap():
    dense = [_chunk("a"), _chunk("b")]
    sparse = [_chunk("a"), _chunk("c")]
    result = reciprocal_rank_fusion(dense, sparse)
    ids = [r.chunk_id for r in result]
    assert ids.count("a") == 1


def test_rrf_top_chunk_in_both_lists_scores_highest():
    dense = [_chunk("star"), _chunk("b")]
    sparse = [_chunk("star"), _chunk("c")]
    result = reciprocal_rank_fusion(dense, sparse)
    assert result[0].chunk_id == "star"


def test_rrf_score_is_sum_of_reciprocal_ranks():
    dense = [_chunk("a")]
    sparse = [_chunk("a")]
    result = reciprocal_rank_fusion(dense, sparse, k=60)
    expected = 1 / (60 + 1) + 1 / (60 + 1)
    assert abs(result[0].score - expected) < 1e-9


def test_rrf_higher_k_reduces_score_difference():
    dense = [_chunk("a"), _chunk("b")]
    sparse = []

    result_k10 = reciprocal_rank_fusion(dense, sparse, k=10)
    result_k100 = reciprocal_rank_fusion(dense, sparse, k=100)

    diff_k10 = result_k10[0].score - result_k10[1].score
    diff_k100 = result_k100[0].score - result_k100[1].score
    assert diff_k10 > diff_k100


def test_rrf_preserves_text():
    dense = [_chunk("id1", text="TELUS experience")]
    result = reciprocal_rank_fusion(dense, [])
    assert result[0].text == "TELUS experience"
