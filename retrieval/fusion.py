from __future__ import annotations

from typing import List

from retrieval.types import ScoredChunk


def reciprocal_rank_fusion(
    dense_results: List[ScoredChunk],
    sparse_results: List[ScoredChunk],
    k: int = 60,
) -> List[ScoredChunk]:
    """Combine dense and sparse ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i), summed over each list the chunk appears in.
    Higher k dampens the impact of rank differences between lists.
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    sections: dict[str, str] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        texts[chunk.chunk_id] = chunk.text
        sections[chunk.chunk_id] = chunk.section

    for rank, chunk in enumerate(sparse_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk.chunk_id not in texts:
            texts[chunk.chunk_id] = chunk.text
            sections[chunk.chunk_id] = chunk.section

    return [
        ScoredChunk(chunk_id=cid, text=texts[cid], score=score, section=sections.get(cid, ""))
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
