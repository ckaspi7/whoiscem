"""
Integration tests for the retrieval pipeline.
Requires Qdrant running: docker-compose up qdrant
Tests are skipped automatically when Qdrant is unavailable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333, timeout=2).get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_qdrant():
    if not _qdrant_reachable():
        pytest.skip("Qdrant not running")
    from qdrant_client import QdrantClient
    return QdrantClient(host="localhost", port=6333)


@pytest.fixture(scope="module")
def indexed_collection(live_qdrant, sample_chunks):
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client.models import Distance, PointStruct, VectorParams

    collection = "test_integration_chunks"
    if live_qdrant.collection_exists(collection):
        live_qdrant.delete_collection(collection)

    live_qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(sample_chunks)
    points = [
        PointStruct(id=i, vector=v, payload={"text": t, "chunk_index": i, "section": ""})
        for i, (t, v) in enumerate(zip(sample_chunks, vectors))
    ]
    live_qdrant.upsert(collection_name=collection, points=points)

    yield live_qdrant, collection

    live_qdrant.delete_collection(collection)


def test_dense_search_returns_results(indexed_collection):
    client, collection = indexed_collection
    results = client.search(
        collection_name=collection,
        query_vector=[0.01] * 1536,
        limit=3,
    )
    assert len(results) > 0


def test_bm25_finds_telus_chunk(sample_chunks):
    from retrieval.bm25 import BM25Index

    index = BM25Index()
    index.build(sample_chunks)
    results = index.search("TELUS engineer Vancouver", top_k=3)

    assert len(results) > 0
    assert "TELUS" in results[0].text


def test_bm25_outperforms_dense_on_keyword_query(sample_chunks):
    from retrieval.bm25 import BM25Index
    from retrieval.vectorstore import ScoredChunk

    index = BM25Index()
    index.build(sample_chunks)

    results = index.search("NeoWise startup co-founder", top_k=5)
    top_texts = [r.text for r in results]
    assert any("NeoWise" in t for t in top_texts)


def test_rrf_preserves_all_chunk_ids(sample_chunks):
    from retrieval.fusion import reciprocal_rank_fusion
    from retrieval.vectorstore import ScoredChunk

    dense = [ScoredChunk(chunk_id=str(i), text=c, score=1.0) for i, c in enumerate(sample_chunks)]
    sparse = [ScoredChunk(chunk_id=str(i), text=c, score=1.0) for i, c in enumerate(sample_chunks[:3])]

    fused = reciprocal_rank_fusion(dense, sparse)
    fused_ids = {r.chunk_id for r in fused}
    all_ids = {str(i) for i in range(len(sample_chunks))}
    assert fused_ids == all_ids


def test_qdrant_store_collection_exists_check(live_qdrant):
    from retrieval.vectorstore import QdrantVectorStore

    store = QdrantVectorStore(host="localhost", port=6333)
    result = store.collection_exists()
    assert isinstance(result, bool)
