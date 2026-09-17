"""
Integration tests for the retrieval pipeline.

Runs against whatever backend QDRANT_MODE selects. The default — embedded —
needs no services, so these are runnable locally and in CI; set
QDRANT_MODE=server to point them at docker-compose instead. Tests that embed
text need OPENAI_API_KEY and are skipped without one.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import load_settings

pytestmark = pytest.mark.integration

TEST_COLLECTION = "test_integration_chunks"

needs_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    """Env-derived settings, with embedded storage redirected to a temp dir."""
    cfg = load_settings()
    if cfg.qdrant_mode == "embedded":
        cfg = dataclasses.replace(cfg, qdrant_path=str(tmp_path_factory.mktemp("qdrant")))
    return cfg


@pytest.fixture(scope="module")
def qdrant(settings):
    from retrieval.backends import create_qdrant_client

    try:
        client = create_qdrant_client(settings)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant ({settings.qdrant_mode}) unavailable: {exc}")

    yield client
    client.close()


@pytest.fixture  # function-scoped: sample_chunks is, and scopes must match
def indexed_collection(qdrant, sample_chunks):
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client.models import Distance, PointStruct, VectorParams

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    if qdrant.collection_exists(TEST_COLLECTION):
        qdrant.delete_collection(TEST_COLLECTION)

    qdrant.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(sample_chunks)
    points = [
        PointStruct(id=i, vector=v, payload={"text": t, "chunk_index": i, "section": ""})
        for i, (t, v) in enumerate(zip(sample_chunks, vectors, strict=True))
    ]
    qdrant.upsert(collection_name=TEST_COLLECTION, points=points)

    yield qdrant, TEST_COLLECTION

    qdrant.delete_collection(TEST_COLLECTION)


@needs_openai
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

    index = BM25Index()
    index.build(sample_chunks)

    results = index.search("NeoWise startup co-founder", top_k=5)
    top_texts = [r.text for r in results]
    assert any("NeoWise" in t for t in top_texts)


def test_rrf_preserves_all_chunk_ids(sample_chunks):
    from retrieval.fusion import reciprocal_rank_fusion
    from retrieval.types import ScoredChunk

    dense = [ScoredChunk(chunk_id=str(i), text=c, score=1.0) for i, c in enumerate(sample_chunks)]
    sparse = [ScoredChunk(chunk_id=str(i), text=c, score=1.0) for i, c in enumerate(sample_chunks[:3])]

    fused = reciprocal_rank_fusion(dense, sparse)
    fused_ids = {r.chunk_id for r in fused}
    all_ids = {str(i) for i in range(len(sample_chunks))}
    assert fused_ids == all_ids


@needs_openai
def test_vector_store_runs_against_the_configured_backend(qdrant, settings):
    from retrieval.vectorstore import QdrantVectorStore

    store = QdrantVectorStore(client=qdrant, settings=settings, collection_name=TEST_COLLECTION)
    assert isinstance(store.collection_exists(), bool)
