from __future__ import annotations

import os
from typing import List

import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from retrieval.types import ScoredChunk

COLLECTION_NAME = "resume_chunks"
VECTOR_SIZE = 1536  # text-embedding-ada-002 / text-embedding-3-small


class QdrantVectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def collection_exists(self) -> bool:
        return self._client.collection_exists(COLLECTION_NAME)

    def build_from_pdf(self, pdf_path: str) -> None:
        text = self._extract_text(pdf_path)
        chunks = self._semantic_chunk(text)
        self._upsert(chunks)

    def dense_search(self, query: str, top_k: int = 20) -> List[ScoredChunk]:
        query_vec = self._embeddings.embed_query(query)
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=str(r.id),
                text=r.payload.get("text", ""),
                score=r.score,
                section=r.payload.get("section", ""),
            )
            for r in results
        ]

    def get_all_chunks(self) -> List[ScoredChunk]:
        records, _ = self._client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=str(r.id),
                text=r.payload.get("text", ""),
                score=0.0,
                section=r.payload.get("section", ""),
            )
            for r in records
        ]

    def _extract_text(self, pdf_path: str) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)

    def _semantic_chunk(self, text: str) -> List[str]:
        splitter = SemanticChunker(
            OpenAIEmbeddings(model="text-embedding-3-small"),
            breakpoint_threshold_type="percentile",
        )
        return splitter.split_text(text)

    def _upsert(self, chunks: List[str]) -> None:
        if not self._client.collection_exists(COLLECTION_NAME):
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

        vectors = self._embeddings.embed_documents(chunks)
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"text": chunk, "chunk_index": i, "section": ""},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)
