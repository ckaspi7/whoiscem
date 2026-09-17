from __future__ import annotations

import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import Settings, load_settings
from retrieval.backends import create_qdrant_client
from retrieval.types import ScoredChunk

COLLECTION_NAME = "resume_chunks"
VECTOR_SIZE = 1536  # text-embedding-ada-002 / text-embedding-3-small


class QdrantVectorStore:
    """Dense retrieval over resume chunks, backed by any Qdrant deployment mode.

    The client is injected rather than constructed from hardcoded coordinates:
    pass one explicitly, or let ``config.Settings`` pick embedded / server /
    cloud from the environment.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        settings: Settings | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._settings = settings or load_settings()
        self._client = client or create_qdrant_client(self._settings)
        self._collection = collection_name
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    @property
    def collection_name(self) -> str:
        return self._collection

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def build_from_pdf(self, pdf_path: str) -> None:
        text = self._extract_text(pdf_path)
        chunks = self._semantic_chunk(text)
        self._upsert(chunks)

    def dense_search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        query_vec = self._embeddings.embed_query(query)
        results = self._client.search(
            collection_name=self._collection,
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

    def get_all_chunks(self) -> list[ScoredChunk]:
        records, _ = self._client.scroll(
            collection_name=self._collection,
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

    def close(self) -> None:
        """Release the client. Embedded mode holds an exclusive lock on its directory."""
        self._client.close()

    def _extract_text(self, pdf_path: str) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)

    def _semantic_chunk(self, text: str) -> list[str]:
        splitter = SemanticChunker(
            OpenAIEmbeddings(model="text-embedding-3-small"),
            breakpoint_threshold_type="percentile",
        )
        return splitter.split_text(text)

    def _upsert(self, chunks: list[str]) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

        vectors = self._embeddings.embed_documents(chunks)
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"text": chunk, "chunk_index": i, "section": ""},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
