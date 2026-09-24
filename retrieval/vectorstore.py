from __future__ import annotations

import hashlib
from pathlib import Path

import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import Settings, load_settings
from retrieval.backends import create_qdrant_client
from retrieval.cache import RetrievalCache
from retrieval.chunking import chunk_markdown
from retrieval.types import ScoredChunk, TextChunk

COLLECTION_NAME = "resume_chunks"
VECTOR_SIZE = 1536  # text-embedding-ada-002 / text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"

# Bumped when the chunking strategy changes, so an index built by an older
# version is rebuilt rather than silently reused. A stale index is the quietest
# failure in a RAG system: everything works, the answers are just from the
# previous corpus.
CHUNKER_VERSION = "md-sections-v1"


def fingerprint(path: str) -> str:
    """Identifies the indexed inputs: the source bytes plus the chunker used."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return f"{CHUNKER_VERSION}:{digest}"


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
        cache: RetrievalCache | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._client = client or create_qdrant_client(self._settings)
        self._collection = collection_name
        self._embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        # None means uncached — every existing caller that builds a store
        # without passing one keeps calling the API directly, unchanged.
        self._cache = cache

    @property
    def collection_name(self) -> str:
        return self._collection

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def build_from_file(self, path: str) -> None:
        """Index the resume, replacing whatever is there. Markdown or PDF."""
        text = self._extract_text(path)
        chunks = self._chunk(path, text)
        self._upsert(chunks, fingerprint(path))

    def needs_rebuild(self, path: str) -> bool:
        """True when there is no index, or it was built from different inputs."""
        if not self.collection_exists():
            return True
        return self.index_fingerprint() != fingerprint(path)

    def index_fingerprint(self) -> str | None:
        """The fingerprint recorded on the indexed points, if any."""
        try:
            records, _ = self._client.scroll(collection_name=self._collection, limit=1, with_payload=True)
        except Exception:
            return None
        if not records:
            return None
        return records[0].payload.get("fingerprint")

    def dense_search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        query_vec = self._embed_query(query)
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
        """Every chunk in the collection.

        Paginated. The previous hard cap of 500 with no paging meant that past
        500 chunks the sparse index silently held a subset of what the dense
        index held, and fusion joined two different corpora.
        """
        chunks: list[ScoredChunk] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            chunks.extend(
                ScoredChunk(
                    chunk_id=str(r.id),
                    text=r.payload.get("text", ""),
                    score=0.0,
                    section=r.payload.get("section", ""),
                )
                for r in records
            )
            if offset is None:
                break
        return chunks

    def close(self) -> None:
        """Release the client. Embedded mode holds an exclusive lock on its directory."""
        self._client.close()

    def _embed_query(self, text: str) -> list[float]:
        """Embed one query, through the cache when one is configured.

        Only the query path is cached, not indexing: embed_documents runs once
        per rebuild already, while a query can repeat many times — the same
        question asked twice, or, concretely, eval/ablate_retrieval.py
        re-embedding one query for every strategy variant that uses it.
        """
        if self._cache:
            cached = self._cache.get_embedding(EMBEDDING_MODEL, text)
            if cached is not None:
                return cached

        vector = self._embeddings.embed_query(text)

        if self._cache:
            self._cache.set_embedding(EMBEDDING_MODEL, text, vector)
        return vector

    def _extract_text(self, path: str) -> str:
        """Read the resume as text.

        Markdown is read as-is; a PDF goes through pdfplumber, which flattens
        layout and loses the heading structure. Chunking is identical either
        way for now, so the source format is the only thing that differs.
        """
        if path.lower().endswith((".md", ".markdown", ".txt")):
            return Path(path).read_text(encoding="utf-8")
        with pdfplumber.open(path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)

    def _chunk(self, path: str, text: str) -> list[TextChunk]:
        """Structural chunking for Markdown; semantic chunking for PDFs.

        A PDF has no headings left after extraction, so there is nothing
        structural to use and the embedding-distance splitter is the fallback.
        """
        if path.lower().endswith((".md", ".markdown")):
            return chunk_markdown(text)
        splitter = SemanticChunker(
            OpenAIEmbeddings(model=EMBEDDING_MODEL),
            breakpoint_threshold_type="percentile",
        )
        return [TextChunk(text=piece) for piece in splitter.split_text(text)]

    def _upsert(self, chunks: list[TextChunk], corpus_fingerprint: str) -> None:
        # Recreated rather than upserted: a new chunking can produce fewer
        # chunks, and the leftovers would stay searchable forever.
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        vectors = self._embeddings.embed_documents([c.text for c in chunks])
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={
                    "text": chunk.text,
                    "chunk_index": i,
                    "section": chunk.section,
                    "fingerprint": corpus_fingerprint,
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
