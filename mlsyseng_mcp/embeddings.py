"""Embedding generation and RAG retrieval via sentence-transformers + ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"


class EmbeddingStore:
    """Manages vector embeddings for ML Principles content."""

    def __init__(self, chroma_path: str = CHROMA_DB_PATH, model_name: str = EMBEDDING_MODEL):
        self._chroma_path = chroma_path
        self._model_name = model_name
        self._client = None
        self._collection = None
        self._embed_fn = None

    def _ensure_ready(self):
        if self._collection is not None:
            return

        Path(self._chroma_path).mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self._chroma_path,
                    anonymized_telemetry=False,
                )
            )
        except TypeError:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._chroma_path)

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        try:
            from sentence_transformers import SentenceTransformer

            self._embed_fn = SentenceTransformer(self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed – embeddings will use ChromaDB defaults"
            )

    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self._embed_fn is None:
            return None
        vectors = self._embed_fn.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def index_chapter(
        self,
        chapter_slug: str,
        chapter_title: str,
        markdown: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ):
        """Split *markdown* into chunks and index them."""
        self._ensure_ready()
        chunks = self._chunk(markdown, chunk_size, overlap)
        if not chunks:
            return

        ids = [f"{chapter_slug}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"chapter_slug": chapter_slug, "chapter_title": chapter_title, "chunk_idx": i}
            for i in range(len(chunks))
        ]

        embeddings = self._embed(chunks)
        kwargs: Dict[str, Any] = {
            "ids": ids,
            "documents": chunks,
            "metadatas": metadatas,
        }
        if embeddings:
            kwargs["embeddings"] = embeddings

        self._collection.upsert(**kwargs)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
        self._ensure_ready()

        query_kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}

        embedding = self._embed([query])
        if embedding:
            query_kwargs = {
                "query_embeddings": embedding,
                "n_results": n_results,
            }

        if chapter_filter:
            query_kwargs["where"] = {"chapter_slug": chapter_filter}

        results = self._collection.query(**query_kwargs)

        hits: List[Dict[str, Any]] = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_ready()
        count = self._collection.count()
        return {"collection": COLLECTION_NAME, "total_chunks": count, "model": self._model_name}

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += size - overlap
        return chunks
