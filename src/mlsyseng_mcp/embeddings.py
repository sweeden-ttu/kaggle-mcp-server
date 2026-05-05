"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingStore:
    """Manage embeddings for ML Principles chapters via ChromaDB."""

    def __init__(self, chroma_path: Optional[str] = None):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MODEL_NAME)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._client = chromadb.Client(
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=self.chroma_path,
                        anonymized_telemetry=False,
                    )
                )
            except TypeError:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def index_chapter(
        self,
        chapter_id: int,
        folder_name: str,
        title: str,
        content: str,
        concepts: List[str],
    ) -> int:
        """Index a chapter's content into ChromaDB.

        Returns:
            Number of chunks indexed.
        """
        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        ids = [f"{folder_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_id": chapter_id,
                "folder_name": folder_name,
                "title": title,
                "chunk_index": i,
                "concepts": ", ".join(concepts),
            }
            for i in range(len(chunks))
        ]

        embeddings = self.embed_texts(chunks)

        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            if existing and existing.get("ids"):
                existing_ids = set(existing["ids"])
        except Exception:
            pass

        if existing_ids:
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
        else:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters.

        Returns:
            List of result dicts with document, metadata, and distance.
        """
        query_embedding = self.embed_text(query)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output: List[Dict[str, Any]] = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                entry: Dict[str, Any] = {
                    "document": doc,
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
                if results.get("metadatas") and results["metadatas"][0]:
                    entry["metadata"] = results["metadatas"][0][i]
                output.append(entry)

        return output

    def infer_skills(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Given a query (e.g. competition description), infer which experts/skills are relevant.

        Returns:
            List of dicts with expert info and relevance scores.
        """
        results = self.search(query, n_results=n_results)

        seen_chapters: Dict[str, Dict[str, Any]] = {}
        for r in results:
            meta = r.get("metadata", {})
            folder = meta.get("folder_name", "unknown")
            if folder not in seen_chapters:
                seen_chapters[folder] = {
                    "folder_name": folder,
                    "title": meta.get("title", ""),
                    "concepts": meta.get("concepts", "").split(", ") if meta.get("concepts") else [],
                    "relevance": 1.0 - (r.get("distance", 1.0) or 1.0),
                    "sample_text": r.get("document", "")[:200],
                }
            else:
                current_relevance = 1.0 - (r.get("distance", 1.0) or 1.0)
                if current_relevance > seen_chapters[folder]["relevance"]:
                    seen_chapters[folder]["relevance"] = current_relevance

        return sorted(
            seen_chapters.values(),
            key=lambda x: x["relevance"],
            reverse=True,
        )

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "model": MODEL_NAME,
            "collection": COLLECTION_NAME,
            "chroma_path": self.chroma_path,
            "total_chunks": count,
        }
