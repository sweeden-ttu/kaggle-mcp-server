"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over ML Principles content for expert/skill selection.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _default_chroma_path() -> str:
    return os.environ.get(
        "CHROMA_DB_PATH",
        str(Path.home() / ".openclaw" / "workspace" / "mlsyseng" / "chroma_db"),
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


class EmbeddingEngine:
    """Manages embeddings and semantic search via ChromaDB."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.chroma_path = chroma_path or _default_chroma_path()
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="ml_principles",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chapter(self, chapter_number: int, title: str, content: str, concepts: List[str]):
        """Index a chapter's content into ChromaDB."""
        chunks = _chunk_text(content)
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"ch{chapter_number:02d}_chunk_{i:04d}_{_content_hash(chunk)}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "concepts": ",".join(concepts[:10]),
            })

        embeddings = self.embed_texts(documents)

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(f"Indexed chapter {chapter_number} ({len(chunks)} chunks)")

    def index_all_chapters(self, db: Database):
        """Index all chapters from the database into ChromaDB."""
        chapters = db.get_all_chapters()
        for chapter in chapters:
            if chapter.get("markdown_content"):
                self.index_chapter(
                    chapter_number=chapter["chapter_number"],
                    title=chapter["title"],
                    content=chapter["markdown_content"],
                    concepts=chapter.get("concepts", []),
                )

    def search(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search over indexed ML Principles content."""
        query_embedding = self.embed_texts([query])[0]

        where_filter = None
        if chapter_filter is not None:
            where_filter = {"chapter_number": chapter_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "content": doc,
                    "chapter_number": meta.get("chapter_number"),
                    "title": meta.get("title"),
                    "chunk_index": meta.get("chunk_index"),
                    "concepts": meta.get("concepts", "").split(","),
                    "similarity": 1.0 - dist,
                })

        return hits

    def infer_relevant_experts(
        self, query: str, db: Database, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Infer which experts are most relevant for a query using RAG."""
        search_results = self.search(query, n_results=top_k * 2)

        chapter_scores: Dict[int, float] = {}
        for hit in search_results:
            ch_num = hit["chapter_number"]
            score = hit["similarity"]
            chapter_scores[ch_num] = max(chapter_scores.get(ch_num, 0), score)

        sorted_chapters = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        relevant_experts = []
        all_experts = db.get_all_experts()
        expert_by_chapter = {e.get("chapter_id"): e for e in all_experts}

        for ch_num, score in sorted_chapters:
            chapter = db.get_chapter(ch_num)
            if not chapter:
                continue
            expert = expert_by_chapter.get(chapter["id"])
            if expert:
                expert["relevance_score"] = score
                relevant_experts.append(expert)

        return relevant_experts

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding index statistics."""
        try:
            count = self.collection.count()
            return {
                "total_chunks_indexed": count,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception:
            return {
                "total_chunks_indexed": 0,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
                "status": "not initialized",
            }
