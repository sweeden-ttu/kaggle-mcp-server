"""Embedding generation and RAG retrieval for MLSysEng MoE using ChromaDB."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingEngine:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: Optional[str] = None):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name or DEFAULT_MODEL
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False),
                )
            except ImportError:
                raise RuntimeError("chromadb is required. Install with: pip install chromadb")
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _doc_id(self, chapter_num: int, chunk_idx: int) -> str:
        return f"ch{chapter_num:02d}_chunk{chunk_idx:04d}"

    def index_chapter(self, chapter_num: int, title: str, content: str, concepts: Optional[List[str]] = None):
        """Index a chapter's content into ChromaDB with embeddings."""
        if not content.strip():
            logger.warning("Empty content for chapter %d, skipping", chapter_num)
            return 0

        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        ids = [self._doc_id(chapter_num, i) for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_num": chapter_num,
                "title": title,
                "chunk_idx": i,
                "total_chunks": len(chunks),
                "concepts": ",".join(concepts or []),
            }
            for i in range(len(chunks))
        ]

        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            existing_ids = set(existing["ids"]) if existing["ids"] else set()
        except Exception:
            pass

        if existing_ids:
            self.collection.delete(ids=list(existing_ids))

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=chunks[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info("Indexed chapter %d (%s): %d chunks", chapter_num, title, len(chunks))
        return len(chunks)

    def search(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search over indexed ML Principles content."""
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        where_filter = None
        if chapter_filter is not None:
            where_filter = {"chapter_num": chapter_filter}

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        hits = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 1.0,
                    "similarity": 1.0 - (results["distances"][0][i] if results["distances"] else 1.0),
                })

        return hits

    def infer_skills(self, competition_description: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Infer which experts and skills are relevant for a competition.

        Searches the knowledge base to find relevant ML concepts, then maps
        them to experts and their skills.
        """
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[int, float] = {}
        chapter_titles: Dict[int, str] = {}
        for hit in hits:
            ch = hit["metadata"].get("chapter_num", 0)
            sim = hit.get("similarity", 0.0)
            chapter_scores[ch] = max(chapter_scores.get(ch, 0.0), sim)
            chapter_titles[ch] = hit["metadata"].get("title", f"Chapter {ch}")

        inferred = []
        for ch_num, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
            inferred.append({
                "chapter_num": ch_num,
                "title": chapter_titles[ch_num],
                "relevance_score": round(score, 4),
            })

        return inferred

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding collection statistics."""
        try:
            count = self.collection.count()
            return {"collection": COLLECTION_NAME, "total_chunks": count, "model": self.model_name}
        except Exception:
            return {"collection": COLLECTION_NAME, "total_chunks": 0, "model": self.model_name}
