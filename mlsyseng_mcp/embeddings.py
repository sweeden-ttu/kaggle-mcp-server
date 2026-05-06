"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import MLSysEngDatabase

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DB_PATH = os.path.expanduser(
    os.getenv("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word boundaries."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingEngine:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_DB_PATH
        self.model_name = model_name
        self._model = None
        self._client = None
        self._collection = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._model

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.chroma_path)
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
        return self._collection

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chapter(self, chapter_name: str, content: str, concepts: List[str]) -> Dict[str, Any]:
        """Index a chapter's content into ChromaDB."""
        collection = self._get_collection()
        chunks = _chunk_text(content)

        if not chunks:
            return {"chapter_name": chapter_name, "chunks_indexed": 0}

        embeddings = self.embed_texts(chunks)

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chapter_name}_{_content_hash(chunk)}_{i}"
            ids.append(chunk_id)
            metadatas.append({
                "chapter_name": chapter_name,
                "chunk_index": i,
                "concepts": ",".join(concepts[:10]),
            })

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return {
            "chapter_name": chapter_name,
            "chunks_indexed": len(chunks),
        }

    def index_all_chapters(self, db: MLSysEngDatabase) -> Dict[str, Any]:
        """Index all extracted chapters into ChromaDB."""
        chapters = db.list_chapters(status="extracted")
        results = []

        for chapter in chapters:
            result = self.index_chapter(
                chapter["chapter_name"],
                chapter["markdown_content"],
                chapter["concepts"],
            )
            results.append(result)

        total_chunks = sum(r["chunks_indexed"] for r in results)
        return {
            "status": "complete",
            "chapters_indexed": len(results),
            "total_chunks": total_chunks,
            "results": results,
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed ML Principles content."""
        collection = self._get_collection()
        query_embedding = self.embed_texts([query])[0]

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_name": chapter_filter}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append({
                    "content": doc,
                    "chapter_name": results["metadatas"][0][i].get("chapter_name", ""),
                    "concepts": results["metadatas"][0][i].get("concepts", ""),
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "relevance": 1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                })

        return search_results

    def infer_skills_for_competition(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Infer which experts and skills are most relevant for a competition.

        Uses semantic search to find relevant chapter content, then maps
        back to experts and their associated skills.
        """
        results = self.search(competition_description, n_results=n_results)

        chapter_relevance: Dict[str, float] = {}
        for r in results:
            ch = r["chapter_name"]
            relevance = r.get("relevance", 0.0)
            if ch not in chapter_relevance or relevance > chapter_relevance[ch]:
                chapter_relevance[ch] = relevance

        ranked_chapters = sorted(
            chapter_relevance.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "query": competition_description,
            "relevant_chapters": [
                {"chapter": ch, "relevance": round(rel, 4)}
                for ch, rel in ranked_chapters
            ],
            "search_results": results,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "collection_name": COLLECTION_NAME,
                "total_chunks": count,
                "model_name": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {"error": str(e)}
