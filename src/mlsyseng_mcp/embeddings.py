"""Embedding generation and RAG retrieval with ChromaDB.

Uses sentence-transformers (all-MiniLM-L6-v2) for semantic search
across extracted ML Principles content.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _default_chroma_path() -> str:
    return os.environ.get(
        "CHROMA_DB_PATH",
        os.path.expanduser("~/.mlsyseng/chroma_db"),
    )


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingEngine:
    """Manages embeddings and RAG retrieval over ML Principles content."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
    ):
        self.chroma_path = chroma_path or _default_chroma_path()
        self.model_name = model_name
        self._collection = None
        self._client = None
        self._embedding_fn = None

    def _ensure_chroma(self):
        """Lazy-initialize ChromaDB client and collection."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for embedding operations. "
                "Install with: pip install chromadb"
            )

        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.chroma_path)

        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not available; ChromaDB will use default embeddings"
            )
            self._embedding_fn = None

        self._collection = self._client.get_or_create_collection(
            name="mlsyseng_principles",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chapter(
        self,
        chapter_name: str,
        slug: str,
        content: str,
        concepts: Optional[List[str]] = None,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> int:
        """Index a chapter's content into ChromaDB."""
        self._ensure_chroma()

        existing = self._collection.get(
            where={"chapter_slug": slug},
        )
        if existing and existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        chunks = _chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{slug}_chunk_{i}_{_content_hash(chunk)}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_name": chapter_name,
                "chapter_slug": slug,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "concepts": ",".join(concepts) if concepts else "",
            })

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Indexed %d chunks for chapter '%s'", len(chunks), chapter_name)
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        self._ensure_chroma()

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_slug": chapter_filter}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                item = {
                    "content": doc,
                    "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
                items.append(item)

        return items

    def index_all_chapters(self, db: Database) -> Dict[str, int]:
        """Index all chapters from the database into ChromaDB."""
        chapters = db.list_chapters()
        stats = {"indexed": 0, "chunks": 0, "skipped": 0}

        for ch in chapters:
            full = db.get_chapter(ch["chapter_name"])
            if not full or not full.get("markdown_content"):
                stats["skipped"] += 1
                continue

            concepts = db.get_concepts_for_chapter(ch["id"])
            concept_names = [c["concept_name"] for c in concepts]

            n_chunks = self.index_chapter(
                chapter_name=ch["chapter_name"],
                slug=ch["slug"],
                content=full["markdown_content"],
                concepts=concept_names,
            )
            stats["indexed"] += 1
            stats["chunks"] += n_chunks

        return stats

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return stats about the ChromaDB collection."""
        self._ensure_chroma()
        count = self._collection.count()
        return {
            "total_documents": count,
            "chroma_path": self.chroma_path,
            "model": self.model_name,
            "collection_name": "mlsyseng_principles",
        }

    def infer_skills_for_competition(
        self,
        competition_description: str,
        n_results: int = 10,
        db: Optional[Database] = None,
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are relevant for a competition."""
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for r in results:
            slug = r["metadata"].get("chapter_slug", "")
            score = r.get("score", 0.0)
            chapter_scores[slug] = max(chapter_scores.get(slug, 0.0), score)

        recommendations = []
        if db:
            for slug, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
                expert = db.get_expert_by_slug(slug)
                if expert:
                    recommendations.append({
                        "expert_name": expert["expert_name"],
                        "slug": slug,
                        "relevance_score": round(score, 4),
                        "capabilities": expert.get("capabilities", []),
                        "skills": expert.get("skills", []),
                        "strategy": expert.get("strategy", ""),
                    })

        return recommendations
