"""Embedding generation and RAG retrieval for MLSysEng MoE.

Uses sentence-transformers for embedding generation and ChromaDB
for vector storage and similarity search.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from .database import ChapterRecord, Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _default_chroma_path() -> str:
    return os.environ.get(
        "CHROMA_DB_PATH",
        os.path.expanduser("~/.mlsyseng/chroma_db"),
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _chunk_id(chapter_id: str, chunk_idx: int) -> str:
    raw = f"{chapter_id}::chunk_{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


class EmbeddingEngine:
    """Manages embeddings and semantic search over chapter content."""

    def __init__(
        self,
        db: Database,
        model_name: str = DEFAULT_MODEL,
        chroma_path: Optional[str] = None,
    ):
        self.db = db
        self.model_name = model_name
        self.chroma_path = chroma_path or _default_chroma_path()
        self._model = None
        self._collection = None
        self._client = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.chroma_path)
            self._collection = self._client.get_or_create_collection(
                name="mlsyseng_chapters",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection ready at %s", self.chroma_path)
        except ImportError:
            logger.error(
                "chromadb not installed. Install with: pip install chromadb"
            )
            raise

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        self._ensure_model()
        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def index_chapter(self, chapter: ChapterRecord) -> int:
        """Index a chapter's content into ChromaDB. Returns chunk count."""
        self._ensure_model()
        self._ensure_collection()

        chunks = _chunk_text(chapter.content_md)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            chunk_id = _chunk_id(chapter.chapter_id, i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            embeddings.append(self.embed_text(chunk))

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(
            "Indexed %d chunks for chapter: %s", len(chunks), chapter.title
        )
        return len(chunks)

    def index_all_chapters(self) -> dict[str, int]:
        """Index all chapters from the database. Returns {chapter_id: chunk_count}."""
        chapters = self.db.list_chapters()
        results = {}
        for chapter in chapters:
            if chapter.content_md:
                count = self.index_chapter(chapter)
                results[chapter.chapter_id] = count
        return results

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search over indexed chapter content."""
        self._ensure_model()
        self._ensure_collection()

        query_embedding = self.embed_text(query)

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_id": chapter_filter}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": 1.0 - results["distances"][0][i],
                })
        return hits

    def infer_relevant_experts(
        self, query: str, n_results: int = 10
    ) -> list[dict]:
        """Infer which experts are most relevant for a query.

        Returns experts ranked by how many of the top search results
        belong to their chapter.
        """
        hits = self.search(query, n_results=n_results)
        chapter_scores: dict[str, float] = {}

        for hit in hits:
            cid = hit["metadata"]["chapter_id"]
            chapter_scores[cid] = chapter_scores.get(cid, 0.0) + hit["relevance"]

        experts = self.db.list_experts()
        expert_rankings = []
        for expert in experts:
            score = chapter_scores.get(expert.chapter_id, 0.0)
            if score > 0:
                expert_rankings.append({
                    "expert_name": expert.expert_name,
                    "slug": expert.slug,
                    "relevance_score": round(score, 4),
                    "capabilities": expert.capabilities,
                    "skills": expert.skills,
                })

        expert_rankings.sort(key=lambda x: x["relevance_score"], reverse=True)
        return expert_rankings

    def get_collection_stats(self) -> dict:
        """Get statistics about the ChromaDB collection."""
        try:
            self._ensure_collection()
            count = self._collection.count()
            return {
                "total_chunks": count,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {"error": str(e)}
