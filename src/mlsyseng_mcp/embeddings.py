"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML Principles content for
skill inference and expert matching.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


def _get_chroma_path() -> str:
    return os.environ.get(
        "CHROMA_DB_PATH",
        os.path.expanduser("~/.mlsyseng/chroma_db"),
    )


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _text_id(text: str, prefix: str = "") -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}" if prefix else h


class EmbeddingEngine:
    """Manages embeddings and semantic search via sentence-transformers + ChromaDB."""

    def __init__(
        self,
        db: Database,
        model_name: str = DEFAULT_MODEL,
        chroma_path: Optional[str] = None,
    ):
        self.db = db
        self.model_name = model_name
        self.chroma_path = chroma_path or _get_chroma_path()
        self._model = None
        self._chroma_client = None
        self._collection = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )
        return self._model

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise RuntimeError(
                    "chromadb is required: pip install chromadb"
                )
            os.makedirs(self.chroma_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self._collection = self._chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chapter(self, chapter_num: int) -> Dict[str, Any]:
        """Generate embeddings for a chapter's content and store in ChromaDB."""
        chapter = self.db.get_chapter(chapter_num)
        if not chapter or not chapter.get("markdown_content"):
            return {"status": "error", "message": f"Chapter {chapter_num} not found or empty"}

        model = self._get_model()
        collection = self._get_collection()
        text = chapter["markdown_content"]
        chunks = _chunk_text(text)

        if not chunks:
            return {"status": "skipped", "message": "No content to embed"}

        prefix = f"ch{chapter_num:02d}"
        ids = [_text_id(c, prefix) for c in chunks]
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        metadatas = [
            {
                "chapter_num": chapter_num,
                "chapter_title": chapter["title"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        return {
            "status": "completed",
            "chapter_num": chapter_num,
            "chunks_indexed": len(chunks),
        }

    def index_all_chapters(self) -> List[Dict[str, Any]]:
        """Index all chapters from the database."""
        chapters = self.db.list_chapters()
        results = []
        for ch in chapters:
            try:
                result = self.index_chapter(ch["chapter_num"])
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to index chapter {ch['chapter_num']}: {e}")
                results.append({
                    "status": "error",
                    "chapter_num": ch["chapter_num"],
                    "message": str(e),
                })
        return results

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
        model = self._get_model()
        collection = self._get_collection()

        query_embedding = model.encode([query], show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                dist = results["distances"][0][i] if results.get("distances") else None
                matches.append({
                    "text": doc,
                    "chapter_num": meta.get("chapter_num"),
                    "chapter_title": meta.get("chapter_title", ""),
                    "similarity": 1.0 - dist if dist is not None else None,
                    "chunk_index": meta.get("chunk_index"),
                })
        return matches

    def infer_experts_for_query(self, query: str, n_results: int = 10) -> List[str]:
        """Given a query (e.g., competition description), infer which expert slugs are relevant."""
        matches = self.search(query, n_results=n_results)
        chapter_nums = set()
        for m in matches:
            if m.get("chapter_num"):
                chapter_nums.add(m["chapter_num"])

        experts = self.db.list_experts()
        relevant_slugs = []
        for expert in experts:
            if expert.get("chapter_id"):
                ch = self.db.get_chapter_by_id(expert["chapter_id"]) if hasattr(self.db, "get_chapter_by_id") else None
                if ch and ch.get("chapter_num") in chapter_nums:
                    relevant_slugs.append(expert["slug"])
            else:
                relevant_slugs.append(expert["slug"])

        return relevant_slugs if relevant_slugs else [e["slug"] for e in experts[:3]]

    def get_context_for_competition(self, competition_name: str, description: str = "") -> str:
        """Generate a context prompt for a competition using RAG."""
        query = f"{competition_name} {description}".strip()
        matches = self.search(query, n_results=8)

        if not matches:
            return f"No ML Principles context found for '{competition_name}'."

        context_parts = [f"## ML Principles Context for '{competition_name}'\n"]
        seen_chapters = set()
        for m in matches:
            ch_title = m.get("chapter_title", "Unknown")
            if ch_title not in seen_chapters:
                context_parts.append(f"\n### From: {ch_title}\n")
                seen_chapters.add(ch_title)
            text_preview = m["text"][:500] if len(m["text"]) > 500 else m["text"]
            sim = m.get("similarity")
            sim_str = f" (relevance: {sim:.3f})" if sim is not None else ""
            context_parts.append(f"{text_preview}{sim_str}\n")

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding index statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "total_chunks": count,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
                "error": str(e),
            }
