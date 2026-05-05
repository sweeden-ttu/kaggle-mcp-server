"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB.

Chunks chapter content, generates embeddings, stores in ChromaDB, and
provides semantic search for skill inference.
"""

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from .database import MoEDatabase

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap
    return chunks


class EmbeddingEngine:
    """Manages embedding generation and ChromaDB vector storage."""

    def __init__(
        self,
        db: Optional[MoEDatabase] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        chroma_path: Optional[str] = None,
    ):
        self.db = db or MoEDatabase()
        self.model_name = model_name
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self._model = None
        self._chroma_client = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                os.makedirs(self.chroma_path, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False),
                )
                self._collection = self._chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
        return self._collection

    def embed_chapter(self, chapter_id: str, force: bool = False) -> int:
        """Generate embeddings for a chapter's content and store in ChromaDB.

        Returns the number of chunks embedded.
        """
        chapter = self.db.get_chapter(chapter_id)
        if not chapter:
            raise ValueError(f"Chapter '{chapter_id}' not found")

        content = chapter.get("content_md", "")
        if not content:
            logger.warning("No content for chapter %s", chapter_id)
            return 0

        existing = self.db.count_embeddings(chapter_id)
        if existing > 0 and not force:
            logger.info("Chapter %s already embedded (%d chunks)", chapter_id, existing)
            return existing

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            emb_id = f"{chapter_id}_chunk_{i}"
            ids.append(emb_id)
            metadatas.append({
                "chapter_id": chapter_id,
                "chunk_index": i,
                "chapter_title": chapter.get("title", ""),
            })
            self.db.insert_embedding_meta(
                embedding_id=emb_id,
                chapter_id=chapter_id,
                chunk_index=i,
                chunk_text=chunk[:500],
                model_name=self.model_name,
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info("Embedded %d chunks for chapter %s", len(chunks), chapter_id)
        return len(chunks)

    def embed_all_chapters(self, force: bool = False) -> Dict[str, int]:
        """Embed all extracted chapters."""
        chapters = self.db.list_chapters()
        results = {}
        for ch in chapters:
            if ch.get("status") != "extracted":
                continue
            try:
                count = self.embed_chapter(ch["chapter_id"], force=force)
                results[ch["chapter_id"]] = count
            except Exception as exc:
                logger.error("Embedding failed for %s: %s", ch["chapter_id"], exc)
                results[ch["chapter_id"]] = -1
        return results

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over embedded ML Principles content."""
        query_embedding = self.model.encode([query]).tolist()

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_id": chapter_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                hit = {
                    "id": doc_id,
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    "similarity": 1.0 - (results["distances"][0][i] if results.get("distances") else 0.0),
                }
                hits.append(hit)
        return hits

    def infer_skills_for_competition(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> Dict[str, Any]:
        """Use RAG to infer which experts and skills apply to a competition."""
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        chapter_titles: Dict[str, str] = {}
        for hit in hits:
            cid = hit["metadata"].get("chapter_id", "")
            if cid:
                chapter_scores[cid] = chapter_scores.get(cid, 0.0) + hit["similarity"]
                chapter_titles[cid] = hit["metadata"].get("chapter_title", cid)

        ranked_chapters = sorted(
            chapter_scores.items(), key=lambda x: x[1], reverse=True
        )

        from .expert_registry import ExpertRegistry
        registry = ExpertRegistry(self.db)

        recommended = []
        for cid, score in ranked_chapters[:5]:
            experts = self.db.list_experts()
            for exp in experts:
                if exp.get("chapter_id") == cid:
                    recommended.append({
                        "expert": exp,
                        "relevance_score": round(score, 4),
                        "chapter_title": chapter_titles.get(cid, ""),
                    })

        all_skills = set()
        for rec in recommended:
            all_skills.update(rec["expert"].get("skills", []))

        return {
            "recommended_experts": recommended,
            "skills": sorted(all_skills),
            "search_hits": len(hits),
            "chapters_matched": len(ranked_chapters),
        }

    def get_rdagent_context(
        self,
        competition_name: str,
        description: str = "",
        n_results: int = 5,
    ) -> str:
        """Generate a context prompt for rdagent from ML Principles knowledge."""
        query = f"{competition_name} {description}".strip()
        hits = self.search(query, n_results=n_results)

        if not hits:
            return f"No ML Principles context found for '{competition_name}'."

        lines = [
            f"# ML Principles Context for: {competition_name}",
            "",
            "The following ML principles are relevant to this competition:",
            "",
        ]
        for i, hit in enumerate(hits, 1):
            chapter = hit["metadata"].get("chapter_title", "Unknown")
            sim = hit.get("similarity", 0)
            lines.append(f"## Principle {i} (from {chapter}, relevance: {sim:.2f})")
            lines.append("")
            lines.append(hit.get("text", "")[:1000])
            lines.append("")

        return "\n".join(lines)
