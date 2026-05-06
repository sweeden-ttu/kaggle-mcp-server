"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML chapter content and
skill inference for competition entries.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser("~/.mlsyseng/chroma_db")
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingStore:
    """Manages embeddings in ChromaDB with sentence-transformer encoding."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: str = MODEL_NAME):
        self.chroma_path = chroma_path or os.environ.get(
            "CHROMA_DB_PATH", DEFAULT_CHROMA_PATH)
        Path(self.chroma_path).parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
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
                logger.warning("sentence-transformers not installed; embeddings disabled")
                return None
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.chroma_path)
                self._collection = self._client.get_or_create_collection(
                    name="ml_principles",
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.warning("chromadb not installed; vector storage disabled")
                return None
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = 500,
                    overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for embedding."""
        if not text:
            return []
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _text_id(self, text: str, prefix: str = "") -> str:
        h = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"{prefix}_{h}" if prefix else h

    def index_chapter(self, chapter_num: int, title: str, content: str,
                      concepts: Optional[List[Dict[str, str]]] = None):
        """Index a chapter's content into ChromaDB."""
        if not self.collection or not self.model:
            logger.warning("Cannot index: missing chromadb or sentence-transformers")
            return

        chunks = self._chunk_text(content)
        if not chunks:
            return

        prefix = f"ch{chapter_num:02d}"

        existing = self.collection.get(where={"chapter_num": chapter_num})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        ids = [self._text_id(c, prefix) for c in chunks]
        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()
        metadatas = [
            {"chapter_num": chapter_num, "title": title, "chunk_idx": i}
            for i in range(len(chunks))
        ]

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=chunks[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        if concepts:
            concept_texts = [
                f"{c['concept']}: {c.get('description', '')}" for c in concepts
            ]
            concept_ids = [self._text_id(t, f"{prefix}_concept") for t in concept_texts]
            concept_embeddings = self.model.encode(
                concept_texts, show_progress_bar=False).tolist()
            concept_metas = [
                {"chapter_num": chapter_num, "title": title,
                 "type": "concept", "concept": c["concept"],
                 "category": c.get("category", "general")}
                for c in concepts
            ]
            self.collection.add(
                ids=concept_ids,
                embeddings=concept_embeddings,
                documents=concept_texts,
                metadatas=concept_metas,
            )

        logger.info("Indexed chapter %d: %d chunks, %d concepts",
                     chapter_num, len(chunks), len(concepts or []))

    def search(self, query: str, n_results: int = 5,
               chapter_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        if not self.collection or not self.model:
            return []

        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        where = {"chapter_num": chapter_filter} if chapter_filter else None
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1.0 - results["distances"][0][i],
                })
        return hits

    def infer_skills(self, competition_description: str,
                     experts: List[Dict[str, Any]],
                     n_results: int = 10) -> List[Dict[str, Any]]:
        """Infer which experts and skills are relevant for a competition.

        Uses semantic search to find relevant content, then maps back to experts.
        """
        hits = self.search(competition_description, n_results=n_results)

        expert_scores: Dict[str, float] = {}
        expert_hits: Dict[str, List[str]] = {}

        for hit in hits:
            ch_num = hit["metadata"].get("chapter_num")
            if ch_num is None:
                continue
            for expert in experts:
                if expert.get("chapter_id") and self._expert_matches_chapter(expert, ch_num):
                    slug = expert["slug"]
                    score = hit.get("similarity", 0)
                    expert_scores[slug] = max(expert_scores.get(slug, 0), score)
                    if slug not in expert_hits:
                        expert_hits[slug] = []
                    snippet = hit["document"][:200]
                    expert_hits[slug].append(snippet)

        recommendations = []
        for expert in experts:
            slug = expert["slug"]
            if slug in expert_scores:
                recommendations.append({
                    "expert": expert,
                    "relevance_score": expert_scores[slug],
                    "matching_context": expert_hits.get(slug, []),
                })

        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return recommendations

    @staticmethod
    def _expert_matches_chapter(expert: Dict[str, Any], chapter_num: int) -> bool:
        slug = expert.get("slug", "")
        ch_id = expert.get("chapter_id")
        if ch_id == chapter_num:
            return True
        try:
            num = int(slug.split("_")[0])
            return num == chapter_num
        except (ValueError, IndexError):
            return False

    def get_stats(self) -> Dict[str, Any]:
        if not self.collection:
            return {"status": "disabled", "reason": "chromadb not available"}
        count = self.collection.count()
        return {
            "total_vectors": count,
            "chroma_path": self.chroma_path,
            "model": self.model_name,
        }
