"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB.

Provides semantic search over extracted ML Principles chapter content
and skill inference for competitions.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"


class EmbeddingStore:
    """Manages embeddings with sentence-transformers and ChromaDB."""

    def __init__(
        self,
        chroma_path: str = CHROMA_DB_PATH,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chroma_path = chroma_path
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
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.error("chromadb not installed. Run: pip install chromadb")
                raise
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping chunks for embedding."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks if chunks else [text[:2000]]

    def index_chapter(
        self,
        chapter_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Index a chapter's content into ChromaDB.

        Returns the number of chunks indexed.
        """
        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{chapter_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_id": chapter_id,
                "title": title,
                "chunk_index": i,
                **(metadata or {}),
            }
            for i in range(len(chunks))
        ]

        # Upsert to handle re-indexing gracefully
        self.collection.upsert(
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
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content.

        Returns list of results with document text, metadata, and distance.
        """
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_id": chapter_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": 1.0 - dist,
                })

        return output

    def infer_skills(
        self,
        competition_description: str,
        available_experts: List[Dict[str, Any]],
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are most relevant for a competition.

        Uses semantic search to match competition description against
        indexed chapter content, then maps results to experts.
        """
        search_results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for result in search_results:
            cid = result["metadata"].get("chapter_id", "")
            similarity = result.get("similarity", 0.0)
            chapter_scores[cid] = max(chapter_scores.get(cid, 0.0), similarity)

        expert_recommendations = []
        for expert in available_experts:
            cid = expert.get("chapter_id", "")
            slug = expert.get("slug", "")
            score = chapter_scores.get(cid, 0.0)
            if not score:
                score = chapter_scores.get(slug, 0.0)

            if score > 0:
                expert_recommendations.append({
                    "expert_name": expert["expert_name"],
                    "slug": expert["slug"],
                    "relevance_score": round(score, 4),
                    "skills": expert.get("skills", []),
                    "capabilities": expert.get("capabilities", []),
                    "strategy": expert.get("strategy", ""),
                })

        expert_recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return expert_recommendations

    def get_rdagent_context(
        self,
        competition_name: str,
        description: str = "",
        n_results: int = 5,
    ) -> str:
        """Generate a context prompt for rdagent based on ML Principles.

        Returns a formatted string with relevant knowledge chunks.
        """
        query = f"{competition_name} {description}".strip()
        results = self.search(query, n_results=n_results)

        if not results:
            return f"No relevant ML Principles found for '{competition_name}'."

        sections = [f"## ML Principles Context for: {competition_name}\n"]
        for i, result in enumerate(results, 1):
            chapter = result["metadata"].get("title", "Unknown")
            similarity = result.get("similarity", 0.0)
            text = result["document"][:500]
            sections.append(
                f"### {i}. {chapter} (relevance: {similarity:.2%})\n{text}\n"
            )

        sections.append(
            "\n## Recommendations\n"
            "Based on the above ML Principles, consider:\n"
            "1. Start with a baseline model using standard techniques\n"
            "2. Apply feature engineering informed by domain knowledge\n"
            "3. Use cross-validation for reliable performance estimates\n"
            "4. Iterate using the convergence loop until improvement plateaus\n"
        )

        return "\n".join(sections)

    def get_stats(self) -> Dict[str, Any]:
        """Return embedding store statistics."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "model": self.model_name,
                "collection": COLLECTION_NAME,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {"error": str(e)}
