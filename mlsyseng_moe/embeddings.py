"""Embedding generation and RAG retrieval using ChromaDB."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingStore:
    """Vector store for semantic search over ML Principles content."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or CHROMA_DB_PATH
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_directory)
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

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def add_chunks(
        self, chapter_name: str, chunks: List[str], chapter_id: int
    ) -> int:
        """Add text chunks with embeddings to the vector store."""
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks).tolist()

        ids = [f"{chapter_name}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"chapter_name": chapter_name, "chapter_id": chapter_id, "chunk_index": i}
            for i in range(len(chunks))
        ]

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
        """Semantic search over indexed content."""
        query_embedding = self.model.encode([query]).tolist()

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_name": chapter_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "relevance": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                    }
                )

        return search_results

    def infer_experts_for_competition(
        self, competition_description: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Given a competition description, infer which experts are most relevant.

        Returns ranked list of chapter experts with relevance scores.
        """
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for r in results:
            chapter = r["metadata"].get("chapter_name", "unknown")
            score = r.get("relevance", 0)
            if chapter in chapter_scores:
                chapter_scores[chapter] = max(chapter_scores[chapter], score)
            else:
                chapter_scores[chapter] = score

        ranked = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chapter_name": ch, "relevance": score} for ch, score in ranked]

    def get_context_for_competition(
        self, competition_name: str, description: str = "", n_results: int = 10
    ) -> str:
        """Generate a context prompt for rdagent or competition entry."""
        query = f"{competition_name} {description}".strip()
        results = self.search(query, n_results=n_results)

        if not results:
            return f"No relevant ML principles found for competition: {competition_name}"

        context_parts = [
            f"## ML Principles Context for: {competition_name}\n",
            "Based on indexed ML Principles chapters, here are the most relevant concepts:\n",
        ]

        for i, r in enumerate(results, 1):
            chapter = r["metadata"].get("chapter_name", "unknown")
            relevance = r.get("relevance", 0)
            text_preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
            context_parts.append(
                f"### {i}. {chapter} (relevance: {relevance:.3f})\n{text_preview}\n"
            )

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Return embedding store statistics."""
        try:
            count = self.collection.count()
            return {
                "total_embeddings": count,
                "model": EMBEDDING_MODEL,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            return {"error": str(e)}
