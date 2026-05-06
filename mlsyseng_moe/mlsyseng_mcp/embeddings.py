"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingEngine:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def client(self):
        """Lazy-load the ChromaDB client."""
        if self._client is None:
            import chromadb
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        chapter_slug: str,
    ):
        """Index knowledge chunks into ChromaDB."""
        if not chunks:
            return

        ids = [f"{chapter_slug}_{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "chapter_slug": chapter_slug,
                "chunk_index": i,
                **(c.get("metadata", {})),
            }
            for i, c in enumerate(chunks)
        ]

        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            if existing and existing["ids"]:
                existing_ids = set(existing["ids"])
        except Exception:
            pass

        if existing_ids:
            self.collection.delete(ids=list(existing_ids))

        embeddings = self.generate_embeddings(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(chunks)} chunks for {chapter_slug}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed knowledge."""
        query_embedding = self.generate_embeddings([query])[0]

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_slug": chapter_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1.0 - results["distances"][0][i],
                })

        return search_results

    def get_relevant_experts(
        self,
        query: str,
        expert_registry,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find the most relevant experts for a given query/competition."""
        search_results = self.search(query, n_results=n_results * 2)

        chapter_scores: Dict[str, float] = {}
        for result in search_results:
            slug = result["metadata"].get("chapter_slug", "")
            score = result["similarity"]
            chapter_scores[slug] = max(chapter_scores.get(slug, 0), score)

        sorted_chapters = sorted(
            chapter_scores.items(), key=lambda x: x[1], reverse=True
        )[:n_results]

        relevant_experts = []
        for chapter_slug, score in sorted_chapters:
            expert = expert_registry.get_expert_by_chapter(chapter_slug)
            if expert:
                expert["relevance_score"] = score
                relevant_experts.append(expert)

        return relevant_experts

    def get_context_for_competition(
        self,
        competition_name: str,
        description: str = "",
        n_results: int = 10,
    ) -> str:
        """Generate a context prompt for a competition using RAG."""
        query = f"{competition_name} {description}".strip()
        results = self.search(query, n_results=n_results)

        if not results:
            return f"No relevant ML Principles found for: {competition_name}"

        context_parts = [
            f"# ML Principles Context for: {competition_name}\n",
            "Based on semantic search of ML Principles chapters:\n",
        ]

        for i, result in enumerate(results, 1):
            chapter = result["metadata"].get("chapter", "Unknown")
            similarity = result["similarity"]
            content = result["content"][:500]
            context_parts.append(
                f"## [{i}] From: {chapter} (relevance: {similarity:.3f})\n{content}\n"
            )

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding/vector store statistics."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        return {
            "model": self.model_name,
            "chroma_path": self.chroma_path,
            "collection": COLLECTION_NAME,
            "total_vectors": count,
        }
