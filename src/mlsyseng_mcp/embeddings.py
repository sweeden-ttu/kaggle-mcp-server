"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_knowledge"


class EmbeddingStore:
    """Manages embeddings via sentence-transformers and ChromaDB for RAG."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: Optional[str] = None):
        self._chroma_path = chroma_path or CHROMA_DB_PATH
        self._model_name = model_name or MODEL_NAME
        self._model = None
        self._client = None
        self._collection = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; embeddings will use hash-based fallback"
            )
            self._model = None

    def _ensure_chroma(self):
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings

            Path(self._chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            logger.warning("chromadb not installed; vector search will be unavailable")
            self._client = None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings, falling back to hash-based vectors."""
        self._ensure_model()
        if self._model is not None:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()

        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).hexdigest()
            vec = [int(h[i : i + 2], 16) / 255.0 for i in range(0, min(len(h), 768), 2)]
            while len(vec) < 384:
                vec.append(0.0)
            result.append(vec[:384])
        return result

    def index_chapter(
        self,
        chapter_name: str,
        content: str,
        concepts: List[Dict[str, str]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Index a chapter's content and concepts into ChromaDB."""
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("ChromaDB not available, skipping indexing for %s", chapter_name)
            return

        chunks = self._chunk_text(content, chunk_size, chunk_overlap)

        docs = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{chapter_name}_chunk_{i}"
            docs.append(chunk)
            ids.append(doc_id)
            metadatas.append({
                "chapter": chapter_name,
                "type": "content",
                "chunk_index": i,
            })

        for concept in concepts:
            doc_id = f"{chapter_name}_concept_{concept['name'].lower().replace(' ', '_')}"
            text = f"{concept['name']}: {concept.get('description', '')}"
            docs.append(text)
            ids.append(doc_id)
            metadatas.append({
                "chapter": chapter_name,
                "type": "concept",
                "concept_name": concept["name"],
                "category": concept.get("category", "general"),
            })

        if not docs:
            return

        embeddings = self._embed(docs)
        batch_size = 100
        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            self._collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            "Indexed %d documents for chapter %s (%d chunks + %d concepts)",
            len(docs), chapter_name, len(chunks), len(concepts),
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_chapter: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed knowledge."""
        self._ensure_chroma()
        if self._collection is None:
            return []

        query_embedding = self._embed([query])[0]

        where = {}
        if filter_chapter:
            where["chapter"] = filter_chapter
        if filter_type:
            where["type"] = filter_type

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None,
                })

        return output

    def infer_skills_for_competition(
        self, competition_description: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are relevant for a competition."""
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for r in results:
            chapter = r["metadata"].get("chapter", "unknown")
            distance = r.get("distance", 1.0)
            relevance = max(0.0, 1.0 - distance)
            chapter_scores[chapter] = max(chapter_scores.get(chapter, 0.0), relevance)

        ranked = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"chapter": chapter, "relevance_score": round(score, 4)}
            for chapter, score in ranked
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return stats about the vector collection."""
        self._ensure_chroma()
        if self._collection is None:
            return {"status": "unavailable", "count": 0}
        return {
            "status": "active",
            "count": self._collection.count(),
            "name": COLLECTION_NAME,
            "path": self._chroma_path,
        }

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks by word boundary."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += chunk_size - overlap
        return chunks
