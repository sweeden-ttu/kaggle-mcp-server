"""Embedding generation and RAG retrieval for MLSysEng MoE.

Uses sentence-transformers (all-MiniLM-L6-v2) for semantic search
and ChromaDB for vector storage.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingStore:
    """Manages embeddings and vector search over chapter content."""

    def __init__(self, chroma_path: str | None = None, model_name: str | None = None):
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.model_name = model_name or EMBEDDING_MODEL
        self._model = None
        self._client = None
        self._collection = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
        return self._model

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.chroma_path)
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.warning(
                    "chromadb not installed. Install with: pip install chromadb"
                )
                raise
        return self._collection

    def embed_text(self, text: str) -> list[float]:
        model = self._get_model()
        return model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return model.encode(texts).tolist()

    def index_chapter(
        self,
        chapter_number: int,
        title: str,
        content: str,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> int:
        """Split chapter content into chunks, embed, and store in ChromaDB.

        Returns the number of chunks indexed.
        """
        collection = self._get_collection()
        chunks = self._chunk_text(content, chunk_size, overlap)
        if not chunks:
            return 0

        ids = [f"ch{chapter_number}_chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        existing_ids = set()
        try:
            existing = collection.get(
                where={"chapter_number": chapter_number}
            )
            existing_ids = set(existing["ids"])
        except Exception:
            pass

        if existing_ids:
            collection.delete(ids=list(existing_ids))

        embeddings = self.embed_batch(chunks)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(
            "Indexed %d chunks for chapter %d (%s)", len(chunks), chapter_number, title
        )
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: int | None = None,
    ) -> list[dict]:
        """Semantic search over indexed chapter content."""
        collection = self._get_collection()
        query_embedding = self.embed_text(query)

        where = None
        if chapter_filter is not None:
            where = {"chapter_number": chapter_filter}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity": 1.0 - results["distances"][0][i],
                    }
                )
        return hits

    def infer_skills(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> list[dict]:
        """Infer which experts and skills are relevant for a competition.

        Searches embeddings and returns ranked expert recommendations.
        """
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: dict[int, float] = {}
        chapter_titles: dict[int, str] = {}
        for hit in hits:
            ch_num = hit["metadata"]["chapter_number"]
            score = hit["similarity"]
            chapter_scores[ch_num] = max(chapter_scores.get(ch_num, 0.0), score)
            chapter_titles[ch_num] = hit["metadata"]["title"]

        recommendations = []
        for ch_num, score in sorted(
            chapter_scores.items(), key=lambda x: x[1], reverse=True
        ):
            recommendations.append(
                {
                    "chapter_number": ch_num,
                    "title": chapter_titles[ch_num],
                    "relevance_score": round(score, 4),
                }
            )
        return recommendations

    def get_stats(self) -> dict:
        """Return collection statistics."""
        try:
            collection = self._get_collection()
            return {
                "total_chunks": collection.count(),
                "collection_name": COLLECTION_NAME,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
            if start >= len(words):
                break
        return chunks
