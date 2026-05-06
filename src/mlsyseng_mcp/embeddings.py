"""
Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML Principles chapters to infer
which experts and skills are needed for a given competition.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "mlsyseng_chapters"


class EmbeddingEngine:
    """Manages embeddings and semantic search via ChromaDB + sentence-transformers."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.chroma_path = chroma_path or os.environ.get(
            "CHROMA_DB_PATH",
            str(Path.home() / ".mlsyseng" / "chroma_db"),
        )
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                os.makedirs(self.chroma_path, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.chroma_path)
                self._collection = self._client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise RuntimeError(
                    "chromadb is required. Install with: pip install chromadb"
                )
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap

        return chunks

    def index_chapter(
        self,
        chapter_id: str,
        chapter_name: str,
        content: str,
        concepts: List[str],
    ) -> int:
        """
        Index a chapter's content into ChromaDB.

        Returns the number of chunks indexed.
        """
        collection = self._get_collection()
        model = self._get_model()

        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        ids = [f"{chapter_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = model.encode(chunks).tolist()
        metadatas = [
            {
                "chapter_id": chapter_id,
                "chapter_name": chapter_name,
                "chunk_index": i,
                "concepts": ",".join(concepts),
            }
            for i in range(len(chunks))
        ]

        existing = set()
        try:
            existing_docs = collection.get(ids=ids)
            if existing_docs and existing_docs["ids"]:
                existing = set(existing_docs["ids"])
        except Exception:
            pass

        new_ids = [id_ for id_ in ids if id_ not in existing]
        if not new_ids:
            return 0

        idx_map = {id_: i for i, id_ in enumerate(ids)}
        new_indices = [idx_map[id_] for id_ in new_ids]

        collection.add(
            ids=new_ids,
            embeddings=[embeddings[i] for i in new_indices],
            documents=[chunks[i] for i in new_indices],
            metadatas=[metadatas[i] for i in new_indices],
        )

        return len(new_ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed chapters.

        Returns list of results with score, content, and metadata.
        """
        collection = self._get_collection()
        model = self._get_model()

        query_embedding = model.encode([query]).tolist()

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_id": chapter_filter}

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity": 1.0 - results["distances"][0][i],
                    }
                )

        return output

    def infer_relevant_experts(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Given a competition description, infer which chapter experts are relevant.

        Returns a ranked list of chapter_ids with relevance scores.
        """
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        chapter_names: Dict[str, str] = {}
        chapter_concepts: Dict[str, set] = {}

        for r in results:
            ch_id = r["metadata"]["chapter_id"]
            ch_name = r["metadata"]["chapter_name"]
            sim = r["similarity"]

            if ch_id not in chapter_scores or chapter_scores[ch_id] < sim:
                chapter_scores[ch_id] = sim

            chapter_names[ch_id] = ch_name

            if ch_id not in chapter_concepts:
                chapter_concepts[ch_id] = set()
            for c in r["metadata"].get("concepts", "").split(","):
                if c.strip():
                    chapter_concepts[ch_id].add(c.strip())

        ranked = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "chapter_id": ch_id,
                "chapter_name": chapter_names[ch_id],
                "relevance_score": score,
                "concepts": sorted(chapter_concepts.get(ch_id, set())),
            }
            for ch_id, score in ranked
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return statistics about the indexed collection."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "total_chunks": count,
                "collection_name": _COLLECTION_NAME,
                "chroma_path": self.chroma_path,
                "model": self.model_name,
            }
        except Exception as e:
            return {"error": str(e)}
