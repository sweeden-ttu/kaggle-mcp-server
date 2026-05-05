"""Embedding generation and RAG retrieval with ChromaDB."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_chroma_path() -> Path:
    return Path(os.environ.get(
        "CHROMA_DB_PATH",
        os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db")
    ))


class EmbeddingEngine:
    """Manages embeddings and semantic search using sentence-transformers and ChromaDB."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "ml_principles"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    def __init__(self, chroma_path: Optional[Path] = None):
        self.chroma_path = chroma_path or _default_chroma_path()
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks by words."""
        chunk_size = chunk_size or self.CHUNK_SIZE
        overlap = overlap or self.CHUNK_OVERLAP
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def index_chapter(self, chapter_number: int, title: str, content: str, concepts: List[str]):
        """Index a chapter's content into ChromaDB."""
        collection = self._get_collection()
        model = self._get_model()

        chunks = self._chunk_text(content)
        if not chunks:
            logger.warning(f"No content to index for chapter {chapter_number}")
            return

        ids = [f"ch{chapter_number:02d}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
                "concepts": ",".join(concepts[:10]),
            }
            for i in range(len(chunks))
        ]

        embeddings = model.encode(chunks).tolist()

        existing_ids = set()
        try:
            existing = collection.get(where={"chapter_number": chapter_number})
            if existing and existing["ids"]:
                existing_ids = set(existing["ids"])
                collection.delete(ids=list(existing_ids))
        except Exception:
            pass

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:batch_end],
                documents=chunks[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(f"Indexed {len(chunks)} chunks for chapter {chapter_number}: {title}")

    def search(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        collection = self._get_collection()
        model = self._get_model()

        query_embedding = model.encode([query]).tolist()

        where_filter = None
        if chapter_filter is not None:
            where_filter = {"chapter_number": chapter_filter}

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": 1.0 - results["distances"][0][i],
                })

        return output

    def infer_skills(self, competition_description: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Infer which experts and skills are most relevant for a competition."""
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[int, float] = {}
        chapter_titles: Dict[int, str] = {}
        for r in results:
            ch_num = r["metadata"]["chapter_number"]
            score = r["relevance"]
            chapter_scores[ch_num] = max(chapter_scores.get(ch_num, 0), score)
            chapter_titles[ch_num] = r["metadata"]["title"]

        ranked = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "chapter_number": ch_num,
                "title": chapter_titles[ch_num],
                "relevance_score": score,
            }
            for ch_num, score in ranked
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding index statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.COLLECTION_NAME,
                "model": self.MODEL_NAME,
                "chroma_path": str(self.chroma_path),
            }
        except Exception as e:
            return {"error": str(e)}
