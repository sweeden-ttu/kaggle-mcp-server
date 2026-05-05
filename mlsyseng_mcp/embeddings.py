"""Embedding generation and RAG retrieval for MLSysEng MoE system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)

COLLECTION_NAME = "ml_principles"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


class EmbeddingStore:
    """ChromaDB-backed embedding store for semantic search over ML Principles."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or CHROMA_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise

    def index_chapter(
        self, chapter_name: str, chapter_number: int, content: str, concepts: List[str]
    ) -> int:
        """Index a chapter's content into the vector store. Returns chunk count."""
        collection = self._get_collection()
        chunks = _chunk_text(content)

        ids = [f"{chapter_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_name": chapter_name,
                "chapter_number": chapter_number,
                "chunk_index": i,
                "concepts": ", ".join(concepts),
            }
            for i in range(len(chunks))
        ]

        existing = collection.get(where={"chapter_name": chapter_name})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        logger.info("Indexed %d chunks for chapter: %s", len(chunks), chapter_name)
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
        collection = self._get_collection()
        results = collection.query(query_texts=[query], n_results=n_results)

        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                hit = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
                hits.append(hit)
        return hits

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding store statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return {"total_chunks": count, "collection": COLLECTION_NAME, "model": EMBEDDING_MODEL}
        except Exception as e:
            return {"error": str(e)}

    def infer_skills_for_competition(
        self, competition_description: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are relevant for a competition."""
        hits = self.search(competition_description, n_results=n_results)
        chapter_scores: Dict[str, float] = {}
        chapter_concepts: Dict[str, List[str]] = {}

        for hit in hits:
            chapter = hit["metadata"].get("chapter_name", "unknown")
            distance = hit.get("distance", 1.0)
            score = 1.0 - distance

            if chapter not in chapter_scores or score > chapter_scores[chapter]:
                chapter_scores[chapter] = score
            concepts = hit["metadata"].get("concepts", "")
            if concepts:
                chapter_concepts.setdefault(chapter, []).extend(
                    [c.strip() for c in concepts.split(",")]
                )

        recommendations = []
        for chapter, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
            concepts = sorted(set(chapter_concepts.get(chapter, [])))
            recommendations.append(
                {"chapter": chapter, "relevance_score": round(score, 4), "concepts": concepts}
            )
        return recommendations
