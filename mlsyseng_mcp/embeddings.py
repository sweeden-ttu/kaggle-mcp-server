"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import logging
import os
from typing import Any, Dict, List, Optional

from mlsyseng_mcp.database import MoEDatabase

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


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


class EmbeddingManager:
    """Manages embeddings and vector search using ChromaDB."""

    def __init__(self, chroma_path: str = DEFAULT_CHROMA_PATH):
        self.chroma_path = chroma_path
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                os.makedirs(self.chroma_path, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False),
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required for embeddings. Install with: pip install chromadb"
                )
        return self._client

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            try:
                from chromadb.utils import embedding_functions

                self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._embedding_fn

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chapter(
        self, chapter_id: str, title: str, text: str, concepts: List[str]
    ) -> int:
        """Index a chapter's text into the vector store. Returns chunk count."""
        chunks = _chunk_text(text)
        if not chunks:
            return 0

        ids = [f"{chapter_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_id": chapter_id,
                "title": title,
                "chunk_index": i,
                "concepts": ", ".join(concepts),
            }
            for i in range(len(chunks))
        ]

        existing_ids = set()
        try:
            existing = self.collection.get(
                where={"chapter_id": chapter_id},
                include=[],
            )
            if existing and existing["ids"]:
                existing_ids = set(existing["ids"])
                self.collection.delete(ids=list(existing_ids))
        except Exception:
            pass

        batch_size = 100
        total = 0
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self.collection.add(
                ids=ids[start:end],
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
            )
            total += end - start

        logger.info("Indexed %d chunks for chapter %s", total, chapter_id)
        return total

    def index_all_chapters(self, db: MoEDatabase) -> Dict[str, int]:
        """Index all chapters from the database into the vector store."""
        chapters = db.list_chapters()
        results = {}
        for chapter in chapters:
            count = self.index_chapter(
                chapter_id=chapter["chapter_id"],
                title=chapter["title"],
                text=chapter["markdown"],
                concepts=chapter.get("concepts", []),
            )
            results[chapter["chapter_id"]] = count
        return results

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
        where = {"chapter_id": chapter_filter} if chapter_filter else None
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search error: %s", e)
            return []

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "relevance": 1.0 - results["distances"][0][i],
                    }
                )
        return hits

    def infer_experts_for_competition(
        self,
        competition_description: str,
        db: MoEDatabase,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Use RAG to infer which experts and skills are relevant for a competition."""
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for hit in hits:
            chapter_id = hit["metadata"]["chapter_id"]
            relevance = hit["relevance"]
            if chapter_id in chapter_scores:
                chapter_scores[chapter_id] = max(chapter_scores[chapter_id], relevance)
            else:
                chapter_scores[chapter_id] = relevance

        ranked_experts = []
        for chapter_id, score in sorted(
            chapter_scores.items(), key=lambda x: x[1], reverse=True
        ):
            expert = db.get_expert_by_slug(chapter_id)
            if not expert:
                experts_list = db.list_experts()
                expert = next(
                    (e for e in experts_list if e.get("chapter_id") == chapter_id),
                    None,
                )
            if expert:
                ranked_experts.append(
                    {
                        "expert_name": expert["expert_name"],
                        "slug": expert["slug"],
                        "relevance_score": round(score, 4),
                        "skills": expert.get("skills", []),
                        "strategy": expert.get("strategy", ""),
                    }
                )

        return ranked_experts

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding store statistics."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {"error": str(e)}
