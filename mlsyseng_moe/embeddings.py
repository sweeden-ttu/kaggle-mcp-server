"""Embedding generation and RAG retrieval for MLSysEng MoE.

Uses sentence-transformers for embedding and ChromaDB for vector storage.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingEngine:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _ensure_initialized(self):
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for embeddings. Install with: pip install chromadb"
            )

        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.chroma_path)

        try:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        except Exception:
            logger.warning(
                "sentence-transformers not available; using chromadb default embeddings"
            )
            self._embedding_fn = None

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )

    def index_chapter(self, chapter_id: int, title: str, content: str, concepts: list[str]):
        """Index a chapter's content and concepts into ChromaDB."""
        self._ensure_initialized()

        chunks = self._chunk_text(content, chunk_size=500, overlap=50)
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"ch{chapter_id}_chunk{i}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_id": str(chapter_id),
                "title": title,
                "chunk_index": i,
                "type": "content",
            })

        for j, concept in enumerate(concepts):
            doc_id = f"ch{chapter_id}_concept{j}"
            ids.append(doc_id)
            documents.append(f"ML Concept: {concept} (from {title})")
            metadatas.append({
                "chapter_id": str(chapter_id),
                "title": title,
                "concept": concept,
                "type": "concept",
            })

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            f"Indexed chapter '{title}': {len(chunks)} chunks, {len(concepts)} concepts"
        )

    def search(self, query: str, n_results: int = 5, filter_type: Optional[str] = None) -> list[dict]:
        """Semantic search over indexed content."""
        self._ensure_initialized()

        where = {"type": filter_type} if filter_type else None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                item = {
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "id": results["ids"][0][i] if results["ids"] else None,
                }
                items.append(item)
        return items

    def search_concepts(self, query: str, n_results: int = 10) -> list[dict]:
        """Search specifically over concepts."""
        return self.search(query, n_results=n_results, filter_type="concept")

    def get_relevant_experts(self, query: str, db: Database, n_results: int = 5) -> list[dict]:
        """Find relevant experts for a query using semantic search."""
        search_results = self.search(query, n_results=n_results)

        chapter_ids = set()
        for result in search_results:
            ch_id = result["metadata"].get("chapter_id")
            if ch_id:
                chapter_ids.add(int(ch_id))

        experts = db.list_experts()
        relevant = []
        for expert in experts:
            if expert.chapter_id in chapter_ids:
                relevant.append({
                    "expert": expert.to_dict(),
                    "relevance": "direct_chapter_match",
                })

        if not relevant:
            for expert in experts:
                expert_text = " ".join(expert.capabilities_list + [expert.expert_name])
                if any(
                    term.lower() in expert_text.lower()
                    for term in query.split()
                    if len(term) > 3
                ):
                    relevant.append({
                        "expert": expert.to_dict(),
                        "relevance": "keyword_match",
                    })

        return relevant

    def index_all_chapters(self, db: Database):
        """Index all extracted chapters from the database."""
        chapters = db.list_chapters(status="extracted")
        for chapter in chapters:
            self.index_chapter(
                chapter_id=chapter.id,
                title=chapter.title,
                content=chapter.content_md,
                concepts=chapter.concept_list,
            )
        logger.info(f"Indexed {len(chapters)} chapters")

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks
