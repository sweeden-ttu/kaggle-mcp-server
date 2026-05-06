"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from mlsyseng_mcp.database import Database, ChapterRecord

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingEngine:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(
        self,
        db: Database,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.db = db
        self.chroma_path = Path(chroma_path).expanduser()
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._chroma_client = None
        self._collection = None

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

                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.chroma_path)
                )
                self._collection = self._chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise RuntimeError(
                    "chromadb is required. Install with: pip install chromadb"
                )
        return self._collection

    def index_chapter(self, chapter: ChapterRecord) -> int:
        """Generate embeddings for a chapter and store in ChromaDB."""
        model = self._get_model()
        collection = self._get_collection()

        chunks = _chunk_text(chapter.content_md)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chapter.chapter_id}_chunk_{i}_{_content_hash(chunk)}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "chapter_id": chapter.chapter_id,
                    "chapter_name": chapter.chapter_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "concepts": ", ".join(chapter.concepts[:20]),
                }
            )

        existing = collection.get(
            where={"chapter_id": chapter.chapter_id}, include=[]
        )
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])

        embeddings = model.encode(documents, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d chunks for chapter %s", len(chunks), chapter.chapter_id
        )
        return len(chunks)

    def index_all_chapters(self) -> dict:
        """Index all chapters from the database."""
        chapters = self.db.list_chapters()
        total_chunks = 0
        indexed = 0
        for chapter in chapters:
            n = self.index_chapter(chapter)
            total_chunks += n
            indexed += 1
        return {
            "chapters_indexed": indexed,
            "total_chunks": total_chunks,
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search over indexed ML Principles content."""
        model = self._get_model()
        collection = self._get_collection()

        query_embedding = model.encode([query]).tolist()

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_id": chapter_filter}

        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "relevance": 1.0 - results["distances"][0][i],
                    }
                )
        return output

    def get_relevant_experts(
        self,
        query: str,
        n_results: int = 10,
    ) -> list[str]:
        """Find which chapter experts are most relevant to a query."""
        results = self.search(query, n_results=n_results)
        chapter_scores: dict[str, float] = {}
        for r in results:
            ch_id = r["metadata"]["chapter_id"]
            score = r["relevance"]
            if ch_id in chapter_scores:
                chapter_scores[ch_id] = max(chapter_scores[ch_id], score)
            else:
                chapter_scores[ch_id] = score

        sorted_chapters = sorted(
            chapter_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [ch_id for ch_id, _ in sorted_chapters]

    def get_stats(self) -> dict:
        """Get embedding index statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
        except Exception:
            count = 0
        return {
            "total_vectors": count,
            "model": self.model_name,
            "chroma_path": str(self.chroma_path),
            "collection": COLLECTION_NAME,
        }
