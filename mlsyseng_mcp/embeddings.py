"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML Principles content for
skill inference and expert matching.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"


class EmbeddingStore:
    """Manages embeddings and vector search via ChromaDB."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: Optional[str] = None):
        self.chroma_path = chroma_path or os.environ.get("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH)
        self.model_name = model_name or EMBEDDING_MODEL
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
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
        self,
        chapter_number: str,
        title: str,
        content: str,
        concepts: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """Index a chapter's content into ChromaDB.

        Splits content into overlapping chunks for better retrieval.
        Returns number of chunks indexed.
        """
        if not content or not content.strip():
            return 0

        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._make_chunk_id(chapter_number, i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "concepts": ",".join(concepts[:20]) if concepts else "",
            })

        existing_ids = set()
        try:
            result = self.collection.get(ids=ids)
            if result and result["ids"]:
                existing_ids = set(result["ids"])
        except Exception:
            pass

        new_ids = [i for i in ids if i not in existing_ids]
        if not new_ids:
            return 0

        new_docs = [d for i, d in zip(ids, documents) if i not in existing_ids]
        new_meta = [m for i, m in zip(ids, metadatas) if i not in existing_ids]

        batch_size = 100
        indexed = 0
        for start in range(0, len(new_ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=new_ids[start:end],
                documents=new_docs[start:end],
                metadatas=new_meta[start:end],
            )
            indexed += len(new_ids[start:end])

        return indexed

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content.

        Returns list of results with document text, metadata, and distance.
        """
        where = None
        if chapter_filter:
            where = {"chapter_number": chapter_filter}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        items = []
        if not results or not results["ids"] or not results["ids"][0]:
            return items

        for i, doc_id in enumerate(results["ids"][0]):
            items.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance": 1.0 - results["distances"][0][i],
            })

        return items

    def infer_experts(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Infer which experts are relevant for a competition.

        Returns ranked list of chapter matches with relevance scores.
        """
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, Dict[str, Any]] = {}
        for r in results:
            ch_num = r["metadata"]["chapter_number"]
            if ch_num not in chapter_scores:
                chapter_scores[ch_num] = {
                    "chapter_number": ch_num,
                    "title": r["metadata"]["title"],
                    "max_relevance": r["relevance"],
                    "total_relevance": r["relevance"],
                    "match_count": 1,
                    "concepts": r["metadata"].get("concepts", "").split(","),
                }
            else:
                entry = chapter_scores[ch_num]
                entry["max_relevance"] = max(entry["max_relevance"], r["relevance"])
                entry["total_relevance"] += r["relevance"]
                entry["match_count"] += 1

        ranked = sorted(
            chapter_scores.values(),
            key=lambda x: x["total_relevance"],
            reverse=True,
        )
        return ranked

    def index_all_chapters(self, db: Database) -> Dict[str, int]:
        """Index all extracted chapters from the database."""
        chapters = db.get_all_chapters()
        stats = {"indexed": 0, "chunks": 0, "skipped": 0}

        for ch in chapters:
            if not ch.get("content_markdown"):
                stats["skipped"] += 1
                continue

            n_chunks = self.index_chapter(
                chapter_number=ch["chapter_number"],
                title=ch["title"],
                content=ch["content_markdown"],
                concepts=ch.get("concepts"),
            )

            if n_chunks > 0:
                db.update_chapter_embedding(
                    ch["chapter_number"],
                    self._make_chunk_id(ch["chapter_number"], 0),
                )
                stats["indexed"] += 1
                stats["chunks"] += n_chunks
            else:
                stats["skipped"] += 1

        return stats

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks at sentence boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for sep in ("\n\n", "\n", ". ", " "):
                    boundary = text.rfind(sep, start + chunk_size // 2, end)
                    if boundary != -1:
                        end = boundary + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]

    def _make_chunk_id(self, chapter_number: str, chunk_index: int) -> str:
        raw = f"{chapter_number}_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
