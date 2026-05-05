"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word boundaries."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class EmbeddingStore:
    """Manages embeddings and semantic search via ChromaDB."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: str = EMBEDDING_MODEL):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_embedding_function(self):
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions

            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        return self._embedding_fn

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chapters(self, db: Database, force: bool = False) -> Dict[str, Any]:
        """Index all chapter content from the database into ChromaDB."""
        collection = self._get_collection()
        chapters = db.get_all_chapter_content()

        if not chapters:
            return {"status": "no_chapters", "message": "No chapters to index"}

        if force:
            client = self._get_client()
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = None
            collection = self._get_collection()

        total_chunks = 0
        for chapter in chapters:
            content = chapter.get("markdown_content", "")
            if not content:
                continue

            chunks = _chunk_text(content)
            if not chunks:
                continue

            chapter_num = chapter["chapter_num"]
            title = chapter["title"]
            concepts = chapter.get("concepts", [])

            ids = [f"ch{chapter_num}_chunk{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "chapter_num": chapter_num,
                    "title": title,
                    "chunk_index": i,
                    "concepts": ", ".join(concepts[:10]),
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

            new_ids = []
            new_docs = []
            new_metas = []
            for doc_id, doc, meta in zip(ids, chunks, metadatas):
                if doc_id not in existing or force:
                    new_ids.append(doc_id)
                    new_docs.append(doc)
                    new_metas.append(meta)

            if new_ids:
                collection.upsert(ids=new_ids, documents=new_docs, metadatas=new_metas)
                total_chunks += len(new_ids)

        return {
            "status": "success",
            "chunks_indexed": total_chunks,
            "chapters_processed": len(chapters),
        }

    def search(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        collection = self._get_collection()

        where = None
        if chapter_filter is not None:
            where = {"chapter_num": chapter_filter}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results.get("distances") else None
                items.append({
                    "content": doc,
                    "chapter_num": meta.get("chapter_num"),
                    "title": meta.get("title"),
                    "chunk_index": meta.get("chunk_index"),
                    "concepts": meta.get("concepts", ""),
                    "similarity": 1.0 - distance if distance is not None else None,
                })
        return items

    def get_stats(self) -> Dict[str, Any]:
        """Return embedding store statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "collection": COLLECTION_NAME,
                "total_chunks": count,
                "model": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            return {"error": str(e)}
