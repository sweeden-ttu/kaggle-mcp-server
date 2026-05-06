"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB.

Provides semantic search over extracted ML Principles chapter content.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingStore:
    """ChromaDB-backed vector store for ML chapter content."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: str = EMBEDDING_MODEL):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        return self._embedding_fn

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
        return self._collection

    def index_chapter(
        self,
        chapter_id: str,
        title: str,
        content: str,
        concepts: Optional[List[str]] = None,
        force: bool = False,
    ) -> int:
        """Index a chapter's content into ChromaDB.

        Returns the number of chunks indexed.
        """
        if not content.strip():
            logger.warning("Empty content for chapter %s, skipping", chapter_id)
            return 0

        if not force:
            existing = self.collection.get(
                where={"chapter_id": chapter_id},
                limit=1,
            )
            if existing and existing["ids"]:
                logger.info("Chapter %s already indexed, skipping", chapter_id)
                return 0

        if force:
            try:
                existing = self.collection.get(
                    where={"chapter_id": chapter_id},
                )
                if existing and existing["ids"]:
                    self.collection.delete(ids=existing["ids"])
            except Exception:
                pass

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chapter_id}_chunk_{i:04d}_{_content_hash(chunk)}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_id": chapter_id,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "concepts": json.dumps(concepts or []),
            })

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Indexed %d chunks for chapter %s", len(chunks), chapter_id)
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters.

        Returns list of dicts with document, metadata, and distance.
        """
        where_filter = None
        if chapter_filter:
            where_filter = {"chapter_id": chapter_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                hits.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "chapter_id": meta.get("chapter_id", ""),
                    "title": meta.get("title", ""),
                })

        return hits

    def search_concepts(
        self,
        query: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for concepts across all chapters."""
        return self.search(query, n_results=n_results)

    def get_rdagent_context(
        self,
        competition_name: str,
        description: str = "",
        n_results: int = 8,
    ) -> str:
        """Generate a context prompt for rdagent from relevant ML principles.

        Searches the knowledge base using the competition description and
        assembles a structured context string.
        """
        search_query = f"{competition_name} {description}".strip()
        hits = self.search(search_query, n_results=n_results)

        if not hits:
            return f"No ML Principles context found for '{competition_name}'."

        sections = [
            f"# ML Principles Context for: {competition_name}",
            "",
        ]

        seen_chapters = set()
        for hit in hits:
            ch_id = hit.get("chapter_id", "unknown")
            title = hit.get("title", "Unknown")
            if ch_id not in seen_chapters:
                sections.append(f"## {title}")
                seen_chapters.add(ch_id)
            sections.append(hit["document"])
            sections.append("")

        return "\n".join(sections)

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        return {
            "collection": COLLECTION_NAME,
            "total_chunks": count,
            "chroma_path": self.chroma_path,
            "model": self.model_name,
        }
