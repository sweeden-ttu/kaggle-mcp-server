"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"


def _lazy_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        return None


def _lazy_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        return None


class EmbeddingStore:
    """Manages vector embeddings for ML Principles chapters using ChromaDB."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            self._model = _lazy_sentence_transformer(self.model_name)
        return self._model

    @property
    def client(self):
        if self._client is None:
            chromadb = _lazy_chromadb()
            if chromadb is None:
                raise ImportError("chromadb is required for embeddings. Install with: pip install chromadb")
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping word-level chunks."""
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
        chapter_num: int,
        title: str,
        content_md: str,
        concepts: Optional[List[str]] = None,
    ) -> int:
        """Index a chapter's content into ChromaDB. Returns number of chunks created."""
        if self.model is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.collection.delete(where={"chapter_num": chapter_num})

        chunks = self._chunk_text(content_md)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"ch{chapter_num}_chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_num": chapter_num,
                "title": title,
                "chunk_index": i,
                "concepts": json.dumps(concepts or []),
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def index_all(self, db: Database) -> Dict[str, Any]:
        """Index all chapters from the database."""
        chapters = db.list_chapters()
        results = {"total": len(chapters), "indexed": 0, "chunks": 0, "errors": []}

        for ch in chapters:
            if not ch.get("content_md"):
                continue
            try:
                concepts = json.loads(ch["concepts"]) if ch.get("concepts") else []
                n = self.index_chapter(
                    ch["chapter_num"], ch["title"], ch["content_md"], concepts
                )
                results["indexed"] += 1
                results["chunks"] += n
            except Exception as exc:
                results["errors"].append(
                    {"chapter": ch["chapter_num"], "error": str(exc)}
                )
        return results

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
        if self.model is None:
            raise ImportError("sentence-transformers is required for search")

        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return output

    def infer_relevant_experts(
        self, query: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Infer which experts are most relevant for a query based on embedding similarity."""
        results = self.search(query, n_results=n_results * 2)

        seen_chapters: Dict[int, Dict[str, Any]] = {}
        for r in results:
            ch_num = r["metadata"]["chapter_num"]
            if ch_num not in seen_chapters:
                concepts = r["metadata"].get("concepts", "[]")
                if isinstance(concepts, str):
                    concepts = json.loads(concepts)
                seen_chapters[ch_num] = {
                    "chapter_num": ch_num,
                    "title": r["metadata"]["title"],
                    "relevance_score": 1.0 - r["distance"],
                    "concepts": concepts,
                }
            else:
                existing = seen_chapters[ch_num]
                new_score = 1.0 - r["distance"]
                existing["relevance_score"] = max(existing["relevance_score"], new_score)

        ranked = sorted(
            seen_chapters.values(), key=lambda x: x["relevance_score"], reverse=True
        )
        return ranked[:n_results]

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding store statistics."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "collection": COLLECTION_NAME,
            "total_chunks": count,
            "model": self.model_name,
            "chroma_path": self.chroma_path,
        }
