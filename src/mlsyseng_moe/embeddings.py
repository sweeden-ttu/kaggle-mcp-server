"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB.

Provides semantic search over extracted chapter content to support
skill inference and expert selection.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _default_chroma_path() -> str:
    env = os.environ.get("CHROMA_DB_PATH")
    if env:
        return env
    return str(Path.home() / ".openclaw" / "workspace" / "mlsyseng" / "chroma_db")


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
    """Manages vector embeddings with ChromaDB and sentence-transformers."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
    ):
        self.chroma_path = chroma_path or _default_chroma_path()
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError("chromadb is required: pip install chromadb")

            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="ml_principles",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chapter(self, chapter_id: str, title: str, content: str, concepts: List[str]) -> int:
        """Chunk and index a chapter's content. Returns the number of chunks added."""
        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{chapter_id}:{i}".encode()).hexdigest()
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_id": chapter_id,
                "title": title,
                "chunk_index": i,
                "concepts": ",".join(concepts),
            })

        embeddings = self.embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def index_all_chapters(self, db: Database) -> Dict[str, int]:
        """Index all chapters from the database into ChromaDB."""
        chapters = db.list_chapters()
        results = {}
        for ch in chapters:
            content = ch.get("content_md", "")
            if not content:
                continue
            n = self.index_chapter(
                chapter_id=ch["chapter_id"],
                title=ch["title"],
                content=content,
                concepts=ch.get("concepts", []),
            )
            results[ch["chapter_id"]] = n
        return results

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        query_embedding = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "text": doc,
                    "chapter_id": meta.get("chapter_id", ""),
                    "title": meta.get("title", ""),
                    "concepts": meta.get("concepts", "").split(",") if meta.get("concepts") else [],
                    "similarity": 1.0 - dist,
                })
        return hits

    def infer_skills(self, competition_description: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Given a competition description, infer relevant experts and skills."""
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        chapter_titles: Dict[str, str] = {}
        chapter_concepts: Dict[str, set] = {}

        for hit in hits:
            cid = hit["chapter_id"]
            score = hit["similarity"]
            chapter_scores[cid] = max(chapter_scores.get(cid, 0), score)
            chapter_titles[cid] = hit["title"]
            for c in hit["concepts"]:
                chapter_concepts.setdefault(cid, set()).add(c)

        ranked = sorted(chapter_scores.items(), key=lambda x: -x[1])
        return [
            {
                "chapter_id": cid,
                "title": chapter_titles.get(cid, ""),
                "relevance": round(score, 4),
                "concepts": sorted(chapter_concepts.get(cid, set())),
            }
            for cid, score in ranked
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            count = self.collection.count()
            return {"total_chunks": count, "chroma_path": self.chroma_path}
        except Exception:
            return {"total_chunks": 0, "chroma_path": self.chroma_path}
