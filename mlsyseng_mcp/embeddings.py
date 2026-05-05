"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import hashlib
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
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word boundaries."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class EmbeddingStore:
    """Vector store for ML Principles content using ChromaDB and sentence-transformers."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: str = EMBEDDING_MODEL):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embed_fn = None

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.chroma_path)
            except ImportError:
                raise ImportError(
                    "chromadb is required for RAG. Install with: pip install chromadb"
                )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedding_fn(self):
        if self._embed_fn is None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name)
                self._embed_fn = model.encode
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: pip install sentence-transformers"
                )
        return self._embed_fn

    def _embed(self, texts: List[str]) -> List[List[float]]:
        fn = self._get_embedding_fn()
        embeddings = fn(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def index_chapter(
        self,
        chapter_name: str,
        content: str,
        concepts: List[str],
    ) -> int:
        """Index a chapter's content as chunked embeddings."""
        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{chapter_name}:{i}".encode()).hexdigest()
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter": chapter_name,
                "chunk_index": i,
                "concepts": ", ".join(concepts[:20]),
            })

        embeddings = self._embed(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Indexed %d chunks for chapter %s", len(chunks), chapter_name)
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed knowledge base."""
        query_embedding = self._embed([query])[0]

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter": chapter_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "content": doc,
                    "chapter": meta.get("chapter", ""),
                    "concepts": meta.get("concepts", ""),
                    "similarity": 1.0 - dist,
                })

        return hits

    def infer_skills(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are needed for a competition."""
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        chapter_concepts: Dict[str, List[str]] = {}

        for hit in hits:
            chapter = hit["chapter"]
            score = hit["similarity"]
            if chapter not in chapter_scores or score > chapter_scores[chapter]:
                chapter_scores[chapter] = score
            concepts = hit.get("concepts", "")
            if concepts:
                chapter_concepts.setdefault(chapter, []).extend(
                    c.strip() for c in concepts.split(",")
                )

        recommendations = []
        for chapter, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
            unique_concepts = sorted(set(chapter_concepts.get(chapter, [])))
            recommendations.append({
                "chapter": chapter,
                "relevance_score": round(score, 4),
                "relevant_concepts": unique_concepts[:10],
            })

        return recommendations

    def index_all_chapters(self, db: Database) -> Dict[str, int]:
        """Index all extracted chapters from the database."""
        chapters = db.list_chapters(status="extracted")
        stats = {"indexed": 0, "chunks": 0, "skipped": 0}

        for ch in chapters:
            content = ch.get("markdown_content", "")
            if not content:
                stats["skipped"] += 1
                continue
            count = self.index_chapter(
                chapter_name=ch["chapter_name"],
                content=content,
                concepts=ch.get("concepts", []),
            )
            stats["indexed"] += 1
            stats["chunks"] += count

        return stats
