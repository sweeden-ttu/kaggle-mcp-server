"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB.

Provides semantic search over extracted ML Principles chapter content.
"""

import json
import os
from typing import Optional

from .database import MLSysEngDB

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word boundaries."""
    words = text.split()
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
    """ChromaDB-backed vector store for ML Principles content."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or DEFAULT_CHROMA_PATH
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = None
        self._collection = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chapter(self, chapter_slug: str, title: str, content: str, concepts: list[str]):
        """Index a chapter's content into the vector store."""
        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = [f"{chapter_slug}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chapter_slug": chapter_slug,
                "title": title,
                "chunk_index": i,
                "concepts": json.dumps(concepts),
            }
            for i in range(len(chunks))
        ]

        embeddings = self.embed_texts(chunks)

        existing = set()
        try:
            result = self.collection.get(ids=ids)
            if result and result["ids"]:
                existing = set(result["ids"])
        except Exception:
            pass

        new_ids = []
        new_docs = []
        new_metas = []
        new_embeds = []
        for i, doc_id in enumerate(ids):
            if doc_id not in existing:
                new_ids.append(doc_id)
                new_docs.append(chunks[i])
                new_metas.append(metadatas[i])
                new_embeds.append(embeddings[i])

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_docs,
                metadatas=new_metas,
                embeddings=new_embeds,
            )

        return len(new_ids)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search over indexed content."""
        query_embedding = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1.0 - results["distances"][0][i],
                })
        return hits

    def get_stats(self) -> dict:
        """Return stats about the vector store."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "total_chunks": count,
            "persist_dir": self.persist_dir,
            "model": EMBEDDING_MODEL,
            "collection": COLLECTION_NAME,
        }


def index_all_chapters(db: MLSysEngDB, store: Optional[EmbeddingStore] = None) -> dict:
    """Index all extracted chapters into the vector store."""
    if store is None:
        store = EmbeddingStore()

    chapters = db.list_chapters()
    results = {"indexed": 0, "chunks_added": 0, "skipped": 0}

    for chapter in chapters:
        if chapter.status != "extracted" or not chapter.markdown_content:
            results["skipped"] += 1
            continue

        concepts = json.loads(chapter.concepts) if chapter.concepts else []
        added = store.index_chapter(
            chapter_slug=chapter.slug,
            title=chapter.title,
            content=chapter.markdown_content,
            concepts=concepts,
        )
        results["indexed"] += 1
        results["chunks_added"] += added

    return results


def search_concepts(
    query: str, n_results: int = 5, store: Optional[EmbeddingStore] = None
) -> list[dict]:
    """Search for ML concepts across all indexed chapters."""
    if store is None:
        store = EmbeddingStore()
    return store.search(query, n_results=n_results)
