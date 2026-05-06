"""Embedding generation and RAG retrieval using ChromaDB."""

import os
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingEngine:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(self, chroma_path: Optional[str] = None):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)

        self._model: Optional["SentenceTransformer"] = None
        self._client: Optional["chromadb.ClientAPI"] = None
        self._collection = None

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    @property
    def client(self) -> "chromadb.ClientAPI":
        if self._client is None:
            if not HAS_CHROMADB:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
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

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        return self.model.encode(texts).tolist()

    def add_chapter_content(
        self,
        chapter_id: int,
        chapter_number: int,
        title: str,
        content: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> int:
        """Chunk and embed chapter content, storing in ChromaDB."""
        chunks = self._chunk_text(content, chunk_size, overlap)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"ch{chapter_number}_chunk{i}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_id": chapter_id,
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        embeddings = self.embed_batch(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[int] = None,
    ) -> list[dict]:
        """Semantic search over indexed content."""
        query_embedding = self.embed_text(query)

        where_filter = None
        if chapter_filter is not None:
            where_filter = {"chapter_number": chapter_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "relevance": 1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                })

        return search_results

    def get_relevant_experts(
        self,
        query: str,
        n_results: int = 3,
    ) -> list[int]:
        """Find the most relevant chapter numbers for a query."""
        results = self.search(query, n_results=n_results * 2)
        chapter_numbers = []
        seen = set()
        for r in results:
            ch_num = r["metadata"].get("chapter_number")
            if ch_num and ch_num not in seen:
                seen.add(ch_num)
                chapter_numbers.append(ch_num)
            if len(chapter_numbers) >= n_results:
                break
        return chapter_numbers

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "chroma_path": self.chroma_path,
        }

    def _chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 64
    ) -> list[str]:
        """Split text into overlapping chunks by word count."""
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
            start = end - overlap
            if start >= len(words):
                break

        return chunks

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self._collection = None
