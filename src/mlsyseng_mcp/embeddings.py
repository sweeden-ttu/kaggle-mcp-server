"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


def get_chroma_path() -> str:
    return os.environ.get("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH)


class EmbeddingStore:
    """Manages embeddings and similarity search over ML Principles content."""

    def __init__(self, chroma_path: Optional[str] = None, collection_name: str = COLLECTION_NAME):
        self._chroma_path = chroma_path or get_chroma_path()
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self._chroma_path, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()

    def index_chapter(self, chapter_number: int, title: str, content: str,
                      concepts: list[dict], chunk_size: int = 500) -> int:
        """Index a chapter's content into ChromaDB with chunking."""
        chunks = self._chunk_content(content, chunk_size)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"ch{chapter_number:02d}_chunk_{i:04d}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
                "type": "content",
            })

        for concept in concepts:
            doc_id = f"ch{chapter_number:02d}_concept_{concept['name'][:30].replace(' ', '_').lower()}"
            ids.append(doc_id)
            documents.append(f"{concept['name']}: {concept.get('description', '')}")
            metadatas.append({
                "chapter_number": chapter_number,
                "title": title,
                "type": "concept",
                "concept_name": concept["name"],
                "category": concept.get("category", "general"),
            })

        embeddings = self.embed_batch(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(ids)

    def search(self, query: str, n_results: int = 5,
               filter_chapter: Optional[int] = None) -> list[dict]:
        """Semantic search over indexed content."""
        query_embedding = self.embed_text(query)

        where_filter = None
        if filter_chapter is not None:
            where_filter = {"chapter_number": filter_chapter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                search_results.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": 1.0 - dist,
                })

        return search_results

    def get_relevant_experts(self, competition_description: str,
                             n_results: int = 3) -> list[dict]:
        """Find the most relevant chapter experts for a competition."""
        results = self.search(competition_description, n_results=n_results * 2)

        chapter_scores: dict[int, float] = {}
        for r in results:
            ch = r["metadata"].get("chapter_number")
            if ch is not None:
                score = r["similarity"]
                chapter_scores[ch] = max(chapter_scores.get(ch, 0), score)

        sorted_chapters = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chapter_number": ch, "relevance_score": score}
                for ch, score in sorted_chapters[:n_results]]

    def get_stats(self) -> dict:
        """Get embedding store statistics."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self._collection_name,
            "chroma_path": self._chroma_path,
            "embedding_model": EMBEDDING_MODEL,
        }

    def _chunk_content(self, content: str, chunk_size: int = 500) -> list[str]:
        """Split content into overlapping chunks by word count."""
        words = content.split()
        if not words:
            return []

        chunks = []
        overlap = chunk_size // 5
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap

        return chunks
