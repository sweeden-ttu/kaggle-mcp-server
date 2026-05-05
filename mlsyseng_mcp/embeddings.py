"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
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
        start += chunk_size - overlap
    return chunks


class EmbeddingEngine:
    """Manages embedding generation and ChromaDB vector storage."""

    def __init__(self, chroma_path: str = CHROMA_DB_PATH, model_name: str = MODEL_NAME):
        self._chroma_path = chroma_path
        self._model_name = model_name
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                os.makedirs(self._chroma_path, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self._chroma_path)
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.error(
                    "chromadb not installed. Install with: pip install chromadb"
                )
                raise
        return self._collection

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        return self.model.encode(texts).tolist()

    def index_chapter(
        self,
        chapter_id: str,
        title: str,
        content: str,
        db=None,
    ) -> int:
        """Chunk and index a chapter's content. Returns number of chunks indexed."""
        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        batch_embeddings = self.embed_batch(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, batch_embeddings)):
            emb_id = f"{chapter_id}_chunk_{i}"
            ids.append(emb_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_id": chapter_id,
                "chapter_title": title,
                "chunk_index": i,
            })
            embeddings.append(emb)

            if db is not None:
                db.add_embedding_meta(emb_id, chapter_id, i, chunk)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info("Indexed %d chunks for chapter %s", len(chunks), title)
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content.

        Returns list of dicts with keys: text, chapter_id, chapter_title, score, chunk_index.
        """
        query_embedding = self.embed_text(query)

        where_filter = None
        if chapter_id:
            where_filter = {"chapter_id": chapter_id}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

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
                    "chapter_title": meta.get("chapter_title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": 1.0 - dist,
                })
        return hits

    def infer_relevant_experts(
        self,
        query: str,
        expert_list: List[Dict[str, Any]],
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Given a query (e.g. competition description), find the most relevant experts.

        Searches the vector store for relevant chapters, then maps back to experts.
        """
        hits = self.search(query, n_results=n_results)
        chapter_scores: Dict[str, float] = {}
        for hit in hits:
            cid = hit["chapter_id"]
            chapter_scores[cid] = max(chapter_scores.get(cid, 0), hit["score"])

        ranked = []
        for expert in expert_list:
            cid = expert.get("chapter_id", "")
            score = chapter_scores.get(cid, 0.0)
            if score > 0:
                ranked.append({**expert, "relevance_score": score})

        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
