"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _get_chroma_client(path: Optional[str] = None):
    """Get or create a ChromaDB persistent client."""
    import chromadb

    persist_dir = path or DEFAULT_CHROMA_PATH
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _get_embedding_function(model_name: str = DEFAULT_MODEL_NAME):
    """Get the embedding function for ChromaDB."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


class EmbeddingStore:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_chroma_client(self.chroma_path)
        return self._client

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            self._embedding_fn = _get_embedding_function(self.model_name)
        return self._embedding_fn

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
        return self._collection

    def index_chapter(self, chapter_name: str, content: str, chapter_id: int) -> int:
        """Index a chapter's content into the vector store."""
        existing = self.collection.get(where={"chapter_name": chapter_name})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{chapter_name}_{_content_hash(chunk)}_{i}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_name": chapter_name,
                "chapter_id": chapter_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return len(chunks)

    def index_all_chapters(self, db: Database) -> Dict[str, int]:
        """Index all chapters from the database into the vector store."""
        chapters = db.get_chapters()
        results = {}

        for ch in chapters:
            content = db.get_chapter_content(ch["id"])
            if content:
                count = self.index_chapter(ch["chapter_name"], content, ch["id"])
                results[ch["chapter_name"]] = count
                logger.info(f"Indexed {count} chunks for {ch['chapter_name']}")

        return results

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        where = {"chapter_name": chapter_filter} if chapter_filter else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                hit = {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "id": results["ids"][0][i] if results["ids"] else None,
                }
                hits.append(hit)

        return hits

    def infer_relevant_experts(
        self,
        competition_description: str,
        db: Database,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Infer which experts are relevant for a competition based on RAG search."""
        hits = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for hit in hits:
            ch_name = hit["metadata"].get("chapter_name", "")
            distance = hit.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            chapter_scores[ch_name] = max(
                chapter_scores.get(ch_name, 0.0), similarity
            )

        experts = db.get_experts()
        relevant = []
        for expert in experts:
            slug = expert.get("slug", "")
            name = expert.get("expert_name", "")
            score = 0.0
            for ch_name, ch_score in chapter_scores.items():
                if slug in ch_name.lower().replace(" ", "_") or ch_name.lower() in name.lower():
                    score = max(score, ch_score)
            if score > 0:
                expert["relevance_score"] = score
                relevant.append(expert)

        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding store statistics."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "total_chunks": count,
            "chroma_path": self.chroma_path,
            "model": self.model_name,
            "collection": COLLECTION_NAME,
        }
