"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import MoEDatabase

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_knowledge"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_len = len(sentence.split())
        if current_len + s_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = []
            overlap_len = 0
            for s in reversed(current_chunk):
                words = s.split()
                if overlap_len + len(words) > overlap:
                    break
                overlap_words.insert(0, s)
                overlap_len += len(words)
            current_chunk = overlap_words
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += s_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _chunk_id(chapter_id: str, chunk_index: int) -> str:
    return f"{chapter_id}_chunk_{chunk_index:04d}"


class EmbeddingEngine:
    """Manages embeddings and ChromaDB vector store for RAG retrieval."""

    def __init__(self, chroma_path: str = DEFAULT_CHROMA_PATH, db: Optional[MoEDatabase] = None):
        self.chroma_path = chroma_path
        self.db = db
        self._collection = None
        self._client = None
        self._model = None

    def _ensure_chroma(self):
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required: pip install chromadb"
            )

        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def index_chapter(self, chapter_id: str, content: str, title: str = "") -> int:
        """Chunk and embed a chapter's content, storing in ChromaDB."""
        self._ensure_chroma()

        if not content.strip():
            return 0

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = [_chunk_id(chapter_id, i) for i in range(len(chunks))]
        metadatas = [{"chapter_id": chapter_id, "title": title, "chunk_index": i} for i in range(len(chunks))]
        embeddings = self._embed(chunks)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        if self.db:
            for cid, chunk in zip(ids, chunks):
                self.db.save_chunk_meta(cid, chapter_id, chunk[:200])

        return len(chunks)

    def index_all_chapters(self, db: MoEDatabase) -> Dict[str, int]:
        """Index all extracted chapters from the database."""
        chapters = db.list_chapters(status="done")
        results = {}
        for ch in chapters:
            n = self.index_chapter(ch.chapter_id, ch.content_md, ch.title)
            results[ch.chapter_id] = n
            logger.info(f"Indexed {n} chunks for chapter {ch.chapter_id}")
        return results

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed knowledge base."""
        self._ensure_chroma()

        query_embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "chunk_id": doc_id,
                    "chapter_id": results["metadatas"][0][i].get("chapter_id", ""),
                    "title": results["metadatas"][0][i].get("title", ""),
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": 1.0 - results["distances"][0][i],
                })
        return hits

    def get_rdagent_context(self, competition_name: str, description: str = "", n_results: int = 10) -> str:
        """Generate a context prompt for rdagent from the knowledge base."""
        query = f"{competition_name} {description}".strip()
        if not query:
            return "No query provided for context generation."

        hits = self.search(query, n_results=n_results)
        if not hits:
            return f"No relevant ML principles found for '{competition_name}'."

        context_parts = [
            f"# ML Principles Context for: {competition_name}",
            "",
        ]
        if description:
            context_parts.append(f"**Competition Description**: {description}")
            context_parts.append("")

        context_parts.append("## Relevant Knowledge Chunks")
        context_parts.append("")

        for i, hit in enumerate(hits, 1):
            context_parts.append(f"### {i}. {hit['title']} (relevance: {hit['relevance']:.3f})")
            context_parts.append(hit["text"])
            context_parts.append("")

        return "\n".join(context_parts)

    def infer_experts_for_competition(
        self, competition_name: str, description: str, db: MoEDatabase, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Given a competition, find the most relevant experts using RAG."""
        query = f"{competition_name} {description}".strip()
        hits = self.search(query, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for hit in hits:
            ch_id = hit["chapter_id"]
            chapter_scores[ch_id] = chapter_scores.get(ch_id, 0.0) + hit["relevance"]

        ranked_experts = []
        for ch_id, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
            experts = [e for e in db.list_experts() if e.chapter_id == ch_id]
            for expert in experts:
                ranked_experts.append({
                    "expert": expert.to_dict(),
                    "relevance_score": round(score, 4),
                })

        return ranked_experts

    def get_collection_count(self) -> int:
        """Return total number of stored embeddings."""
        self._ensure_chroma()
        return self._collection.count()
