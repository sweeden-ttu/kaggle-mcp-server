"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import MLSysEngDatabase

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


class EmbeddingEngine:
    """Manages embedding generation and semantic search via ChromaDB."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: str = EMBEDDING_MODEL):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        return self._embedding_fn

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chapters(self, db: MLSysEngDatabase, force: bool = False) -> Dict[str, Any]:
        """Index all chapters from the database into ChromaDB."""
        chapters = db.list_chapters()
        if not chapters:
            return {"status": "no_chapters", "indexed": 0}

        if force:
            try:
                self.client.delete_collection(COLLECTION_NAME)
                self._collection = None
            except Exception:
                pass

        total_chunks = 0
        for chapter in chapters:
            chunks = _chunk_text(chapter.content_md)
            ids = [f"{chapter.chapter_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "chapter_id": chapter.chapter_id,
                    "title": chapter.title,
                    "concepts": ",".join(chapter.concepts[:20]),
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]

            existing_ids = set()
            if not force:
                try:
                    result = self.collection.get(ids=ids)
                    existing_ids = set(result["ids"])
                except Exception:
                    pass

            new_ids = [id_ for id_ in ids if id_ not in existing_ids]
            if not new_ids:
                continue

            new_indices = [ids.index(id_) for id_ in new_ids]
            self.collection.add(
                ids=new_ids,
                documents=[chunks[i] for i in new_indices],
                metadatas=[metadatas[i] for i in new_indices],
            )
            total_chunks += len(new_ids)

        return {
            "status": "indexed",
            "chapters_processed": len(chapters),
            "chunks_added": total_chunks,
            "total_chunks_in_collection": self.collection.count(),
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        where = None
        if chapter_filter:
            where = {"chapter_id": chapter_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1.0 - results["distances"][0][i],
                }
            )
        return matches

    def infer_skills(
        self,
        competition_description: str,
        db: MLSysEngDatabase,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Given a competition description, infer which experts and skills are relevant."""
        matches = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for m in matches:
            ch_id = m["metadata"]["chapter_id"]
            sim = m["similarity"]
            chapter_scores[ch_id] = max(chapter_scores.get(ch_id, 0), sim)

        experts = db.list_experts()
        expert_map = {e.chapter_id: e for e in experts}

        recommendations = []
        for ch_id, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
            expert = expert_map.get(ch_id)
            if expert:
                recommendations.append(
                    {
                        "expert_name": expert.expert_name,
                        "slug": expert.slug,
                        "relevance_score": round(score, 4),
                        "capabilities": expert.capabilities,
                        "skills": expert.skills,
                        "strategy": expert.strategy,
                    }
                )

        return recommendations

    def get_rdagent_context(
        self,
        competition_name: str,
        description: str = "",
        n_results: int = 8,
    ) -> str:
        """Generate a context prompt for rdagent from the knowledge base."""
        query = f"{competition_name} {description}".strip()
        matches = self.search(query, n_results=n_results)

        if not matches:
            return f"No relevant ML principles found for '{competition_name}'."

        context_parts = [
            f"# ML Principles Context for: {competition_name}",
            "",
            "Based on the ML Principles knowledge base, here are the most relevant concepts:",
            "",
        ]

        for i, m in enumerate(matches, 1):
            chapter = m["metadata"].get("title", "Unknown")
            concepts = m["metadata"].get("concepts", "")
            context_parts.append(f"## {i}. From: {chapter}")
            if concepts:
                context_parts.append(f"**Key concepts**: {concepts}")
            context_parts.append("")
            text_preview = m["text"][:300]
            context_parts.append(text_preview)
            context_parts.append("")

        return "\n".join(context_parts)

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": count,
            "embedding_model": self.model_name,
            "chroma_path": self.chroma_path,
        }
