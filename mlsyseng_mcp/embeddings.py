"""Embedding generation and RAG retrieval using ChromaDB."""

import os
import logging
from typing import Optional

from mlsyseng_mcp.database import get_all_chapters, get_all_experts

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingStore:
    """Manages embeddings and semantic search over ML Principles content."""

    def __init__(self, chroma_path: Optional[str] = None, db_path: Optional[str] = None):
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.db_path = db_path
        self._client = None
        self._collection = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.chroma_path, exist_ok=True)
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

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def index_chapters(self, db_path: Optional[str] = None) -> dict:
        """Index all chapters from database into ChromaDB."""
        chapters = get_all_chapters(db_path or self.db_path)

        if not chapters:
            return {"status": "no_chapters", "message": "No chapters to index"}

        documents = []
        metadatas = []
        ids = []

        for chapter in chapters:
            content = chapter.get("content_md", "")
            if not content:
                continue

            chunks = self._chunk_text(content, chunk_size=512, overlap=64)
            for i, chunk in enumerate(chunks):
                doc_id = f"ch{chapter['chapter_number']}_chunk{i}"
                documents.append(chunk)
                metadatas.append({
                    "chapter_number": chapter["chapter_number"],
                    "title": chapter["title"],
                    "chunk_index": i,
                    "concepts": ", ".join(chapter.get("concepts", [])[:10]),
                })
                ids.append(doc_id)

        if not documents:
            return {"status": "no_content", "message": "No content to embed"}

        embeddings = self.embed_texts(documents)

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        return {
            "status": "completed",
            "chunks_indexed": len(documents),
            "chapters_processed": len(chapters),
        }

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search over indexed content."""
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "relevance": 1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                })

        return search_results

    def infer_skills(self, competition_description: str,
                     db_path: Optional[str] = None) -> list[dict]:
        """Infer which experts and skills are needed for a competition."""
        results = self.search(competition_description, n_results=10)

        relevant_chapters = set()
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("chapter_number"):
                relevant_chapters.add(meta["chapter_number"])

        experts = get_all_experts(db_path or self.db_path)
        recommended = []

        for expert in experts:
            chapter_match = any(
                expert.get("expert_name", "").startswith(ch)
                for ch in relevant_chapters
            )

            concept_overlap = self._concept_overlap(
                competition_description, expert.get("capabilities", [])
            )

            if chapter_match or concept_overlap > 0.3:
                recommended.append({
                    "expert": expert["expert_name"],
                    "slug": expert["slug"],
                    "skills": expert.get("skills", []),
                    "strategy": expert.get("strategy", ""),
                    "relevance": max(
                        concept_overlap,
                        0.8 if chapter_match else 0.0
                    ),
                })

        recommended.sort(key=lambda x: x["relevance"], reverse=True)
        return recommended

    def _concept_overlap(self, query: str, capabilities: list[str]) -> float:
        """Compute concept overlap between query and capabilities."""
        if not capabilities:
            return 0.0

        query_lower = query.lower()
        matches = sum(1 for cap in capabilities if any(
            word in query_lower for word in cap.lower().split()
        ))
        return matches / max(len(capabilities), 1)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        """Split text into overlapping chunks by word boundaries."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap

        return chunks
