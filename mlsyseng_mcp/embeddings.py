"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over indexed ML Principles chapters
and skill inference for competition entry building.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


class EmbeddingEngine:
    """Manages embeddings and vector search via ChromaDB + sentence-transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chroma_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.chroma_path = _ensure_dir(chroma_path or DEFAULT_CHROMA_PATH)
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._model

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.Client(
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=self.chroma_path,
                        anonymized_telemetry=False,
                    )
                )
            except TypeError:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _doc_id(self, chapter_number: str, chunk_idx: int) -> str:
        return f"ch{chapter_number}_chunk{chunk_idx}"

    def index_chapter(self, chapter_number: str, title: str, content: str) -> int:
        """Index a chapter's content into ChromaDB. Returns number of chunks indexed."""
        chunks = _chunk_text(content)
        if not chunks:
            return 0

        ids = [self._doc_id(chapter_number, i) for i in range(len(chunks))]
        metadatas = [
            {"chapter_number": chapter_number, "title": title, "chunk_idx": i}
            for i in range(len(chunks))
        ]

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def index_all_chapters(self, db: Database) -> Dict[str, int]:
        """Index all chapters from the database. Returns {chapter_number: chunk_count}."""
        chapters = db.list_chapters()
        results = {}
        for ch in chapters:
            content = ch.get("content_md", "")
            if not content:
                continue
            count = self.index_chapter(
                ch["chapter_number"], ch["title"], content
            )
            results[ch["chapter_number"]] = count
            logger.info(
                "Indexed chapter %s (%s): %d chunks",
                ch["chapter_number"], ch["title"], count,
            )
        return results

    def search(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed content."""
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                dist = results["distances"][0][i] if results.get("distances") else 0.0
                output.append({
                    "document": doc,
                    "chapter_number": meta.get("chapter_number", ""),
                    "title": meta.get("title", ""),
                    "chunk_idx": meta.get("chunk_idx", 0),
                    "similarity": 1.0 - dist,
                })
        return output

    def infer_skills_for_competition(
        self,
        competition_description: str,
        db: Database,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are needed for a competition.

        Performs semantic search, then maps the best-matching chapters
        to their registered experts and skills.
        """
        search_results = self.search(competition_description, n_results=n_results)

        experts_seen = set()
        recommended = []

        for result in search_results:
            chapter_num = result["chapter_number"]
            if chapter_num in experts_seen:
                continue
            experts_seen.add(chapter_num)

            experts = db.list_experts()
            for expert in experts:
                if expert.get("chapter_id"):
                    ch = db.get_chapter(chapter_num)
                    if ch and ch["id"] == expert["chapter_id"]:
                        recommended.append({
                            "expert": expert,
                            "relevance": result["similarity"],
                            "matched_content": result["document"][:200],
                        })

        return recommended

    def generate_rdagent_context(
        self,
        competition_name: str,
        description: str = "",
        n_results: int = 5,
    ) -> str:
        """Generate a context prompt for rdagent with relevant ML principles."""
        query = f"{competition_name} {description}".strip()
        results = self.search(query, n_results=n_results)

        if not results:
            return f"No relevant ML principles found for '{competition_name}'."

        context_parts = [
            f"# ML Principles Context for: {competition_name}\n",
            "Based on semantic search of ML Principles chapters, "
            "the following knowledge is most relevant:\n",
        ]

        for i, r in enumerate(results, 1):
            context_parts.append(
                f"## {i}. Chapter {r['chapter_number']}: {r['title']} "
                f"(relevance: {r['similarity']:.3f})\n"
            )
            context_parts.append(r["document"])
            context_parts.append("")

        context_parts.append(
            "\n## Recommended Approach\n"
            "Use the above ML principles to guide your approach. "
            "Focus on the most relevant techniques and ensure your "
            "implementation follows established best practices."
        )

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about the embedding store."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "model": self.model_name,
            "chroma_path": self.chroma_path,
            "collection": COLLECTION_NAME,
            "total_chunks": count,
        }
