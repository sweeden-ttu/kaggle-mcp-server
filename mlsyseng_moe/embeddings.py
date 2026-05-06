"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Optional

from mlsyseng_moe.database import get_all_chapters, get_chapter_content, get_connection

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingStore:
    """Manages embeddings in ChromaDB with sentence-transformers."""

    def __init__(self, chroma_path: Optional[str] = None, db_path: Optional[str] = None):
        self.chroma_path = chroma_path or DEFAULT_CHROMA_PATH
        self.db_path = db_path
        self._collection = None
        self._client = None
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL)
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
        return self._model

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.chroma_path)
            except ImportError:
                logger.error("chromadb not installed")
                raise
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chapters(self, force_reindex: bool = False) -> dict:
        """Index all chapter content into ChromaDB."""
        chapters = get_all_chapters(self.db_path)
        stats = {"indexed": 0, "skipped": 0, "errors": 0}

        for chapter in chapters:
            try:
                content_blocks = get_chapter_content(chapter["id"], self.db_path)
                if not content_blocks:
                    stats["skipped"] += 1
                    continue

                ids = []
                documents = []
                metadatas = []

                for block in content_blocks:
                    doc_id = f"ch{chapter['chapter_number']}_block{block['id']}"

                    if not force_reindex:
                        existing = self.collection.get(ids=[doc_id])
                        if existing and existing["ids"]:
                            stats["skipped"] += 1
                            continue

                    ids.append(doc_id)
                    documents.append(block["content"])
                    metadatas.append({
                        "chapter_number": chapter["chapter_number"],
                        "chapter_title": chapter["title"],
                        "section_title": block.get("section_title") or "",
                        "page_number": block.get("page_number") or 0,
                    })

                if ids:
                    embeddings = self.generate_embeddings(documents)
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
                    stats["indexed"] += len(ids)

            except Exception as e:
                logger.error(f"Error indexing chapter {chapter['title']}: {e}")
                stats["errors"] += 1

        return stats

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search over indexed ML Principles content."""
        query_embedding = self.generate_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                })

        return search_results

    def get_relevant_experts(self, query: str, n_results: int = 3) -> list[str]:
        """Find which chapter experts are most relevant to a query."""
        results = self.search(query, n_results=n_results * 2)

        chapter_scores: dict[int, float] = {}
        for result in results:
            ch_num = result["metadata"]["chapter_number"]
            score = result["similarity"]
            chapter_scores[ch_num] = max(chapter_scores.get(ch_num, 0), score)

        sorted_chapters = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
        return [str(ch_num) for ch_num, _ in sorted_chapters[:n_results]]

    def get_context_for_competition(self, competition_name: str, description: str = "") -> str:
        """Generate RAG context for a competition using semantic search."""
        query = f"kaggle competition: {competition_name}. {description}"
        results = self.search(query, n_results=10)

        if not results:
            return "No relevant ML principles found in the knowledge base."

        context_parts = ["## Relevant ML Principles\n"]
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            context_parts.append(
                f"### {i}. From Chapter {meta['chapter_number']}: {meta['chapter_title']}\n"
                f"**Section:** {meta['section_title']}\n"
                f"**Relevance:** {result['similarity']:.3f}\n\n"
                f"{result['content'][:500]}...\n"
            )

        return "\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get embedding store statistics."""
        try:
            count = self.collection.count()
            return {
                "total_embeddings": count,
                "collection_name": COLLECTION_NAME,
                "model": EMBEDDING_MODEL,
                "chroma_path": self.chroma_path,
            }
        except Exception:
            return {
                "total_embeddings": 0,
                "collection_name": COLLECTION_NAME,
                "model": EMBEDDING_MODEL,
                "chroma_path": self.chroma_path,
            }
