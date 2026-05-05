"""
Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML Principles content for
skill inference and expert selection.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


class EmbeddingEngine:
    """Manages embeddings via sentence-transformers and vector storage via ChromaDB."""

    def __init__(
        self,
        chroma_path: str = CHROMA_DB_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.chroma_path = chroma_path
        self.model_name = model_name
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
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb

                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.chroma_path)
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.error(
                    "chromadb not installed. Run: pip install chromadb"
                )
                raise
        return self._collection

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping chunks for embedding."""
        words = text.split()
        chunks: List[str] = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def _text_id(self, text: str, prefix: str = "") -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        return f"{prefix}_{h}" if prefix else h

    def index_chapter(
        self,
        chapter_id: str,
        title: str,
        markdown: str,
        concepts: Optional[List[str]] = None,
    ) -> int:
        """
        Index a chapter's content into the vector store.

        Returns number of chunks indexed.
        """
        if not markdown.strip():
            return 0

        chunks = self._chunk_text(markdown)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        ids = [self._text_id(c, chapter_id) for c in chunks]
        metadatas = [
            {
                "chapter_id": chapter_id,
                "title": title,
                "chunk_index": i,
                "concepts": ",".join(concepts or []),
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed ML Principles content.

        Args:
            query: Search query string
            n_results: Number of results to return
            chapter_filter: Optional chapter_id to limit search

        Returns:
            List of dicts with keys: document, chapter_id, title, score, concepts
        """
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        where = {"chapter_id": chapter_filter} if chapter_filter else None
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "document": doc,
                "chapter_id": meta.get("chapter_id", ""),
                "title": meta.get("title", ""),
                "score": 1.0 - dist,
                "concepts": meta.get("concepts", "").split(","),
            })

        return hits

    def infer_skills(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Given a competition description, infer which experts and skills are relevant.

        Returns a ranked list of expert-skill recommendations.
        """
        hits = self.search(competition_description, n_results=n_results)

        expert_scores: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            cid = hit["chapter_id"]
            if cid not in expert_scores:
                expert_scores[cid] = {
                    "chapter_id": cid,
                    "title": hit["title"],
                    "total_score": 0.0,
                    "concepts": set(),
                    "hit_count": 0,
                }
            expert_scores[cid]["total_score"] += hit["score"]
            expert_scores[cid]["hit_count"] += 1
            for c in hit["concepts"]:
                if c:
                    expert_scores[cid]["concepts"].add(c)

        ranked = sorted(
            expert_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True,
        )
        for r in ranked:
            r["concepts"] = sorted(r["concepts"])
            r["avg_score"] = r["total_score"] / max(r["hit_count"], 1)
        return ranked

    def get_stats(self) -> Dict[str, Any]:
        """Return embedding store statistics."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "model": self.model_name,
            "collection": COLLECTION_NAME,
            "total_chunks": count,
            "chroma_path": self.chroma_path,
        }
