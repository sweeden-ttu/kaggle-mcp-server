"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB.

Provides semantic search over extracted ML Principles content and
automatic skill/expert inference for competitions.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


def _default_chroma_path() -> str:
    return os.environ.get(
        "CHROMA_DB_PATH",
        os.path.expanduser("~/.mlsyseng/chroma_db"),
    )


class EmbeddingEngine:
    """Manages embeddings and vector search over ML Principles content."""

    def __init__(
        self,
        db: Optional[Database] = None,
        model_name: str = DEFAULT_MODEL,
        chroma_path: Optional[str] = None,
    ):
        self.db = db or Database()
        self.model_name = model_name
        self.chroma_path = chroma_path or _default_chroma_path()
        self._model = None
        self._chroma_client = None
        self._collection = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; "
                    "using hash-based fallback embeddings"
                )
                self._model = _FallbackEmbedder()
        return self._model

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path
                )
                self._collection = self._chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.warning("chromadb not installed; using in-memory fallback")
                self._collection = _FallbackCollection()
        return self._collection

    def index_chapters(self, force: bool = False) -> Dict[str, Any]:
        """Index all extracted chapters into the vector store."""
        chapters = self.db.list_chapters()
        collection = self._get_collection()
        model = self._get_model()

        indexed = 0
        skipped = 0

        for chapter in chapters:
            if chapter["status"] != "extracted":
                skipped += 1
                continue

            content = chapter.get("markdown_content", "")
            if not content:
                skipped += 1
                continue

            doc_id = f"chapter_{chapter['chapter_name']}"
            chunks = self._chunk_text(content, chunk_size=500, overlap=50)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"

                if not force:
                    existing = collection.get(ids=[chunk_id])
                    if existing and existing.get("ids"):
                        skipped += 1
                        continue

                embedding = self._encode(model, chunk)
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "chapter": chapter["chapter_name"],
                        "chunk_index": i,
                        "concepts": ",".join(chapter.get("concepts", [])),
                    }],
                )
                indexed += 1

        return {
            "indexed_chunks": indexed,
            "skipped": skipped,
            "total_chapters": len(chapters),
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_chapter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed ML Principles content."""
        collection = self._get_collection()
        model = self._get_model()
        query_embedding = self._encode(model, query)

        where_filter = None
        if filter_chapter:
            where_filter = {"chapter": filter_chapter}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })
        return output

    def infer_experts(
        self,
        competition_description: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Infer which experts are most relevant for a competition."""
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        chapter_concepts: Dict[str, set] = {}

        for r in results:
            chapter = r.get("metadata", {}).get("chapter", "unknown")
            distance = r.get("distance", 1.0)
            score = max(0.0, 1.0 - distance)

            if chapter not in chapter_scores:
                chapter_scores[chapter] = 0.0
                chapter_concepts[chapter] = set()

            chapter_scores[chapter] += score
            concepts_str = r.get("metadata", {}).get("concepts", "")
            if concepts_str:
                chapter_concepts[chapter].update(concepts_str.split(","))

        ranked = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)

        experts_needed = []
        for chapter, score in ranked:
            expert = self.db.get_expert(self._slugify(chapter))
            if expert:
                experts_needed.append({
                    "expert": expert,
                    "relevance_score": round(score, 4),
                    "matching_concepts": sorted(chapter_concepts.get(chapter, set())),
                })
            else:
                experts_needed.append({
                    "chapter": chapter,
                    "relevance_score": round(score, 4),
                    "matching_concepts": sorted(chapter_concepts.get(chapter, set())),
                    "note": "expert not registered yet",
                })

        return experts_needed

    def get_rdagent_context(
        self,
        competition_name: str,
        description: str = "",
    ) -> str:
        """Generate a context prompt for rdagent using ML Principles knowledge."""
        query = f"{competition_name} {description}".strip()
        results = self.search(query, n_results=8)

        context_parts = [
            f"## ML Principles Context for '{competition_name}'\n",
            "Based on the ML Principles knowledge base, here are the most relevant concepts:\n",
        ]

        for i, r in enumerate(results, 1):
            chapter = r.get("metadata", {}).get("chapter", "unknown")
            doc = r.get("document", "")
            context_parts.append(f"### {i}. From {chapter}\n{doc}\n")

        expert_recs = self.infer_experts(query, n_results=5)
        if expert_recs:
            context_parts.append("\n## Recommended Expert Strategies\n")
            for rec in expert_recs:
                expert = rec.get("expert", {})
                name = expert.get("expert_name", rec.get("chapter", "unknown"))
                strategy = expert.get("strategy", "N/A")
                context_parts.append(
                    f"- **{name}** (relevance: {rec['relevance_score']}): {strategy}\n"
                )

        return "\n".join(context_parts)

    def _encode(self, model, text: str) -> List[float]:
        if isinstance(model, _FallbackEmbedder):
            return model.encode(text)
        embedding = model.encode(text)
        return embedding.tolist()

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks

    @staticmethod
    def _slugify(name: str) -> str:
        import re
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[-\s]+", "_", slug).strip("_")
        return slug


class _FallbackEmbedder:
    """Hash-based pseudo-embeddings when sentence-transformers is unavailable."""

    DIM = 384

    def encode(self, text: str) -> List[float]:
        h = hashlib.sha384(text.encode("utf-8")).digest()
        vec = [((b - 128) / 128.0) for b in h]
        norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]


class _FallbackCollection:
    """In-memory vector store when ChromaDB is unavailable."""

    def __init__(self):
        self._store: Dict[str, Dict] = {}

    def get(self, ids: List[str]) -> Dict:
        found = [i for i in ids if i in self._store]
        return {"ids": found}

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ):
        for i, doc_id in enumerate(ids):
            self._store[doc_id] = {
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
    ) -> Dict:
        if not self._store:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_vec = query_embeddings[0]
        scored = []
        for doc_id, data in self._store.items():
            if where:
                match = all(
                    data["metadata"].get(k) == v for k, v in where.items()
                )
                if not match:
                    continue
            dist = self._cosine_distance(query_vec, data["embedding"])
            scored.append((doc_id, data, dist))

        scored.sort(key=lambda x: x[2])
        top = scored[:n_results]

        return {
            "ids": [[t[0] for t in top]],
            "documents": [[t[1]["document"] for t in top]],
            "metadatas": [[t[1]["metadata"] for t in top]],
            "distances": [[t[2] for t in top]],
        }

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))
