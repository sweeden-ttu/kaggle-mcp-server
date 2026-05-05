"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _try_import_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        return None


def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


class EmbeddingStore:
    """Manages embeddings for ML Principles chapter content."""

    def __init__(self, persist_dir: Optional[str] = None, model_name: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_DB_PATH
        self.model_name = model_name or EMBEDDING_MODEL
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = _try_import_sentence_transformers()
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def client(self):
        if self._client is None:
            chromadb = _try_import_chromadb()
            if chromadb is None:
                raise ImportError("chromadb is required. Install with: pip install chromadb")
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
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

    def index_chapter(
        self,
        chapter_name: str,
        markdown_content: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
        db=None,
    ) -> int:
        """Index a chapter's content into the vector store."""
        chunks = self.chunk_text(markdown_content)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        ids = []
        documents = []
        metadatas = []
        embedding_list = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"{chapter_name}_chunk_{i}_{uuid.uuid4().hex[:8]}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_name": chapter_name,
                "chunk_index": i,
                "concepts": ", ".join(concepts[:10]),
            })
            embedding_list.append(emb)

            if db is not None and chapter_id is not None:
                db.insert_embedding_meta(chapter_id, i, chunk[:200], doc_id)

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embedding_list[start:end],
            )

        logger.info("Indexed %d chunks for chapter '%s'", len(chunks), chapter_name)
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
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
                dist = results["distances"][0][i] if results.get("distances") else None
                output.append({
                    "text": doc,
                    "chapter": meta.get("chapter_name", ""),
                    "concepts": meta.get("concepts", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                    "relevance_score": 1.0 - dist if dist is not None else None,
                })
        return output

    def infer_skills_for_competition(
        self,
        competition_description: str,
        expert_registry=None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Given a competition description, find the most relevant chapters
        and recommend experts/skills.
        """
        results = self.search(competition_description, n_results=n_results)

        chapter_scores: Dict[str, float] = {}
        for r in results:
            ch = r["chapter"]
            score = r.get("relevance_score", 0.5)
            chapter_scores[ch] = max(chapter_scores.get(ch, 0), score)

        recommendations = []
        if expert_registry is not None:
            for chapter, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
                experts = expert_registry.get_experts_for_chapter(chapter)
                for expert in experts:
                    recommendations.append({
                        "expert": expert,
                        "relevance_score": score,
                        "chapter": chapter,
                    })
        else:
            for chapter, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
                recommendations.append({
                    "chapter": chapter,
                    "relevance_score": score,
                })

        return recommendations

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
