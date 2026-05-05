"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import database

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_knowledge"


class EmbeddingStore:
    """Manages embeddings and vector search using ChromaDB."""

    def __init__(self, persist_dir: Optional[str] = None, model_name: Optional[str] = None):
        self._persist_dir = persist_dir or CHROMA_DB_PATH
        self._model_name = model_name or EMBEDDING_MODEL
        self._client = None
        self._collection = None
        self._embedding_fn = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    @property
    def embedding_fn(self):
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._model_name
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

    def index_chapter(self, chapter: Dict[str, Any]) -> int:
        """Index a chapter's content into the vector store.

        Splits text into chunks and stores with metadata.
        Returns the number of chunks indexed.
        """
        text = chapter.get("extracted_text", "")
        if not text:
            return 0

        chapter_number = chapter["chapter_number"]
        title = chapter["title"]
        concepts = chapter.get("concepts", [])

        chunks = self._chunk_text(text, chunk_size=500, overlap=50)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"ch{chapter_number:02d}_chunk{i:04d}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
                "concepts": json.dumps(concepts),
                "source": "ml_principles",
            })

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        return len(ids)

    def index_all_chapters(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Index all chapters from the database into ChromaDB."""
        chapters = database.get_all_chapters(db_path=db_path)
        total_chunks = 0
        indexed_chapters = 0

        for chapter in chapters:
            if chapter.get("extracted_text"):
                chunks = self.index_chapter(chapter)
                total_chunks += chunks
                indexed_chapters += 1

        return {
            "indexed_chapters": indexed_chapters,
            "total_chunks": total_chunks,
            "collection_count": self.collection.count(),
        }

    def search(
        self, query: str, n_results: int = 5, filter_chapter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed knowledge.

        Returns a list of results with text, metadata, and distance scores.
        """
        where_filter = None
        if filter_chapter is not None:
            where_filter = {"chapter_number": filter_chapter}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "relevance_score": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                })

        return output

    def search_concepts(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant concepts across all chapters."""
        results = self.search(query, n_results=n_results)

        concept_scores: Dict[str, float] = {}
        for result in results:
            metadata = result.get("metadata", {})
            concepts_json = metadata.get("concepts", "[]")
            concepts = json.loads(concepts_json) if isinstance(concepts_json, str) else concepts_json
            score = result.get("relevance_score", 0.0)

            for concept in concepts:
                if concept in concept_scores:
                    concept_scores[concept] = max(concept_scores[concept], score)
                else:
                    concept_scores[concept] = score

        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"concept": c, "relevance_score": s} for c, s in sorted_concepts]

    def infer_skills_for_competition(
        self, competition_description: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are needed for a competition.

        Uses semantic search to find relevant knowledge and maps to experts.
        """
        results = self.search(competition_description, n_results=n_results)

        chapter_relevance: Dict[int, float] = {}
        for result in results:
            ch_num = result.get("metadata", {}).get("chapter_number")
            if ch_num is not None:
                score = result.get("relevance_score", 0.0)
                chapter_relevance[ch_num] = max(
                    chapter_relevance.get(ch_num, 0.0), score
                )

        inferred = []
        for ch_num, relevance in sorted(
            chapter_relevance.items(), key=lambda x: x[1], reverse=True
        ):
            inferred.append({
                "chapter_number": ch_num,
                "relevance_score": relevance,
            })

        return inferred

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "model": self._model_name,
            "persist_dir": self._persist_dir,
            "collection_name": COLLECTION_NAME,
        }


_store_instance: Optional[EmbeddingStore] = None


def get_store(persist_dir: Optional[str] = None) -> EmbeddingStore:
    """Get or create the global embedding store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = EmbeddingStore(persist_dir=persist_dir)
    return _store_instance
