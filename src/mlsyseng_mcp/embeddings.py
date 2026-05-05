"""Embedding generation and RAG retrieval for MLSysEng MoE.

Uses sentence-transformers for embedding and ChromaDB for vector storage.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import MLSysEngDB

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlsyseng_chapters"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence.split())
        if current_len + slen > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = []
            overlap_len = 0
            for s in reversed(current_chunk):
                ws = len(s.split())
                if overlap_len + ws > overlap:
                    break
                overlap_words.insert(0, s)
                overlap_len += ws
            current_chunk = overlap_words
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += slen

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class EmbeddingEngine:
    """Manages embeddings and semantic search via ChromaDB."""

    def __init__(self, chroma_path: Optional[str] = None, model_name: Optional[str] = None):
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.model_name = model_name or EMBEDDING_MODEL
        self._client = None
        self._collection = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed")
                return None
        return self._embedder

    def _get_collection(self):
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
                logger.warning("chromadb not installed")
                return None
        return self._collection

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a list of texts."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        embeddings = embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chapter(self, chapter_number: int, title: str, content: str) -> int:
        """Index a chapter's content into ChromaDB. Returns number of chunks indexed."""
        collection = self._get_collection()
        if collection is None:
            return 0

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        embeddings = self.embed_texts(chunks)
        if embeddings is None:
            return 0

        ids = []
        documents = []
        metadatas = []
        valid_embeddings = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(
                f"ch{chapter_number}_{i}_{chunk[:50]}".encode()
            ).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "chapter_number": chapter_number,
                "title": title,
                "chunk_index": i,
            })
            valid_embeddings.append(emb)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=valid_embeddings,
        )

        return len(chunks)

    def index_all_chapters(self, db: MLSysEngDB) -> Dict[str, Any]:
        """Index all chapters from the database."""
        chapters = db.get_all_chapters()
        results = {"indexed": 0, "chunks": 0, "errors": []}

        for ch in chapters:
            if not ch.get("content_md"):
                continue
            try:
                n_chunks = self.index_chapter(
                    ch["chapter_number"], ch["title"], ch["content_md"]
                )
                results["indexed"] += 1
                results["chunks"] += n_chunks
            except Exception as e:
                results["errors"].append({
                    "chapter": ch["title"],
                    "error": str(e),
                })

        return results

    def search(
        self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed chapters."""
        collection = self._get_collection()
        if collection is None:
            return []

        embeddings = self.embed_texts([query])
        if embeddings is None:
            return []

        where_filter = None
        if chapter_filter is not None:
            where_filter = {"chapter_number": chapter_filter}

        try:
            results = collection.query(
                query_embeddings=embeddings,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        hits = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "text": doc,
                    "chapter_number": meta.get("chapter_number"),
                    "title": meta.get("title"),
                    "chunk_index": meta.get("chunk_index"),
                    "similarity": 1.0 - dist,
                })

        return hits

    def infer_skills(
        self, competition_description: str, db: MLSysEngDB, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Infer which experts and skills are relevant for a competition."""
        hits = self.search(competition_description, n_results=n_results)
        experts = db.get_all_experts()

        relevant_chapters = set()
        for hit in hits:
            cn = hit.get("chapter_number")
            if cn is not None:
                relevant_chapters.add(cn)

        matched_experts = []
        for expert in experts:
            ch_id = expert.get("chapter_id")
            if ch_id is None:
                continue
            chapter = db.get_chapter(ch_id) if isinstance(ch_id, int) else None
            if chapter and chapter.get("chapter_number") in relevant_chapters:
                matched_experts.append({
                    "expert": expert,
                    "relevance_hits": [
                        h for h in hits
                        if h.get("chapter_number") == chapter["chapter_number"]
                    ],
                })

        matched_experts.sort(
            key=lambda x: max(
                (h["similarity"] for h in x["relevance_hits"]), default=0
            ),
            reverse=True,
        )

        return matched_experts
