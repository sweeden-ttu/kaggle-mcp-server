"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML Principles content.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import database as db

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    "~/.openclaw/workspace/mlsyseng/chroma_db"
)
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chroma_path() -> str:
    return os.environ.get("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH)


def _get_collection():
    """Get or create the ChromaDB collection."""
    import chromadb
    path = _chroma_path()
    Path(path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _get_model():
    """Load the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def _make_id(text: str, idx: int) -> str:
    """Deterministic chunk id for deduplication."""
    h = hashlib.md5(text[:200].encode()).hexdigest()[:12]
    return f"{h}_{idx}"


def index_chapters():
    """Generate embeddings for all chapters and store in ChromaDB."""
    db.init_db()
    chapters = db.get_all_chapters()
    if not chapters:
        return {"status": "no_chapters", "message": "No chapters to index"}

    model = _get_model()
    collection = _get_collection()

    total_chunks = 0
    for chapter in chapters:
        content = chapter.get("markdown_content", "")
        if not content:
            continue

        chunks = _chunk_text(content)
        if not chunks:
            continue

        embeddings = model.encode(chunks).tolist()
        ids = [_make_id(chapter["folder_name"], i) for i in range(len(chunks))]
        metadatas = [{
            "chapter_id": chapter["id"],
            "folder_name": chapter["folder_name"],
            "title": chapter["title"],
            "chunk_index": i,
        } for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    return {
        "status": "completed",
        "chapters_indexed": len(chapters),
        "total_chunks": total_chunks,
        "model": MODEL_NAME,
        "chroma_path": _chroma_path(),
    }


def search(query: str, n_results: int = 5,
           chapter_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Semantic search over the ML Principles knowledge base."""
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    where_filter = None
    if chapter_filter:
        where_filter = {"folder_name": chapter_filter}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else None
            hits.append({
                "text": doc,
                "chapter": meta.get("folder_name", "unknown"),
                "title": meta.get("title", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "similarity": round(1.0 - dist, 4) if dist is not None else None,
            })
    return hits


def infer_skills(competition_description: str,
                 n_results: int = 10) -> List[Dict[str, Any]]:
    """Given a competition description, infer which experts and skills are relevant.

    Returns ranked list of expert matches with similarity scores.
    """
    hits = search(competition_description, n_results=n_results)
    if not hits:
        return []

    chapter_scores: Dict[str, float] = {}
    for hit in hits:
        ch = hit["chapter"]
        score = hit.get("similarity", 0.5)
        chapter_scores[ch] = max(chapter_scores.get(ch, 0), score)

    experts = db.get_all_experts()
    expert_by_chapter = {}
    for exp in experts:
        if exp.get("chapter_id"):
            ch = db.get_chapter(exp["chapter_id"])
            if ch:
                expert_by_chapter[ch["folder_name"]] = exp

    ranked = []
    for ch_name, score in sorted(chapter_scores.items(),
                                  key=lambda x: -x[1]):
        expert = expert_by_chapter.get(ch_name)
        if expert:
            ranked.append({
                "expert_slug": expert["slug"],
                "expert_name": expert["expert_name"],
                "relevance_score": score,
                "capabilities": expert.get("capabilities", []),
                "skills": expert.get("skills", []),
                "strategy": expert.get("strategy", ""),
            })

    return ranked
