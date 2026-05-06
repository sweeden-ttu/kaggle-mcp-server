"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB.

Provides semantic search over extracted ML Principles content.
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.openclaw/workspace/mlsyseng/chroma_db")
)
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _get_chroma_client(persist_dir: str | None = None):
    import chromadb

    path = persist_dir or DEFAULT_CHROMA_PATH
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def _get_embedding_fn():
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            keep = max(0, len(current) - overlap)
            overlap_words = []
            for s in current[keep:]:
                overlap_words.extend(s.split())
            current = [" ".join(overlap_words)] if overlap_words else []
            current_len = len(overlap_words)
        current.append(sentence)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def index_chapter(
    chapter_number: str,
    title: str,
    content: str,
    persist_dir: str | None = None,
) -> int:
    """Index a chapter's content into ChromaDB. Returns number of chunks created."""
    if not content.strip():
        return 0

    client = _get_chroma_client(persist_dir)
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=ef
    )

    existing = collection.get(where={"chapter_number": chapter_number})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunks = _chunk_text(content)
    if not chunks:
        return 0

    ids = [f"{chapter_number}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "chapter_number": chapter_number,
            "title": title,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(
        "Indexed chapter %s (%s) with %d chunks", chapter_number, title, len(chunks)
    )
    return len(chunks)


def search(
    query: str,
    n_results: int = 5,
    persist_dir: str | None = None,
    chapter_filter: str | None = None,
) -> list[dict]:
    """Semantic search over indexed ML Principles content."""
    client = _get_chroma_client(persist_dir)
    ef = _get_embedding_fn()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=ef
        )
    except Exception:
        logger.warning("Collection '%s' not found. Run extraction first.", COLLECTION_NAME)
        return []

    where = {"chapter_number": chapter_filter} if chapter_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            items.append(
                {
                    "text": doc,
                    "chapter_number": meta.get("chapter_number", ""),
                    "title": meta.get("title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": distance,
                    "relevance": 1.0 - (distance or 0.0),
                }
            )
    return items


def infer_skills(
    competition_description: str,
    n_results: int = 10,
    persist_dir: str | None = None,
) -> list[dict]:
    """Infer which experts and skills are needed for a competition."""
    results = search(competition_description, n_results=n_results, persist_dir=persist_dir)

    chapter_scores: dict[str, float] = {}
    chapter_titles: dict[str, str] = {}
    for r in results:
        ch = r["chapter_number"]
        chapter_scores[ch] = chapter_scores.get(ch, 0) + r["relevance"]
        chapter_titles[ch] = r["title"]

    ranked = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "chapter_number": ch,
            "title": chapter_titles[ch],
            "relevance_score": round(score, 4),
        }
        for ch, score in ranked
    ]


def get_collection_stats(persist_dir: str | None = None) -> dict:
    """Get statistics about the ChromaDB collection."""
    try:
        client = _get_chroma_client(persist_dir)
        ef = _get_embedding_fn()
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=ef
        )
        count = collection.count()
        sample = collection.peek(limit=1)
        chapters = set()
        if sample and sample["metadatas"]:
            for meta in sample["metadatas"]:
                chapters.add(meta.get("chapter_number", ""))
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": MODEL_NAME,
        }
    except Exception as e:
        return {"total_chunks": 0, "error": str(e)}
