"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import os
import re
from typing import Any, Dict, List, Optional

_MODEL_NAME = "all-MiniLM-L6-v2"
_chroma_client = None
_collection = None
_st_model = None


def _default_chroma_path() -> str:
    return os.environ.get(
        "CHROMA_DB_PATH",
        os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
    )


def _get_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(_MODEL_NAME)
    return _st_model


def _get_collection(chroma_path: Optional[str] = None):
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        path = chroma_path or _default_chroma_path()
        os.makedirs(path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=path)
        _collection = _chroma_client.get_or_create_collection(
            name="ml_principles",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_words = []
            overlap_len = 0
            for s in reversed(current):
                wc = len(s.split())
                if overlap_len + wc > overlap:
                    break
                overlap_words.insert(0, s)
                overlap_len += wc
            current = overlap_words
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def index_chapter(
    chapter_num: int,
    title: str,
    markdown: str,
    chroma_path: Optional[str] = None,
) -> int:
    """Embed and store chapter content in ChromaDB. Returns number of chunks indexed."""
    collection = _get_collection(chroma_path)
    model = _get_model()

    chunks = chunk_text(markdown)
    if not chunks:
        return 0

    ids = [f"ch{chapter_num:02d}_chunk{i:04d}" for i in range(len(chunks))]
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    metadatas = [
        {"chapter_num": chapter_num, "title": title, "chunk_index": i}
        for i in range(len(chunks))
    ]

    existing = collection.get(where={"chapter_num": chapter_num})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def search(
    query: str,
    n_results: int = 5,
    chapter_filter: Optional[int] = None,
    chroma_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic search over indexed ML Principles content."""
    collection = _get_collection(chroma_path)
    model = _get_model()

    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    where = {"chapter_num": chapter_filter} if chapter_filter else None
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return output


def infer_skills_for_competition(
    competition_description: str,
    experts: List[Dict[str, Any]],
    n_results: int = 10,
    chroma_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Given a competition description, find the most relevant experts and their skills."""
    results = search(competition_description, n_results=n_results, chroma_path=chroma_path)

    chapter_scores: Dict[int, float] = {}
    for r in results:
        ch = r["metadata"]["chapter_num"]
        score = 1.0 - r["distance"]
        chapter_scores[ch] = max(chapter_scores.get(ch, 0.0), score)

    ranked = []
    for expert in experts:
        ch_id = expert.get("chapter_id")
        if ch_id is None:
            continue
        score = chapter_scores.get(ch_id, 0.0)
        if score > 0:
            ranked.append({
                "expert": expert,
                "relevance_score": round(score, 4),
                "skills": expert.get("skills", []),
            })

    ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked


def get_collection_stats(chroma_path: Optional[str] = None) -> Dict[str, Any]:
    """Return statistics about the vector store."""
    collection = _get_collection(chroma_path)
    count = collection.count()
    return {"total_chunks": count, "model": _MODEL_NAME, "collection": "ml_principles"}
