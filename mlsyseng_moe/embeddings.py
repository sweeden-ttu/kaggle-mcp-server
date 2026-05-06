"""Embedding generation and RAG retrieval using ChromaDB."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)

MODEL_NAME = "all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_chroma_client(path: Optional[str] = None) -> chromadb.ClientAPI:
    db_path = path or DEFAULT_CHROMA_PATH
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: Optional[chromadb.ClientAPI] = None) -> chromadb.Collection:
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name="ml_principles",
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_chapter(
    chapter_id: int,
    chapter_title: str,
    content: str,
    chroma_path: Optional[str] = None,
) -> int:
    """Index a chapter's content into ChromaDB. Returns number of chunks indexed."""
    client = get_chroma_client(chroma_path)
    collection = get_collection(client)
    model = get_model()

    chunks = chunk_text(content)
    if not chunks:
        return 0

    ids = [f"ch{chapter_id}_chunk{i}" for i in range(len(chunks))]
    embeddings = model.encode(chunks).tolist()
    metadatas = [
        {"chapter_id": chapter_id, "chapter_title": chapter_title, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def search(
    query: str,
    n_results: int = 5,
    chapter_filter: Optional[int] = None,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Semantic search over indexed ML Principles content."""
    client = get_chroma_client(chroma_path)
    collection = get_collection(client)
    model = get_model()

    query_embedding = model.encode([query]).tolist()

    where_filter = None
    if chapter_filter is not None:
        where_filter = {"chapter_id": chapter_filter}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    search_results = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            search_results.append({
                "content": doc,
                "chapter_title": results["metadatas"][0][i].get("chapter_title", ""),
                "chapter_id": results["metadatas"][0][i].get("chapter_id"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index"),
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

    return search_results


def infer_skills_for_competition(
    competition_description: str,
    experts: list[dict],
    n_results: int = 10,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Infer which experts and skills are most relevant for a competition."""
    results = search(competition_description, n_results=n_results, chroma_path=chroma_path)

    relevant_chapters = set()
    for r in results:
        if r.get("chapter_id"):
            relevant_chapters.add(r["chapter_id"])

    matched_experts = []
    for expert in experts:
        if expert.get("chapter_id") in relevant_chapters:
            matched_experts.append({
                "expert": expert,
                "relevance": "direct_chapter_match",
            })

    if not matched_experts and experts:
        model = get_model()
        comp_emb = model.encode([competition_description])
        for expert in experts:
            cap_text = " ".join(expert.get("capabilities", []))
            if cap_text:
                cap_emb = model.encode([cap_text])
                similarity = float((comp_emb @ cap_emb.T)[0][0])
                if similarity > 0.3:
                    matched_experts.append({
                        "expert": expert,
                        "relevance": f"capability_similarity={similarity:.3f}",
                    })

    matched_experts.sort(
        key=lambda x: x.get("relevance", ""),
        reverse=True,
    )
    return matched_experts


def get_collection_stats(chroma_path: Optional[str] = None) -> dict:
    """Get statistics about the vector store."""
    client = get_chroma_client(chroma_path)
    collection = get_collection(client)
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": "ml_principles",
        "embedding_model": MODEL_NAME,
    }
