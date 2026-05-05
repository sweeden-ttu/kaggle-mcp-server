"""Embedding generation and RAG retrieval using sentence-transformers and ChromaDB."""

import os
import logging
from pathlib import Path
from typing import Optional

from . import database

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


def get_chroma_client(persist_path: Optional[str] = None):
    """Get or create a ChromaDB client."""
    import chromadb

    path = persist_path or CHROMA_DB_PATH
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_embedding_function():
    """Get the sentence-transformers embedding function for ChromaDB."""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_collection(persist_path: Optional[str] = None):
    """Get or create the ML principles collection."""
    client = get_chroma_client(persist_path)
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "ML Principles chapter knowledge base"},
    )


def index_chapter(
    chapter_id: int,
    chapter_number: int,
    title: str,
    content: str,
    persist_path: Optional[str] = None,
) -> int:
    """Index a chapter's content into ChromaDB with chunking."""
    collection = get_collection(persist_path)

    chunks = _chunk_text(content, chunk_size=512, overlap=64)
    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = f"ch{chapter_number:02d}_chunk_{i:04d}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "chapter_id": chapter_id,
            "chapter_number": chapter_number,
            "title": title,
            "chunk_index": i,
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(f"Indexed chapter {chapter_number} ({title}): {len(chunks)} chunks")
    return len(chunks)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word boundaries."""
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


def search(
    query: str,
    n_results: int = 5,
    persist_path: Optional[str] = None,
) -> list[dict]:
    """Semantic search over the indexed knowledge base."""
    collection = get_collection(persist_path)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return matches


def infer_relevant_experts(
    competition_description: str,
    n_results: int = 5,
    db_path: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> list[dict]:
    """Infer which experts are relevant for a competition using RAG."""
    search_results = search(competition_description, n_results=n_results * 2, persist_path=persist_path)

    chapter_scores: dict[int, float] = {}
    for result in search_results:
        ch_num = result["metadata"]["chapter_number"]
        distance = result.get("distance", 1.0)
        relevance = 1.0 / (1.0 + distance)
        chapter_scores[ch_num] = chapter_scores.get(ch_num, 0) + relevance

    sorted_chapters = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
    top_chapters = sorted_chapters[:n_results]

    experts = database.get_all_experts(db_path)
    relevant_experts = []
    for chapter_num, score in top_chapters:
        for expert in experts:
            if expert.get("chapter_id"):
                chapters = database.get_all_chapters(db_path)
                chapter_map = {ch["id"]: ch["chapter_number"] for ch in chapters}
                if chapter_map.get(expert["chapter_id"]) == chapter_num:
                    relevant_experts.append({
                        **expert,
                        "relevance_score": score,
                    })
                    break

    return relevant_experts


def index_all_chapters(
    db_path: Optional[str] = None, persist_path: Optional[str] = None
) -> dict:
    """Index all chapters from the database into ChromaDB."""
    chapters = database.get_all_chapters(db_path)
    total_chunks = 0

    for chapter in chapters:
        content = database.get_chapter_content(chapter["id"], db_path)
        if content:
            chunks = index_chapter(
                chapter_id=chapter["id"],
                chapter_number=chapter["chapter_number"],
                title=chapter["title"],
                content=content,
                persist_path=persist_path,
            )
            total_chunks += chunks

    return {
        "chapters_indexed": len(chapters),
        "total_chunks": total_chunks,
    }


def get_rdagent_context(
    competition_name: str,
    description: str = "",
    n_results: int = 5,
    db_path: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> str:
    """Generate a context prompt for rdagent based on ML Principles knowledge."""
    query = f"{competition_name} {description}".strip()
    results = search(query, n_results=n_results, persist_path=persist_path)

    if not results:
        return f"No ML Principles context found for: {competition_name}"

    context_parts = [
        f"# ML Principles Context for: {competition_name}\n",
        "Based on the ML Principles knowledge base, here are relevant concepts:\n",
    ]

    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        context_parts.append(
            f"## {i}. From Chapter {meta['chapter_number']}: {meta['title']}\n"
        )
        context_parts.append(result["document"][:500])
        context_parts.append("")

    experts = infer_relevant_experts(query, n_results=3, db_path=db_path, persist_path=persist_path)
    if experts:
        context_parts.append("\n## Recommended Expert Strategy\n")
        for expert in experts:
            context_parts.append(f"- **{expert['expert_name']}**: {expert.get('strategy', 'N/A')}")

    return "\n".join(context_parts)
