"""Embedding generation and RAG retrieval with ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Optional

from . import database as db

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"


def get_chroma_client(persist_path: Optional[str] = None):
    """Get or create ChromaDB client."""
    import chromadb

    path = persist_path or DEFAULT_CHROMA_PATH
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_embedding_function():
    """Get the sentence-transformer embedding function for ChromaDB."""
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
        metadata={"hnsw:space": "cosine"},
    )


def index_chapter_content(
    chapter_id: int,
    chapter_number: int,
    title: str,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> int:
    """Index chapter content blocks into ChromaDB for semantic search."""
    content_blocks = db.get_chapter_content(chapter_id, db_path)
    if not content_blocks:
        return 0

    collection = get_collection(chroma_path)

    documents = []
    metadatas = []
    ids = []

    for block in content_blocks:
        content = block["content"]
        if len(content.strip()) < 20:
            continue

        doc_id = f"ch{chapter_number}_block{block['id']}"
        documents.append(content)
        metadatas.append({
            "chapter_id": chapter_id,
            "chapter_number": chapter_number,
            "chapter_title": title,
            "block_type": block["block_type"],
            "page_number": block.get("page_number") or 0,
        })
        ids.append(doc_id)

    if documents:
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

    return len(documents)


def index_all_chapters(
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> dict:
    """Index all chapters into ChromaDB."""
    chapters = db.get_all_chapters(db_path)
    total_indexed = 0

    for chapter in chapters:
        count = index_chapter_content(
            chapter_id=chapter["id"],
            chapter_number=chapter["chapter_number"],
            title=chapter["title"],
            db_path=db_path,
            chroma_path=chroma_path,
        )
        total_indexed += count

    return {
        "chapters_processed": len(chapters),
        "documents_indexed": total_indexed,
    }


def search(
    query: str,
    n_results: int = 5,
    chapter_filter: Optional[int] = None,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Semantic search over indexed ML Principles content."""
    collection = get_collection(chroma_path)

    where_filter = None
    if chapter_filter is not None:
        where_filter = {"chapter_number": chapter_filter}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

    search_results = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            search_results.append({
                "content": doc,
                "chapter_number": metadata.get("chapter_number"),
                "chapter_title": metadata.get("chapter_title"),
                "block_type": metadata.get("block_type"),
                "relevance_score": 1.0 - (distance or 0.0),
            })

    return search_results


def get_context_for_competition(
    competition_name: str,
    description: str = "",
    n_results: int = 10,
    chroma_path: Optional[str] = None,
) -> str:
    """Generate RAG context for a competition using semantic search."""
    query = f"{competition_name} {description}".strip()
    results = search(query, n_results=n_results, chroma_path=chroma_path)

    if not results:
        return "No relevant ML principles found for this competition."

    context_parts = [
        f"## Relevant ML Principles for '{competition_name}'\n"
    ]

    for i, r in enumerate(results, 1):
        score = r.get("relevance_score", 0)
        chapter = r.get("chapter_title", "Unknown")
        content = r["content"][:500]
        context_parts.append(
            f"### {i}. From Chapter: {chapter} (relevance: {score:.2f})\n{content}\n"
        )

    return "\n".join(context_parts)


def infer_skills_for_competition(
    competition_name: str,
    description: str = "",
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Infer which experts and skills are relevant for a competition."""
    results = search(
        f"{competition_name} {description}",
        n_results=15,
        chroma_path=chroma_path,
    )

    chapter_relevance: dict[int, float] = {}
    for r in results:
        ch_num = r.get("chapter_number")
        if ch_num is not None:
            score = r.get("relevance_score", 0)
            chapter_relevance[ch_num] = max(
                chapter_relevance.get(ch_num, 0), score
            )

    experts = db.get_all_experts(db_path)
    relevant_experts = []

    for expert in experts:
        ch_id = expert.get("chapter_id")
        if ch_id is None:
            continue
        chapters = db.get_all_chapters(db_path)
        chapter_map = {c["id"]: c["chapter_number"] for c in chapters}
        ch_num = chapter_map.get(ch_id)
        if ch_num and ch_num in chapter_relevance:
            relevant_experts.append({
                "expert": expert,
                "relevance": chapter_relevance[ch_num],
            })

    relevant_experts.sort(key=lambda x: x["relevance"], reverse=True)
    return relevant_experts
