"""Embedding generation and RAG retrieval using ChromaDB.

Provides semantic search over extracted ML Principles content using
sentence-transformers (all-MiniLM-L6-v2) and ChromaDB for vector storage.
"""

import logging
import os
from typing import Optional

from . import database as db

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.mlsyseng/chroma_db"),
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _get_chroma_client(chroma_path: Optional[str] = None):
    """Get or create a persistent ChromaDB client."""
    import chromadb

    path = chroma_path or DEFAULT_CHROMA_PATH
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def _get_embedding_function():
    """Get sentence-transformers embedding function for ChromaDB."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def index_chapters(
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
    force: bool = False,
) -> dict:
    """Index all extracted chapters into ChromaDB.

    Returns stats about the indexing operation. Returns a skip status
    if chromadb or sentence-transformers is not installed.
    """
    try:
        client = _get_chroma_client(chroma_path)
        embed_fn = _get_embedding_function()
    except ImportError as e:
        return {"status": "skipped", "reason": f"Missing dependency: {e}"}

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    chapters = db.list_chapters(db_path)
    if not chapters:
        return {"status": "no_chapters", "indexed": 0}

    total_chunks = 0
    indexed_chapters = 0

    for chapter in chapters:
        content = chapter.get("markdown_content", "")
        if not content:
            continue

        prefix = f"Chapter {chapter['chapter_number']}: {chapter['title']}"
        chunks = _chunk_text(content)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"ch{chapter['chapter_number']:02d}_chunk{i:04d}"

            existing = collection.get(ids=[doc_id])
            if existing["ids"] and not force:
                continue

            ids.append(doc_id)
            documents.append(f"{prefix}\n\n{chunk}")
            metadatas.append({
                "chapter_number": chapter["chapter_number"],
                "title": chapter["title"],
                "folder_name": chapter["folder_name"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total_chunks += len(ids)
            indexed_chapters += 1

    return {
        "status": "done",
        "indexed_chapters": indexed_chapters,
        "total_chunks": total_chunks,
        "collection_count": collection.count(),
    }


def search(
    query: str,
    n_results: int = 5,
    chroma_path: Optional[str] = None,
    chapter_filter: Optional[int] = None,
) -> list[dict]:
    """Semantic search over indexed ML Principles content.

    Returns list of results with text, metadata, and distance scores.
    Returns empty list if chromadb is not installed.
    """
    try:
        client = _get_chroma_client(chroma_path)
        embed_fn = _get_embedding_function()
    except ImportError:
        logger.warning("chromadb/sentence-transformers not installed, search unavailable")
        return []

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    except Exception:
        return []

    where_filter = None
    if chapter_filter is not None:
        where_filter = {"chapter_number": chapter_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    output = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            entry = {
                "text": doc,
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            }
            output.append(entry)
    return output


def get_relevant_experts(
    query: str,
    n_results: int = 3,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Find the most relevant experts for a given query.

    Searches the vector store, identifies which chapters are most relevant,
    then returns the corresponding expert definitions. Returns empty list
    if chromadb is not installed.
    """
    try:
        results = search(query, n_results=n_results * 2, chroma_path=chroma_path)
    except ImportError:
        results = []

    chapter_scores: dict[int, float] = {}
    for r in results:
        ch_num = r["metadata"].get("chapter_number")
        if ch_num is not None:
            dist = r.get("distance", 1.0)
            relevance = 1.0 - (dist or 0.0)
            chapter_scores[ch_num] = max(chapter_scores.get(ch_num, 0), relevance)

    top_chapters = sorted(chapter_scores.items(), key=lambda x: -x[1])[:n_results]

    experts = db.list_experts(db_path)
    expert_by_chapter: dict[int, dict] = {}
    for e in experts:
        if e.get("chapter_id"):
            ch = db.get_chapter(e["chapter_id"], db_path)
            if ch:
                expert_by_chapter[ch["chapter_number"]] = e

    relevant = []
    for ch_num, score in top_chapters:
        expert = expert_by_chapter.get(ch_num)
        if expert:
            expert["relevance_score"] = score
            relevant.append(expert)

    return relevant


def generate_context(
    competition_name: str,
    description: str = "",
    n_results: int = 5,
    chroma_path: Optional[str] = None,
) -> str:
    """Generate an ML Principles context prompt for a competition.

    Used to provide rdagent or other tools with relevant ML knowledge
    for a specific competition. Returns a fallback message if chromadb
    is not installed or no results are found.
    """
    query = f"{competition_name} {description}".strip()
    try:
        results = search(query, n_results=n_results, chroma_path=chroma_path)
    except ImportError:
        results = []

    if not results:
        return f"No ML Principles context available for: {competition_name}"

    sections = [
        f"# ML Principles Context for: {competition_name}",
        "",
    ]
    if description:
        sections.append(f"**Competition**: {description}")
        sections.append("")

    sections.append("## Relevant ML Principles\n")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        title = meta.get("title", "Unknown")
        ch_num = meta.get("chapter_number", "?")
        sections.append(f"### {i}. Chapter {ch_num}: {title}")
        sections.append(r["text"][:1000])
        sections.append("")

    return "\n".join(sections)
