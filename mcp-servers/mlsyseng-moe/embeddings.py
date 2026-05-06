"""Embedding generation and RAG retrieval for MLSysEng MoE.

Uses sentence-transformers for embedding generation and ChromaDB for
vector storage and similarity search.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

try:
    from . import database as db
except ImportError:
    import database as db

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.expanduser(
    os.environ.get("CHROMA_DB_PATH", "~/.mlsyseng/chroma_db")
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_principles"

_chroma_client = None
_collection = None
_embedding_fn = None


def _get_chroma_client(chroma_path: Optional[str] = None):
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        path = chroma_path or CHROMA_DB_PATH
        Path(path).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=path)
    return _chroma_client


def _get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def _get_collection(chroma_path: Optional[str] = None):
    global _collection
    if _collection is None:
        client = _get_chroma_client(chroma_path)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_chapter(
    chapter_id: int,
    chapter_name: str,
    text: str,
    concepts: list[str],
    chroma_path: Optional[str] = None,
) -> int:
    """Index a chapter's content into ChromaDB.

    Splits text into chunks and stores each with metadata.
    Returns the number of chunks indexed.
    """
    collection = _get_collection(chroma_path)
    chunks = _chunk_text(text, chunk_size=500, overlap=50)

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = f"ch{chapter_id}_chunk{i}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "chunk_index": i,
            "concepts": json.dumps(concepts),
            "source": "ml_principles",
        })

    if concepts:
        concept_doc_id = f"ch{chapter_id}_concepts"
        ids.append(concept_doc_id)
        documents.append(
            f"Chapter: {chapter_name}. Key concepts: {', '.join(concepts)}"
        )
        metadatas.append({
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "chunk_index": -1,
            "concepts": json.dumps(concepts),
            "source": "ml_principles_concepts",
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if words else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def search(
    query: str,
    n_results: int = 5,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Semantic search over indexed ML Principles content."""
    collection = _get_collection(chroma_path)
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return []

    hits = []
    if not results["ids"] or not results["ids"][0]:
        return hits

    for doc_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "id": doc_id,
            "text": document[:500],
            "chapter_name": metadata.get("chapter_name", ""),
            "concepts": json.loads(metadata.get("concepts", "[]")),
            "similarity": 1 - distance,
        })
    return hits


def infer_skills_for_competition(
    competition_description: str,
    n_results: int = 10,
    chroma_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Infer which experts and skills are relevant for a competition.

    Searches the vector store for relevant chapters, then maps to experts.
    """
    hits = search(competition_description, n_results=n_results, chroma_path=chroma_path)

    relevant_chapters = set()
    for hit in hits:
        relevant_chapters.add(hit["chapter_name"])

    experts = db.get_all_experts(db_path)
    scored_experts = []
    for expert in experts:
        chapter_row = None
        if expert.get("chapter_id"):
            chapters = db.get_all_chapters(db_path)
            for ch in chapters:
                if ch["id"] == expert["chapter_id"]:
                    chapter_row = ch
                    break

        relevance = 0.0
        if chapter_row and chapter_row["chapter_name"] in relevant_chapters:
            for hit in hits:
                if hit["chapter_name"] == chapter_row["chapter_name"]:
                    relevance = max(relevance, hit["similarity"])

        expert_concepts = set()
        if chapter_row and chapter_row.get("concepts"):
            concepts = chapter_row["concepts"]
            if isinstance(concepts, str):
                concepts = json.loads(concepts)
            expert_concepts = set(concepts)

        comp_lower = competition_description.lower()
        concept_overlap = sum(
            1 for c in expert_concepts if c.lower() in comp_lower
        )
        relevance += concept_overlap * 0.1

        if relevance > 0:
            scored_experts.append({
                "expert": expert,
                "relevance": round(relevance, 4),
                "matching_concepts": [
                    c for c in expert_concepts if c.lower() in comp_lower
                ],
            })

    scored_experts.sort(key=lambda x: x["relevance"], reverse=True)
    return scored_experts


def index_all_chapters(
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> dict:
    """Index all extracted chapters into the vector store."""
    chapters = db.get_chapters_by_status("completed", db_path)
    total_chunks = 0
    indexed = 0
    for ch in chapters:
        text = ch.get("extracted_text", "")
        concepts = ch.get("concepts", [])
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        if text:
            n = index_chapter(ch["id"], ch["chapter_name"], text, concepts, chroma_path)
            total_chunks += n
            indexed += 1

    return {
        "chapters_indexed": indexed,
        "total_chunks": total_chunks,
    }
