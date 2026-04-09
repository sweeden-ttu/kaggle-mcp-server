"""Embedding generation and RAG retrieval using sentence-transformers + ChromaDB."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_chroma_client = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
    return _model


def _get_collection(db_path: Optional[str] = None):
    global _chroma_client, _collection
    if _collection is None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise RuntimeError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
        persist_dir = db_path or CHROMA_DB_PATH
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_dir,
                anonymized_telemetry=False,
            )
        )
        _collection = _chroma_client.get_or_create_collection(
            name="mlsyseng_chapters",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def generate_embeddings(text: str) -> List[float]:
    """Generate an embedding vector for the given text."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def index_chapter(
    chapter_id: int,
    folder_name: str,
    title: str,
    markdown: str,
    concepts: List[str],
    db_path: Optional[str] = None,
) -> int:
    """Index a chapter's content into ChromaDB. Returns number of chunks added."""
    collection = _get_collection(db_path)
    chunks = _chunk_text(markdown)
    if not chunks:
        return 0

    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True).tolist()

    ids = [f"ch{chapter_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "chapter_id": chapter_id,
            "folder_name": folder_name,
            "title": title,
            "concepts": ", ".join(concepts),
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def search(
    query: str,
    n_results: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic search over indexed ML Principles chapters."""
    collection = _get_collection(db_path)
    model = _get_model()
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            hits.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                    "similarity": 1.0 - (distance or 0.0),
                }
            )
    return hits


def infer_experts_for_competition(
    competition_description: str,
    experts: List[Dict[str, Any]],
    n_results: int = 3,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Use RAG to determine which experts are most relevant for a competition."""
    hits = search(competition_description, n_results=n_results * 2, db_path=db_path)

    chapter_scores: Dict[str, float] = {}
    for hit in hits:
        folder = hit["metadata"].get("folder_name", "")
        score = hit.get("similarity", 0.0)
        chapter_scores[folder] = max(chapter_scores.get(folder, 0.0), score)

    expert_map = {e.get("slug", ""): e for e in experts}
    ranked: List[Dict[str, Any]] = []
    for folder, score in sorted(chapter_scores.items(), key=lambda x: -x[1]):
        for slug, expert in expert_map.items():
            ch_folder = ""
            if expert.get("chapter_id"):
                ch_folder = folder
            if folder.replace(" ", "_").lower() in slug or slug in folder.replace(" ", "_").lower():
                ranked.append({**expert, "relevance_score": score})
                break

    if not ranked:
        for expert in experts[:n_results]:
            ranked.append({**expert, "relevance_score": 0.5})

    return ranked[:n_results]
