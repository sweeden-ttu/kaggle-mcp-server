"""FastMCP server for the MLSysEng Mixture of Experts system.

Exposes tools for:
- Knowledge extraction from ML Principles PDFs
- Semantic search over extracted content
- Expert listing and querying
- Competition entry building
- Convergence loop execution
- RDAgent context generation
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MoEDatabase
from .docling_worker import extract_all_chapters, ML_PRINCIPLES_PATH
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import run_evolve

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MoEDatabase] = None
_embeddings: Optional[EmbeddingStore] = None
_registry: Optional[ExpertRegistry] = None


def _get_db() -> MoEDatabase:
    global _db
    if _db is None:
        _db = MoEDatabase()
    return _db


def _get_embeddings() -> EmbeddingStore:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingStore()
    return _embeddings


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


# ── Knowledge Extraction ────────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans all chapter folders in ML Principles, extracts PDF content
    using docling, generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    extraction_results = extract_all_chapters(db, force=force_reindex)

    indexed_count = 0
    for chapter in db.list_chapters():
        if chapter.get("content_md"):
            try:
                n = embeddings.index_chapter(
                    chapter_id=chapter["chapter_id"],
                    title=chapter["title"],
                    content=chapter["content_md"],
                )
                indexed_count += n
            except Exception as exc:
                logger.warning("Failed to index %s: %s", chapter["chapter_id"], exc)

    experts = registry.create_experts_from_extraction()

    return json.dumps(
        {
            "status": "done",
            "extraction_results": extraction_results,
            "chunks_indexed": indexed_count,
            "experts_created": len(experts),
            "expert_names": [e["expert_name"] for e in experts],
        },
        indent=2,
    )


@mcp.tool()
def evolve(
    competition: str = "titanic",
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> str:
    """Run the convergence loop for a competition.

    Iterates expert-guided optimisation until the state vector converges
    (||state[n] - state[n-1]||_2 < epsilon) or max iterations are reached.

    Args:
        competition: Competition name/slug.
        epsilon: Convergence threshold.
        max_iterations: Maximum number of iterations.
        patience: Required consecutive converging steps.

    Returns:
        JSON with convergence results and history.
    """
    db = _get_db()
    registry = _get_registry()
    experts = registry.list_experts()

    result = run_evolve(
        competition=competition,
        experts=experts,
        db=db,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    return json.dumps(result, indent=2)


# ── Search ──────────────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles.

    Uses embeddings and ChromaDB for vector similarity search over
    extracted chapter content.

    Args:
        query: Search query (natural language).
        n_results: Number of results to return.

    Returns:
        JSON list of matching passages with similarity scores.
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


# ── Experts ─────────────────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """List all chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills,
        strategies, and formulas.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Expert slug or name (e.g., "08_ml_systems").
        question: The question to ask.

    Returns:
        JSON with the expert's recommendation, capabilities, and context.
    """
    registry = _get_registry()
    result = registry.ask_expert(expert_slug, question)
    return json.dumps(result, indent=2)


# ── Competition Entry ───────────────────────────────────────────────


@mcp.tool()
def build_entry(competition: str, description: str = "") -> str:
    """Build competition entry using expert knowledge.

    Combines recommendations from relevant experts into a cohesive
    competition strategy with skills, capabilities, and execution plan.

    Args:
        competition: Competition name (e.g., "titanic").
        description: Optional competition description for better matching.

    Returns:
        JSON competition entry plan.
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()
    experts = registry.list_experts()

    relevant = None
    if description and experts:
        try:
            relevant = embeddings.infer_skills(description, experts)
        except Exception:
            pass

    if relevant is None:
        relevant = experts

    entry = registry.build_competition_entry(
        competition=competition,
        description=description,
        relevant_experts=relevant,
    )

    return json.dumps(entry, indent=2)


# ── RDAgent ─────────────────────────────────────────────────────────


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_results: int = 5,
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt for rdagent based on semantic search
    over extracted ML Principles content.

    Args:
        competition_name: Competition name.
        description: Competition description.
        n_results: Number of context chunks to retrieve.

    Returns:
        Formatted context string for rdagent.
    """
    embeddings = _get_embeddings()
    context = embeddings.get_rdagent_context(
        competition_name=competition_name,
        description=description,
        n_results=n_results,
    )
    return context


# ── Status / Stats ──────────────────────────────────────────────────


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for all chapters.

    Returns:
        JSON list of chapter extraction statuses.
    """
    db = _get_db()
    statuses = db.get_extraction_status()
    return json.dumps(statuses, indent=2)


@mcp.tool()
def get_stats() -> str:
    """System statistics.

    Returns:
        JSON with chapter, expert, concept counts and embedding stats.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        emb_stats = embeddings.get_stats()
    except Exception:
        emb_stats = {"error": "Embedding store not initialised"}

    return json.dumps(
        {"database": db_stats, "embeddings": emb_stats},
        indent=2,
    )


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
