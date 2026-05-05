"""FastMCP server exposing MLSysEng MoE tools."""

import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MLSysEngDatabase
from .docling_worker import run_extraction
from .embeddings import EmbeddingEngine
from .expert_registry import (
    build_competition_entry,
    get_expert_definition,
    query_expert,
    register_experts_from_chapters,
)
from .loop_controller import run_evolve_loop

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MLSysEngDatabase] = None
_embeddings: Optional[EmbeddingEngine] = None


def _get_db() -> MLSysEngDatabase:
    global _db
    if _db is None:
        _db = MLSysEngDatabase()
    return _db


def _get_embeddings() -> EmbeddingEngine:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingEngine()
    return _embeddings


@mcp.tool()
def extract_knowledge(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
) -> str:
    """
    Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract and re-index even if chapters already exist.
        base_path: Override the default ML Principles path.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()

    extraction = run_extraction(db, base_path=base_path, force_reindex=force_reindex)

    experts_result = register_experts_from_chapters(db)

    try:
        emb = _get_embeddings()
        index_result = emb.index_chapters(db, force=force_reindex)
    except Exception as e:
        logger.warning("Embedding indexing failed (optional): %s", e)
        index_result = {"status": "skipped", "reason": str(e)}

    return json.dumps(
        {
            "extraction": extraction,
            "experts": experts_result,
            "embeddings": index_result,
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
    """
    Run the convergence loop for a competition using registered experts.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition slug (e.g., "titanic").
        epsilon: Convergence threshold for L2 norm.
        max_iterations: Maximum number of loop iterations.
        patience: Consecutive converging iterations before exit.

    Returns:
        JSON summary of convergence results.
    """
    db = _get_db()
    experts = db.list_experts()

    result = run_evolve_loop(
        competition=competition,
        db=db,
        experts=experts,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_concepts(
    query: str,
    n_results: int = 5,
    chapter_filter: Optional[str] = None,
) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language search query (e.g., "neural network optimization").
        n_results: Number of results to return.
        chapter_filter: Optional chapter_id to restrict search.

    Returns:
        JSON list of matching text chunks with similarity scores.
    """
    emb = _get_embeddings()
    results = emb.search(query, n_results=n_results, chapter_filter=chapter_filter)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON list of expert definitions.
    """
    db = _get_db()
    experts = db.list_experts()
    result = []
    for e in experts:
        result.append(
            {
                "expert_name": e.expert_name,
                "slug": e.slug,
                "chapter_id": e.chapter_id,
                "capabilities": e.capabilities,
                "skills": e.skills,
                "strategy": e.strategy,
                "formula": e.formula,
                "loop_config": e.loop_config,
            }
        )
    return json.dumps(result, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a Kaggle competition entry using expert knowledge.

    Uses RAG-informed skill selection to determine which experts
    and skills are most relevant, then produces a competition plan.

    Args:
        competition: Competition slug (e.g., "titanic").

    Returns:
        JSON competition entry plan with experts, skills, and pipeline steps.
    """
    db = _get_db()

    try:
        emb = _get_embeddings()
        recommendations = emb.infer_skills(competition, db)
    except Exception:
        recommendations = None

    entry = build_competition_entry(
        db, competition, recommended_experts=recommendations
    )
    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_results: int = 8,
) -> str:
    """
    Generate rdagent context prompt with ML Principles knowledge.

    Searches the knowledge base for relevant principles and generates
    a context prompt suitable for rdagent data_science competitions.

    Args:
        competition_name: Name of the Kaggle competition.
        description: Optional description of the competition.
        n_results: Number of knowledge chunks to retrieve.

    Returns:
        Formatted context prompt for rdagent.
    """
    emb = _get_embeddings()
    context = emb.get_rdagent_context(competition_name, description, n_results)
    return context


@mcp.tool()
def ask_expert(
    expert_name: str,
    question: str,
) -> str:
    """
    Query a specific chapter expert for advice on a topic.

    Args:
        expert_name: Expert name or slug.
        question: Question to ask the expert.

    Returns:
        JSON with expert context and the question for LLM completion.
    """
    db = _get_db()
    result = query_expert(db, expert_name, question)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check the status of knowledge extraction and indexing.

    Returns:
        JSON with database stats and embedding index status.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        emb = _get_embeddings()
        emb_stats = emb.get_collection_stats()
    except Exception as e:
        emb_stats = {"status": "unavailable", "reason": str(e)}

    return json.dumps(
        {"database": db_stats, "embeddings": emb_stats},
        indent=2,
    )


@mcp.tool()
def get_stats() -> str:
    """
    Get overall system statistics.

    Returns:
        JSON with chapter count, expert count, word count, and convergence run count.
    """
    db = _get_db()
    stats = db.get_stats()
    return json.dumps(stats, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
