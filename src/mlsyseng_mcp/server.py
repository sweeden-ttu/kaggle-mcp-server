"""FastMCP server for MLSysEng MoE - Mixture of Experts system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all_chapters
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingStore] = None
_registry: Optional[ExpertRegistry] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
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


@mcp.tool()
def extract_knowledge(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders in ML Principles, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed (default: False)
        base_path: Override path to ML Principles chapters directory

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    results = extract_all_chapters(
        base_path=base_path,
        force_reindex=force_reindex,
        db=db,
    )

    indexed_count = 0
    for r in results:
        if r["status"] in ("extracted", "skipped"):
            chapter = db.get_chapter(r["folder_name"])
            if chapter and chapter.get("content_md"):
                try:
                    chunks = embeddings.index_chapter(
                        chapter_id=chapter["id"],
                        folder_name=chapter["folder_name"],
                        title=chapter["title"],
                        content=chapter["content_md"],
                        concepts=chapter.get("concepts", []),
                    )
                    indexed_count += chunks
                except Exception as exc:
                    logger.warning("Failed to index embeddings for %s: %s", r["folder_name"], exc)

    experts = registry.create_experts_from_chapters()

    return json.dumps({
        "status": "success",
        "chapters_processed": len(results),
        "chapters_extracted": sum(1 for r in results if r["status"] == "extracted"),
        "chapters_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "chapters_failed": sum(1 for r in results if r["status"] == "failed"),
        "chunks_indexed": indexed_count,
        "experts_created": len(experts),
        "details": results,
    }, indent=2)


@mcp.tool()
def evolve(
    competition: str = "titanic",
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Extracts knowledge (if needed), infers relevant experts, and runs the
    state convergence loop until ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition name/slug (default: "titanic")
        epsilon: Convergence threshold (default: 0.001)
        max_iterations: Maximum iterations (default: 10)
        patience: Consecutive converging iterations needed (default: 3)

    Returns:
        JSON with convergence results and expert recommendations.
    """
    embeddings = _get_embeddings()
    registry = _get_registry()

    relevant = embeddings.infer_skills(competition, n_results=5)

    expert_slugs = []
    for r in relevant:
        title = r.get("title", "")
        if title:
            from .expert_registry import _slugify
            expert_slugs.append(_slugify(title))

    controller = LoopController(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    result = controller.run_competition(competition, expert_slugs)

    entry = registry.build_competition_entry(competition, relevant)

    return json.dumps({
        "competition": competition,
        "convergence": {
            "converged": result.converged,
            "iterations": result.iterations,
            "reason": result.reason,
            "l2_norms": result.l2_norms,
            "final_state": result.final_state.values if result.final_state else None,
        },
        "entry": {
            "experts": entry.get("experts", []),
            "strategy": entry.get("combined_strategy", []),
            "skills": entry.get("all_skills", []),
            "metrics": entry.get("metrics", []),
        },
    }, indent=2)


@mcp.tool()
def search_concepts(
    query: str,
    n_results: int = 5,
) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with matching chunks and metadata.
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)

    return json.dumps({
        "query": query,
        "results": results,
        "count": len(results),
    }, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON with expert details including capabilities, skills, and strategies.
    """
    registry = _get_registry()
    experts = registry.list_experts()

    return json.dumps({
        "experts": experts,
        "count": len(experts),
    }, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG-informed skill selection.

    Args:
        competition: Competition name/slug (default: "titanic")

    Returns:
        JSON with entry plan, expert recommendations, and skills.
    """
    embeddings = _get_embeddings()
    registry = _get_registry()

    relevant = embeddings.infer_skills(competition, n_results=5)
    entry = registry.build_competition_entry(competition, relevant)

    return json.dumps({
        "competition": competition,
        "entry": entry,
        "relevant_chapters": relevant,
    }, indent=2)


@mcp.tool()
def ask_expert(
    expert_slug: str,
    question: str,
) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Slug of the expert to query (e.g., "ml_systems")
        question: Question to ask the expert

    Returns:
        JSON with expert context and chapter content preview.
    """
    registry = _get_registry()
    result = registry.query_expert(expert_slug, question)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the progress of knowledge extraction.

    Returns:
        JSON with extraction log entries.
    """
    db = _get_db()
    status = db.get_extraction_status()
    return json.dumps({
        "extraction_log": status,
        "count": len(status),
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with chapter, expert, extraction, and embedding stats.
    """
    db = _get_db()
    embeddings = _get_embeddings()

    db_stats = db.get_stats()
    try:
        emb_stats = embeddings.get_stats()
    except Exception:
        emb_stats = {"error": "Embedding store not initialized"}

    return json.dumps({
        "database": db_stats,
        "embeddings": emb_stats,
    }, indent=2)


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_context_results: int = 5,
) -> str:
    """Generate context and command for running rdagent with ML Principles knowledge.

    Args:
        competition_name: Name of the Kaggle competition
        description: Competition description for context retrieval
        n_context_results: Number of context chunks to retrieve (default: 5)

    Returns:
        JSON with rdagent command and ML context.
    """
    embeddings = _get_embeddings()

    query = f"{competition_name} {description}".strip()
    context_results = embeddings.search(query, n_results=n_context_results)

    context_text = "\n\n".join(
        f"[{r['metadata'].get('title', 'Unknown')}]: {r['document']}"
        for r in context_results
        if r.get("document")
    )

    rdagent_cmd = (
        f"rdagent --competition {competition_name} "
        f"--context-file ml_context.txt"
    )

    return json.dumps({
        "competition": competition_name,
        "command": rdagent_cmd,
        "context": context_text,
        "context_chunks": len(context_results),
        "instructions": (
            "1. Save the context to ml_context.txt\n"
            "2. Run the rdagent command above\n"
            "3. rdagent will use the ML Principles context for informed decisions"
        ),
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
