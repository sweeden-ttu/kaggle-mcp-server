"""FastMCP server for the MLSysEng MoE system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MLSysEngDB
from .docling_worker import discover_chapters, extract_chapter
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MLSysEngDB] = None
_embedding_store: Optional[EmbeddingStore] = None
_expert_registry: Optional[ExpertRegistry] = None
_loop_controller: Optional[LoopController] = None


def _get_db() -> MLSysEngDB:
    global _db
    if _db is None:
        _db = MLSysEngDB()
    return _db


def _get_embedding_store() -> EmbeddingStore:
    global _embedding_store
    if _embedding_store is None:
        _embedding_store = EmbeddingStore()
    return _embedding_store


def _get_expert_registry() -> ExpertRegistry:
    global _expert_registry
    if _expert_registry is None:
        _expert_registry = ExpertRegistry(db=_get_db())
    return _expert_registry


def _get_loop_controller() -> LoopController:
    global _loop_controller
    if _loop_controller is None:
        _loop_controller = LoopController(db=_get_db())
    return _loop_controller


# ── Knowledge Extraction ────────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders, extracts PDF content using docling, generates
    embeddings, and registers chapter experts.

    Args:
        force_reindex: Re-extract and re-index all chapters even if already done (default: False)

    Returns:
        JSON summary of extraction results
    """
    db = _get_db()
    embedding_store = _get_embedding_store()
    registry = _get_expert_registry()

    chapters = discover_chapters()
    if not chapters:
        return json.dumps({
            "status": "warning",
            "message": "No chapter folders found. Check ML_PRINCIPLES_PATH environment variable.",
            "path_checked": os.environ.get("ML_PRINCIPLES_PATH", "~/Desktop/Machine Learning Principles - Chapters"),
        })

    results = []
    for ch_info in chapters:
        chapter_name = ch_info["chapter_name"]

        if not force_reindex:
            existing = db.get_chapter(chapter_name)
            if existing:
                results.append({
                    "chapter": chapter_name,
                    "status": "skipped",
                    "reason": "already indexed",
                })
                continue

        try:
            def status_cb(name, status, progress):
                db.update_extraction_status(name, status, progress)

            name, markdown, concepts = extract_chapter(
                ch_info["path"], db=db, status_callback=status_cb
            )

            chapter_row = db.get_chapter(name)
            chapter_id = chapter_row["id"] if chapter_row else None

            chunk_count = 0
            try:
                chunk_count = embedding_store.index_chapter(
                    chapter_name=name,
                    markdown_content=markdown,
                    concepts=concepts,
                    chapter_id=chapter_id,
                    db=db,
                )
            except ImportError as ie:
                logger.warning("Embedding indexing unavailable: %s", ie)

            expert = registry.register_expert_from_chapter(
                chapter_name=name,
                concepts=concepts,
                chapter_id=chapter_id,
            )

            results.append({
                "chapter": name,
                "status": "success",
                "concepts_found": len(concepts),
                "chunks_indexed": chunk_count,
                "expert_slug": expert["slug"],
            })

        except Exception as exc:
            db.update_extraction_status(chapter_name, "failed", error_message=str(exc))
            results.append({
                "chapter": chapter_name,
                "status": "failed",
                "error": str(exc),
            })

    stats = db.get_stats()
    return json.dumps({
        "status": "completed",
        "chapters_processed": len(results),
        "results": results,
        "stats": stats,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """
    Run the MoE convergence loop for a competition.

    Selects relevant experts using RAG, runs the state convergence loop,
    and builds a competition entry when converged.

    Args:
        competition: Competition name/slug (default: "titanic")
        max_iterations: Maximum iterations before stopping (default: 10)

    Returns:
        JSON with convergence results and competition entry
    """
    registry = _get_expert_registry()
    controller = _get_loop_controller()
    controller.max_iterations = max_iterations

    try:
        embedding_store = _get_embedding_store()
        experts = registry.get_experts_for_competition(
            competition, embedding_store=embedding_store
        )
    except ImportError:
        experts = registry.list_experts()

    if not experts:
        return json.dumps({
            "status": "warning",
            "message": "No experts registered. Run extract_knowledge first.",
        })

    result = controller.run(competition, experts)
    entry = controller.build_competition_entry(competition, experts, result)

    return json.dumps({
        "status": "completed",
        "convergence": {
            "converged": result["converged"],
            "iterations": result["iterations"],
            "final_l2_norm": result["final_l2_norm"],
            "epsilon": result["epsilon"],
        },
        "entry": entry,
    }, indent=2)


# ── Search & Query ──────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON list of matching content with relevance scores
    """
    try:
        store = _get_embedding_store()
        results = store.search(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except ImportError as ie:
        return json.dumps({
            "status": "error",
            "message": f"Embedding search unavailable: {ie}",
        })


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON list of expert definitions
    """
    registry = _get_expert_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a competition entry using expert knowledge and RAG-informed skill selection.

    Args:
        competition: Competition name/slug (default: "titanic")

    Returns:
        JSON with selected experts, skills, and recommended approach
    """
    return evolve(competition)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """
    Query a specific chapter expert about a topic.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems")
        question: Question to ask the expert

    Returns:
        JSON with expert response including relevant knowledge
    """
    registry = _get_expert_registry()
    try:
        store = _get_embedding_store()
    except ImportError:
        store = None

    result = registry.ask_expert(expert_slug, question, embedding_store=store)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check the progress of PDF extraction.

    Returns:
        JSON list of chapter extraction statuses
    """
    db = _get_db()
    statuses = db.get_extraction_status()
    return json.dumps(statuses, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get system statistics.

    Returns:
        JSON with counts of chapters, experts, embeddings, and convergence runs
    """
    db = _get_db()
    stats = db.get_stats()

    try:
        store = _get_embedding_store()
        stats["vector_store_count"] = store.get_collection_count()
    except (ImportError, Exception):
        stats["vector_store_count"] = "unavailable"

    return json.dumps(stats, indent=2)


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_context_results: int = 5,
) -> str:
    """
    Prepare rdagent context with ML Principles knowledge.

    Generates a context prompt enriched with relevant ML Principles content
    for use with rdagent data_science competitions.

    Args:
        competition_name: Competition name (e.g., "image classification")
        description: Competition description
        n_context_results: Number of context results to include (default: 5)

    Returns:
        JSON with rdagent context prompt and relevant knowledge
    """
    registry = _get_expert_registry()

    context_parts = [
        f"Competition: {competition_name}",
        f"Description: {description}" if description else "",
        "",
        "## Relevant ML Principles Knowledge",
    ]

    try:
        store = _get_embedding_store()
        query = f"{competition_name} {description}"
        results = store.search(query, n_results=n_context_results)
        for r in results:
            context_parts.append(f"\n### {r.get('chapter', 'Unknown')} (score: {r.get('relevance_score', 0):.3f})")
            context_parts.append(r.get("text", ""))
    except ImportError:
        context_parts.append("(Embedding search unavailable)")

    experts = registry.list_experts()
    if experts:
        context_parts.append("\n## Available Experts")
        for e in experts[:5]:
            context_parts.append(f"- **{e['expert_name']}**: {', '.join(e.get('capabilities', [])[:3])}")

    context = "\n".join(context_parts)

    return json.dumps({
        "competition": competition_name,
        "context_prompt": context,
        "experts_available": len(experts),
        "rdagent_command": f"rdagent --competition {competition_name} --context-file mlsyseng_context.md",
    }, indent=2)


# ── Entry point ─────────────────────────────────────────────────────


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
