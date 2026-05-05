"""FastMCP server for MLSysEng MoE system.

Exposes tools for knowledge extraction, expert querying, competition entry
building, and convergence loop execution.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import MLSysEngDB
from mlsyseng_mcp.docling_worker import extract_all_chapters, scan_chapter_folders
from mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_mcp.expert_registry import (
    create_expert_from_chapter,
    get_expert_for_query,
    register_all_experts,
)
from mlsyseng_mcp.loop_controller import ConvergenceLoop, build_competition_entry

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-mcp")

_db: Optional[MLSysEngDB] = None
_embedding_engine: Optional[EmbeddingEngine] = None


def _get_db() -> MLSysEngDB:
    global _db
    if _db is None:
        _db = MLSysEngDB()
    return _db


def _get_embedding_engine() -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()

    chapters = extract_all_chapters(force_reindex=force_reindex, db=db)

    try:
        engine = _get_embedding_engine()
        chunks_total = 0
        for ch in chapters:
            content = ch.get("content_md", "")
            if content:
                n = engine.index_chapter(ch["chapter_id"], ch["title"], content, db=db)
                chunks_total += n
    except Exception as exc:
        logger.warning("Embedding indexing skipped: %s", exc)
        chunks_total = 0

    experts = register_all_experts(chapters, db=db)

    return json.dumps(
        {
            "status": "success",
            "chapters_processed": len(chapters),
            "experts_created": len(experts),
            "embedding_chunks": chunks_total,
            "chapters": [
                {"id": c["chapter_id"], "title": c["title"]}
                for c in chapters
            ],
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

    Extracts knowledge (if needed), selects relevant experts via RAG,
    and iterates until state converges.

    Args:
        competition: Kaggle competition name/slug.
        epsilon: Convergence threshold for L2 norm.
        max_iterations: Maximum number of loop iterations.
        patience: Consecutive converging iterations before exit.

    Returns:
        JSON with convergence results.
    """
    db = _get_db()

    chapters = db.list_chapters()
    if not chapters:
        extract_knowledge(force_reindex=False)
        chapters = db.list_chapters()

    experts = db.list_experts()
    if not experts:
        experts = register_all_experts(chapters, db=db)

    try:
        engine = _get_embedding_engine()
        ranked = engine.infer_relevant_experts(competition, experts)
        selected = ranked[:5] if ranked else experts[:3]
    except Exception:
        selected = experts[:3]

    loop = ConvergenceLoop(
        competition=competition,
        experts=selected,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
        db=db,
    )
    result = loop.run()
    return json.dumps(result, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles content.

    Args:
        query: Natural language query.
        n_results: Number of results to return.

    Returns:
        JSON array of search hits with text, chapter, and relevance score.
    """
    engine = _get_embedding_engine()
    hits = engine.search(query, n_results=n_results)
    return json.dumps(hits, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON array of expert definitions.
    """
    db = _get_db()
    experts = db.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a Kaggle competition entry using expert knowledge.

    Selects relevant experts via RAG, assembles skills, strategy,
    and generates a notebook plan.

    Args:
        competition: Kaggle competition name/slug.

    Returns:
        JSON notebook plan with experts, skills, strategy, and metrics.
    """
    db = _get_db()
    experts = db.list_experts()

    if not experts:
        chapters = db.list_chapters()
        if chapters:
            experts = register_all_experts(chapters, db=db)

    try:
        engine = _get_embedding_engine()
    except Exception:
        engine = None

    plan = build_competition_entry(competition, experts, engine, db)
    return json.dumps(plan, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The slug identifier of the expert (e.g. "08_ml_systems").
        question: The question to ask the expert.

    Returns:
        JSON with the expert's capabilities, relevant concepts, and guidance.
    """
    db = _get_db()
    expert = db.get_expert_by_slug(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    try:
        engine = _get_embedding_engine()
        hits = engine.search(question, n_results=3, chapter_id=expert.get("chapter_id"))
    except Exception:
        hits = []

    return json.dumps(
        {
            "expert": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "relevant_content": hits,
            "skills": expert.get("skills", []),
        },
        indent=2,
    )


@mcp.tool()
def get_extraction_status() -> str:
    """Check the current extraction and indexing status.

    Returns:
        JSON with chapter count, expert count, embedding count, and details.
    """
    db = _get_db()
    stats = db.get_stats()

    try:
        engine = _get_embedding_engine()
        embedding_count = engine.get_collection_count()
    except Exception:
        embedding_count = 0

    chapters = db.list_chapters()
    chapter_summary = [
        {
            "id": c["chapter_id"],
            "title": c["title"],
            "has_content": bool(c.get("content_md")),
            "concept_count": len(c.get("concepts", [])),
        }
        for c in chapters
    ]

    return json.dumps(
        {
            "chapters": stats.get("chapters", 0),
            "experts": stats.get("experts", 0),
            "embedding_chunks": embedding_count,
            "convergence_runs": stats.get("convergence_runs", 0),
            "chapter_details": chapter_summary,
        },
        indent=2,
    )


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with table counts and system info.
    """
    db = _get_db()
    stats = db.get_stats()

    try:
        engine = _get_embedding_engine()
        stats["vector_store_count"] = engine.get_collection_count()
    except Exception:
        stats["vector_store_count"] = 0

    stats["db_path"] = db.db_path
    return json.dumps(stats, indent=2)


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_context_results: int = 5,
) -> str:
    """Generate context prompt for rdagent from ML Principles.

    Args:
        competition_name: Name of the Kaggle competition.
        description: Competition description for context matching.
        n_context_results: Number of RAG results to include.

    Returns:
        JSON with rdagent context prompt and recommended approach.
    """
    db = _get_db()
    experts = db.list_experts()

    query = f"{competition_name} {description}"

    try:
        engine = _get_embedding_engine()
        hits = engine.search(query, n_results=n_context_results)
        ranked_experts = engine.infer_relevant_experts(query, experts)
    except Exception:
        hits = []
        ranked_experts = experts[:3]

    context_parts = []
    for hit in hits:
        context_parts.append(
            f"[{hit['chapter_title']}] (relevance: {hit['score']:.3f})\n{hit['text'][:500]}"
        )

    expert_advice = []
    for expert in ranked_experts[:3]:
        expert_advice.append({
            "expert": expert.get("expert_name"),
            "strategy": expert.get("strategy"),
            "skills": expert.get("skills", []),
            "relevance": expert.get("relevance_score", 0),
        })

    prompt = (
        f"Competition: {competition_name}\n"
        f"Description: {description}\n\n"
        f"ML Principles Context:\n"
        + "\n\n".join(context_parts)
        + "\n\nRecommended approach based on ML Principles experts."
    )

    return json.dumps(
        {
            "competition": competition_name,
            "context_prompt": prompt,
            "expert_recommendations": expert_advice,
            "rag_hits": len(hits),
        },
        indent=2,
    )


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
