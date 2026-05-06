"""FastMCP server for the MLSysEng MoE system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MLSysEngDatabase
from .docling_worker import discover_chapters, extract_all_chapters, extract_chapter
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import CompetitionLoopController, LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MLSysEngDatabase] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None


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


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


# --- Knowledge Extraction Tools ---


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract PDFs, index chapters, create experts.

    Scans all chapter folders in ML Principles, extracts PDF content using
    docling, generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON with extraction results including chapters processed and experts created.
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    extraction_result = extract_all_chapters(db, force=force_reindex)

    try:
        indexing_result = embeddings.index_all_chapters(db)
    except ImportError as e:
        indexing_result = {"status": "skipped", "reason": str(e)}

    expert_result = registry.create_experts_from_all_chapters()

    return json.dumps(
        {
            "status": "complete",
            "extraction": {
                "total": extraction_result.get("total", 0),
                "extracted": extraction_result.get("extracted", 0),
                "skipped": extraction_result.get("skipped", 0),
                "failed": extraction_result.get("failed", 0),
            },
            "indexing": indexing_result,
            "experts": {
                "created": expert_result.get("experts_created", 0),
            },
        },
        indent=2,
    )


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """
    Run the state convergence loop for a competition using expert knowledge.

    Identifies relevant experts via RAG, then runs an iterative convergence
    loop until ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition name/description to optimize for.
        max_iterations: Maximum loop iterations.

    Returns:
        JSON with convergence results including final state and expert weights.
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    try:
        inference = embeddings.infer_skills_for_competition(competition)
        relevant_chapters = inference.get("relevant_chapters", [])
    except Exception:
        relevant_chapters = []

    if relevant_chapters:
        experts = registry.get_experts_for_competition(relevant_chapters)
    else:
        experts = registry.list_experts()

    if not experts:
        return json.dumps(
            {
                "status": "no_experts",
                "message": "No experts available. Run extract_knowledge first.",
            }
        )

    controller = CompetitionLoopController(
        db=db,
        experts=experts,
        max_iterations=max_iterations,
    )
    result = controller.run_competition_loop()

    expert_summary = []
    for i, expert in enumerate(experts):
        weight = result["final_state"][i] if i < len(result["final_state"]) else 0.0
        expert_summary.append(
            {
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "final_weight": round(weight, 6),
                "skills": expert.get("skills", []),
            }
        )

    return json.dumps(
        {
            "status": "converged" if result["converged"] else "max_iterations_reached",
            "competition": competition,
            "session_id": result["session_id"],
            "total_iterations": result["total_iterations"],
            "final_l2_norm": result["final_l2_norm"],
            "epsilon": result["epsilon"],
            "expert_weights": expert_summary,
        },
        indent=2,
    )


# --- Search & Retrieval Tools ---


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON with matching content snippets, chapter names, and relevance scores.
    """
    embeddings = _get_embeddings()

    try:
        results = embeddings.search(query, n_results=n_results)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

    return json.dumps(
        {
            "status": "success",
            "query": query,
            "results": [
                {
                    "content": r["content"][:300],
                    "chapter": r["chapter_name"],
                    "relevance": round(r.get("relevance", 0.0), 4),
                }
                for r in results
            ],
        },
        indent=2,
    )


# --- Expert Management Tools ---


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts with their capabilities.

    Returns:
        JSON array of expert definitions including name, skills, strategy, and formula.
    """
    registry = _get_registry()
    experts = registry.list_experts()

    return json.dumps(
        {
            "status": "success",
            "total": len(experts),
            "experts": [
                {
                    "expert_name": e["expert_name"],
                    "slug": e["slug"],
                    "capabilities": e.get("capabilities", []),
                    "skills": e.get("skills", []),
                    "strategy": e.get("strategy", ""),
                    "formula": e.get("formula", {}),
                }
                for e in experts
            ],
        },
        indent=2,
    )


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """
    Query a specific chapter expert.

    Args:
        expert_slug: Expert slug (e.g., "08_ml_systems").
        question: Question to ask the expert.

    Returns:
        JSON with expert's context, capabilities, and relevant knowledge.
    """
    registry = _get_registry()
    embeddings = _get_embeddings()
    expert = registry.get_expert(expert_slug)

    if not expert:
        return json.dumps(
            {"status": "error", "message": f"Expert '{expert_slug}' not found"}
        )

    try:
        search_results = embeddings.search(
            question, n_results=3, chapter_filter=expert["expert_name"]
        )
    except Exception:
        search_results = []

    return json.dumps(
        {
            "status": "success",
            "expert": {
                "name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert.get("capabilities", []),
                "strategy": expert.get("strategy", ""),
                "formula": expert.get("formula", {}),
            },
            "relevant_knowledge": [
                {"content": r["content"][:500], "relevance": round(r.get("relevance", 0.0), 4)}
                for r in search_results
            ],
            "question": question,
        },
        indent=2,
    )


# --- Competition Entry Tools ---


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a competition entry using expert knowledge and RAG-informed skill selection.

    Identifies relevant experts, selects skills, and generates an entry plan
    with notebook structure.

    Args:
        competition: Competition name or description.

    Returns:
        JSON with entry plan including expert assignments, skills, and notebook structure.
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    try:
        inference = embeddings.infer_skills_for_competition(competition)
        relevant_chapters = inference.get("relevant_chapters", [])
    except Exception:
        relevant_chapters = []

    if relevant_chapters:
        experts = registry.get_experts_for_competition(relevant_chapters)
    else:
        experts = registry.list_experts()

    if not experts:
        return json.dumps(
            {
                "status": "no_experts",
                "message": "No experts found. Run extract_knowledge first.",
            }
        )

    all_skills = set()
    expert_assignments = []
    for expert in experts:
        skills = expert.get("skills", [])
        all_skills.update(skills)
        expert_assignments.append(
            {
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "relevance": round(expert.get("relevance", 0.5), 4),
                "skills": skills,
                "strategy": expert.get("strategy", ""),
            }
        )

    notebooks = []
    for expert in experts[:5]:
        slug = expert["slug"]
        notebooks.append(
            {
                "filename": f"Expert_{slug}.ipynb",
                "path": f"~/{competition}/Expert_{slug}.ipynb",
                "expert": expert["expert_name"],
                "sections": [
                    "Data Loading & EDA",
                    "Feature Engineering",
                    "Model Training",
                    "Validation",
                    "Submission",
                ],
            }
        )

    return json.dumps(
        {
            "status": "success",
            "competition": competition,
            "expert_assignments": expert_assignments,
            "total_skills": len(all_skills),
            "skills": sorted(all_skills),
            "notebooks": notebooks,
        },
        indent=2,
    )


# --- RDAgent Integration ---


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """
    Run rdagent with ML Principles context for a competition.

    Args:
        competition: Competition name.
        description: Optional competition description for better context.

    Returns:
        JSON with rdagent context prompt and recommended configuration.
    """
    embeddings = _get_embeddings()

    query = f"{competition} {description}".strip()
    try:
        results = embeddings.search(query, n_results=5)
        context_snippets = [r["content"][:200] for r in results]
    except Exception:
        context_snippets = []

    context_prompt = (
        f"Competition: {competition}\n"
        f"Description: {description}\n\n"
        "ML Principles Context:\n"
    )
    for i, snippet in enumerate(context_snippets, 1):
        context_prompt += f"\n{i}. {snippet}\n"

    return json.dumps(
        {
            "status": "success",
            "competition": competition,
            "context_prompt": context_prompt,
            "rdagent_config": {
                "competition": competition,
                "scenario": "data_science",
                "context_window": len(context_snippets),
            },
        },
        indent=2,
    )


# --- Status & Stats Tools ---


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check the extraction progress for all chapters.

    Returns:
        JSON with extraction status per chapter and overall progress.
    """
    db = _get_db()
    chapters = db.list_chapters()

    status_counts = {}
    for ch in chapters:
        s = ch.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    return json.dumps(
        {
            "status": "success",
            "total_chapters": len(chapters),
            "status_breakdown": status_counts,
            "chapters": [
                {
                    "name": ch["chapter_name"],
                    "status": ch.get("status", "unknown"),
                    "concepts_count": len(ch.get("concepts", [])),
                    "extracted_at": ch.get("extracted_at"),
                }
                for ch in chapters
            ],
        },
        indent=2,
    )


@mcp.tool()
def get_stats() -> str:
    """
    Get system statistics including chapters, experts, and embeddings.

    Returns:
        JSON with comprehensive system statistics.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        emb_stats = embeddings.get_stats()
    except Exception:
        emb_stats = {"status": "unavailable"}

    return json.dumps(
        {
            "status": "success",
            "database": db_stats,
            "embeddings": emb_stats,
        },
        indent=2,
    )


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
