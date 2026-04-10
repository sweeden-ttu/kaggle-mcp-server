"""FastMCP server for MLSysEng MoE system.

Provides MCP tools for knowledge extraction, expert management,
competition entry building, and convergence loop execution.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import DoclingWorker
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-mcp")

_db: Optional[Database] = None
_worker: Optional[DoclingWorker] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_worker() -> DoclingWorker:
    global _worker
    if _worker is None:
        _worker = DoclingWorker(_get_db())
    return _worker


def _get_embeddings() -> EmbeddingEngine:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingEngine(_get_db())
    return _embeddings


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


def _get_loop() -> LoopController:
    global _loop
    if _loop is None:
        _loop = LoopController(_get_db())
    return _loop


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans chapter folders, extracts PDF content using docling, generates
    embeddings, and registers expert definitions.

    Args:
        force_reindex: Force re-extraction even if chapters exist (default: False)

    Returns:
        JSON with extraction results including chapters found, extracted, and experts registered
    """
    try:
        worker = _get_worker()
        extraction = worker.extract_all(force_reindex=force_reindex)

        embeddings = _get_embeddings()
        index_result = embeddings.index_chapters(force=force_reindex)

        registry = _get_registry()
        expert_result = registry.register_from_chapters()

        registry.save_expert_definitions()

        return json.dumps({
            "status": "success",
            "extraction": extraction,
            "indexing": index_result,
            "experts": expert_result,
        }, indent=2)
    except Exception as e:
        logger.exception("extract_knowledge failed")
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Extracts knowledge if not already done, builds entries, and runs
    the state convergence loop until ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition name/slug (default: "titanic")
        max_iterations: Maximum iterations for convergence (default: 10)

    Returns:
        JSON with convergence results and competition entries
    """
    try:
        db = _get_db()
        chapters = db.list_chapters()
        if not chapters:
            worker = _get_worker()
            worker.extract_all()
            _get_embeddings().index_chapters()
            _get_registry().register_from_chapters()

        embeddings = _get_embeddings()
        expert_recs = embeddings.infer_experts(competition, n_results=5)

        experts = []
        for rec in expert_recs:
            expert = rec.get("expert", {})
            if expert:
                experts.append(expert)

        if not experts:
            experts = _get_registry().list_experts()[:5]

        initial_state = [1.0] * max(len(experts), 3)

        loop = _get_loop()
        step_fn = loop.create_step_fn(experts, competition)
        result = loop.run(
            competition=competition,
            initial_state=initial_state,
            step_fn=step_fn,
            max_iterations=max_iterations,
        )

        entries = []
        for expert in experts:
            entry = {
                "competition": competition,
                "expert_slug": expert.get("slug", "unknown"),
                "skills_used": expert.get("skills", []),
                "status": "generated",
            }
            db.upsert_entry(entry)
            entries.append(entry)

        return json.dumps({
            "status": "success",
            "competition": competition,
            "convergence": {
                "converged": result["converged"],
                "iterations": result["total_iterations"],
                "final_l2_norm": result["final_l2_norm"],
                "epsilon": result["epsilon"],
            },
            "experts_used": [e.get("expert_name", "unknown") for e in experts],
            "entries_generated": len(entries),
            "history": result["history"],
        }, indent=2)
    except Exception as e:
        logger.exception("evolve failed")
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with matching content snippets, chapters, and relevance scores
    """
    try:
        embeddings = _get_embeddings()
        results = embeddings.search(query, n_results=n_results)
        return json.dumps({
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON array of expert definitions with capabilities, skills, and strategies
    """
    try:
        registry = _get_registry()
        experts = registry.list_experts()
        return json.dumps({
            "status": "success",
            "experts": experts,
            "count": len(experts),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG.

    Infers which experts are needed, selects relevant skills, and
    generates a competition strategy.

    Args:
        competition: Competition name (default: "titanic")

    Returns:
        JSON with entry plan including experts, skills, and strategy
    """
    try:
        embeddings = _get_embeddings()
        expert_recs = embeddings.infer_experts(competition, n_results=5)

        context = embeddings.get_rdagent_context(competition)

        entry_plan = {
            "competition": competition,
            "experts": [],
            "strategy_pipeline": [],
            "context_summary": context[:2000],
        }

        for rec in expert_recs:
            expert = rec.get("expert", {})
            if not expert:
                continue

            entry_plan["experts"].append({
                "name": expert.get("expert_name", "unknown"),
                "slug": expert.get("slug", "unknown"),
                "relevance": rec.get("relevance_score", 0),
                "skills": expert.get("skills", []),
                "strategy": expert.get("strategy", ""),
                "formula": expert.get("formula", {}),
            })

        all_skills = set()
        for exp in entry_plan["experts"]:
            all_skills.update(exp["skills"])

        entry_plan["strategy_pipeline"] = [
            {"step": 1, "action": "EDA & Data Understanding", "skills": list(all_skills)[:2]},
            {"step": 2, "action": "Feature Engineering", "skills": list(all_skills)[:3]},
            {"step": 3, "action": "Baseline Model", "skills": list(all_skills)[:2]},
            {"step": 4, "action": "Model Tuning & Ensemble", "skills": list(all_skills)},
            {"step": 5, "action": "Submit & Evaluate", "skills": list(all_skills)[:1]},
        ]

        db = _get_db()
        for exp in entry_plan["experts"]:
            db.upsert_entry({
                "competition": competition,
                "expert_slug": exp["slug"],
                "skills_used": exp["skills"],
                "status": "planned",
            })

        return json.dumps({
            "status": "success",
            "entry": entry_plan,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
) -> str:
    """Prepare rdagent context with ML Principles knowledge.

    Generates a context prompt for rdagent data_science competitions
    based on the ML Principles knowledge base.

    Args:
        competition: Competition name
        description: Competition description for better context

    Returns:
        JSON with rdagent context and recommended command
    """
    try:
        embeddings = _get_embeddings()
        context = embeddings.get_rdagent_context(competition, description)

        return json.dumps({
            "status": "success",
            "competition": competition,
            "context": context,
            "suggested_command": (
                f"rdagent --competition {competition} "
                f"--context-file mlsyseng_context.md"
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems")
        question: Question to ask the expert

    Returns:
        JSON with expert response including capabilities, strategy, and relevant context
    """
    try:
        registry = _get_registry()
        expert = registry.get_expert(expert_slug)
        if not expert:
            all_experts = registry.list_experts()
            slugs = [e["slug"] for e in all_experts]
            return json.dumps({
                "status": "error",
                "error": f"Expert '{expert_slug}' not found",
                "available_experts": slugs,
            })

        embeddings = _get_embeddings()
        relevant = embeddings.search(
            question,
            n_results=3,
            filter_chapter=expert.get("expert_name"),
        )

        return json.dumps({
            "status": "success",
            "expert": expert["expert_name"],
            "capabilities": expert["capabilities"],
            "strategy": expert["strategy"],
            "formula": expert["formula"],
            "relevant_knowledge": [
                {"content": r["document"], "relevance": 1.0 - r.get("distance", 0)}
                for r in relevant
            ],
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for all chapters.

    Returns:
        JSON with extraction status for each chapter and stage
    """
    try:
        db = _get_db()
        status = db.get_extraction_status()
        chapters = db.list_chapters()

        return json.dumps({
            "status": "success",
            "extraction_log": status,
            "chapters": [
                {
                    "name": c["chapter_name"],
                    "status": c["status"],
                    "concepts_count": len(c.get("concepts", [])),
                }
                for c in chapters
            ],
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with total chapters, experts, entries, and database info
    """
    try:
        db = _get_db()
        stats = db.get_stats()
        return json.dumps({
            "status": "success",
            **stats,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def index_ml_chapters(force_reindex: bool = False) -> str:
    """Index all PDF chapters (extract + embed + store).

    Args:
        force_reindex: Force re-indexing even if already indexed

    Returns:
        JSON with indexing results
    """
    return extract_knowledge(force_reindex=force_reindex)


@mcp.tool()
def search_ml_principles(query: str, n_results: int = 5) -> str:
    """Semantic search over indexed ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with matching content from ML Principles
    """
    return search_concepts(query=query, n_results=n_results)


@mcp.tool()
def get_rdagent_context(competition_name: str, description: str = "") -> str:
    """Generate context prompt for rdagent based on ML Principles.

    Args:
        competition_name: Name of the competition
        description: Competition description

    Returns:
        JSON with context prompt for rdagent
    """
    return run_rdagent(competition=competition_name, description=description)


@mcp.tool()
def list_indexed_chapters() -> str:
    """List all indexed chapters from the knowledge base.

    Returns:
        JSON with list of indexed chapters and their concepts
    """
    try:
        db = _get_db()
        chapters = db.list_chapters()
        return json.dumps({
            "status": "success",
            "chapters": [
                {
                    "name": c["chapter_name"],
                    "status": c["status"],
                    "concepts": c.get("concepts", []),
                    "has_content": bool(c.get("markdown_content")),
                }
                for c in chapters
            ],
            "count": len(chapters),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def get_indexing_stats() -> str:
    """Get indexing statistics for the knowledge base.

    Returns:
        JSON with indexing statistics
    """
    return get_stats()


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
