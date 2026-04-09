"""FastMCP server for MLSysEng MoE system.

Exposes tools for knowledge extraction, expert management, semantic search,
competition entry building, and convergence loops.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all_chapters, extract_concepts, scan_chapters
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
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


def _get_loop() -> LoopController:
    global _loop
    if _loop is None:
        _loop = LoopController(_get_db(), _get_registry())
    return _loop


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans the ML Principles chapter directory, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Force re-extraction even if chapters already exist.

    Returns:
        JSON string with extraction results.
    """
    db = _get_db()
    chapters_path = os.environ.get(
        "ML_PRINCIPLES_PATH",
        os.path.expanduser("~/ml-principles-chapters"),
    )

    extraction_results = extract_all_chapters(chapters_path, db, force=force_reindex)

    registry = _get_registry()
    experts = registry.create_all_experts()

    try:
        embeddings = _get_embeddings()
        embedding_stats = embeddings.index_all_chapters(db)
    except ImportError:
        embedding_stats = {"error": "chromadb not installed"}
    except Exception as e:
        embedding_stats = {"error": str(e)}

    return json.dumps({
        "extraction": extraction_results,
        "experts_created": len(experts),
        "embedding_stats": embedding_stats,
        "expert_names": [e["expert_name"] for e in experts],
    }, indent=2)


@mcp.tool()
def evolve(
    competition: str,
    description: str = "",
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> str:
    """Run the convergence loop for a competition.

    Iteratively selects and scores experts until the state converges
    (L2 norm of state difference < epsilon).

    Args:
        competition: Competition name/slug (e.g., "titanic").
        description: Competition description for better skill inference.
        epsilon: Convergence threshold.
        max_iterations: Maximum number of iterations.
        patience: Consecutive converging iterations before exit.

    Returns:
        JSON string with convergence results and top experts.
    """
    db = _get_db()
    registry = _get_registry()
    loop = LoopController(
        db=db, registry=registry,
        epsilon=epsilon, max_iterations=max_iterations, patience=patience,
    )
    result = loop.run(competition, description)
    return json.dumps(result, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles content.

    Uses embeddings and ChromaDB for vector similarity search.
    Falls back to database text search if embeddings are unavailable.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON string with search results including content and relevance scores.
    """
    try:
        embeddings = _get_embeddings()
        results = embeddings.search(query, n_results=n_results)
        if results:
            return json.dumps(results, indent=2)
    except Exception as e:
        logger.warning("Embedding search failed, falling back to DB: %s", e)

    db = _get_db()
    db_results = db.search_concepts(query)
    return json.dumps([
        {
            "concept_name": r["concept_name"],
            "description": r.get("description", ""),
            "chapter": r.get("chapter_name", ""),
            "category": r.get("category", ""),
        }
        for r in db_results
    ], indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON string with expert definitions including capabilities, skills, and strategies.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps([
        {
            "expert_name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e.get("capabilities", []),
            "skills": e.get("skills", []),
            "strategy": e.get("strategy", ""),
            "formula": e.get("formula", {}),
        }
        for e in experts
    ], indent=2)


@mcp.tool()
def build_entry(competition: str, description: str = "") -> str:
    """Build a competition entry using expert knowledge.

    Selects relevant experts, combines their strategies, and produces
    a structured competition entry plan.

    Args:
        competition: Competition name/slug (e.g., "titanic").
        description: Competition description for better skill matching.

    Returns:
        JSON string with entry plan including experts, skills, and strategy.
    """
    registry = _get_registry()
    db = _get_db()

    recommended = registry.recommend_experts_for_competition(
        f"{competition} {description}"
    )

    try:
        embeddings = _get_embeddings()
        rag_results = embeddings.infer_skills_for_competition(
            f"{competition} {description}", db=db
        )
        if rag_results and not recommended:
            recommended = rag_results
    except Exception as e:
        logger.warning("RAG inference failed: %s", e)

    all_skills = set()
    strategies = []
    for rec in recommended[:5]:
        all_skills.update(rec.get("skills", []))
        if rec.get("strategy"):
            strategies.append(rec["strategy"])

    entry = {
        "competition": competition,
        "description": description,
        "selected_experts": recommended[:5],
        "combined_skills": sorted(all_skills),
        "strategy": strategies[0] if strategies else "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit",
        "entry_plan": {
            "step_1": "Download and explore competition data",
            "step_2": "Apply preprocessor skills for data cleaning",
            "step_3": "Feature engineering based on expert recommendations",
            "step_4": "Model selection and training using expert strategies",
            "step_5": "Validation using convergence loop",
            "step_6": "Generate submission",
        },
    }
    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(
    competition: str,
    description: str = "",
    n_context_results: int = 5,
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt for rdagent data_science competitions
    based on relevant ML Principles content.

    Args:
        competition: Competition name.
        description: Competition description.
        n_context_results: Number of context results to include.

    Returns:
        JSON string with rdagent context and recommended command.
    """
    context_parts = []

    try:
        embeddings = _get_embeddings()
        results = embeddings.search(
            f"{competition} {description}",
            n_results=n_context_results,
        )
        for r in results:
            context_parts.append(r["content"])
    except Exception:
        pass

    db = _get_db()
    concepts = db.search_concepts(competition)
    for c in concepts[:5]:
        context_parts.append(f"Concept: {c['concept_name']} - {c.get('description', '')}")

    context = "\n\n".join(context_parts) if context_parts else "No ML Principles context available."

    return json.dumps({
        "competition": competition,
        "context": context,
        "recommended_command": f"rdagent --competition {competition} --context-file ml_context.txt",
        "context_sources": len(context_parts),
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems").
        question: Question to ask the expert.

    Returns:
        JSON string with expert response including relevant concepts and strategies.
    """
    registry = _get_registry()
    result = registry.query_expert(expert_slug, question)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for all chapters.

    Returns:
        JSON string with extraction status for each chapter.
    """
    db = _get_db()
    status = db.get_extraction_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON string with counts of chapters, concepts, experts,
        and extraction status summary.
    """
    db = _get_db()
    stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        stats["embedding_stats"] = embeddings.get_collection_stats()
    except Exception:
        stats["embedding_stats"] = {"status": "unavailable"}

    return json.dumps(stats, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
