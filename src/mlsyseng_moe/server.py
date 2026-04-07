"""FastMCP server for the MLSysEng MoE system.

Provides tools for knowledge extraction, expert management,
competition entry building, and state convergence loops.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import run_extraction
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingStore] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
        _db.initialize()
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


def _get_loop() -> LoopController:
    global _loop
    if _loop is None:
        _loop = LoopController(_get_db())
    return _loop


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract even previously indexed chapters.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()
    results = run_extraction(db, force=force_reindex)

    registry = _get_registry()
    experts = registry.register_from_chapters()

    try:
        store = _get_embeddings()
        index_results = store.index_all_chapters(db)
    except ImportError as e:
        index_results = {"error": str(e)}

    return json.dumps({
        "extraction": results,
        "experts_created": len(experts),
        "indexing": index_results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the state convergence loop for a competition.

    Iterates experts, collects outputs as state vectors, and converges
    when state changes fall below epsilon for patience consecutive iterations.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon

    Args:
        competition: The Kaggle competition slug.

    Returns:
        JSON with convergence results.
    """
    registry = _get_registry()
    experts = registry.list_experts()

    if not experts:
        db = _get_db()
        results = run_extraction(db)
        experts = registry.register_from_chapters()

    loop = _get_loop()
    result = loop.run(competition=competition, experts=experts)

    return json.dumps(result, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query string.
        n_results: Maximum number of results to return.

    Returns:
        JSON array of matching content with similarity scores.
    """
    try:
        store = _get_embeddings()
        hits = store.search(query, n_results=n_results)
        return json.dumps(hits, indent=2)
    except ImportError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON array of expert definitions with skills, strategy, and formulas.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    summary = []
    for e in experts:
        summary.append({
            "expert_name": e.get("expert_name"),
            "title": e.get("title"),
            "capabilities": e.get("capabilities", []),
            "skill_names": e.get("skill_names", []),
            "strategy": e.get("strategy"),
            "formula": e.get("formula"),
            "loop_config": e.get("loop_config"),
            "concepts_count": len(e.get("concepts", [])),
        })
    return json.dumps(summary, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Uses RAG-informed skill selection to identify relevant experts,
    then generates a competition strategy with recommended skills.

    Args:
        competition: The Kaggle competition slug.

    Returns:
        JSON with the entry plan including experts, skills, and strategy.
    """
    registry = _get_registry()
    experts = registry.list_experts()

    if not experts:
        return json.dumps({"error": "No experts registered. Run extract-knowledge first."})

    try:
        store = _get_embeddings()
        relevance = store.infer_skills(competition)
        matched_experts = registry.find_experts_for_competition(
            competition, relevance
        )
    except ImportError:
        relevance = []
        matched_experts = experts

    all_skills = set()
    all_metrics = set()
    for e in matched_experts:
        all_skills.update(e.get("skill_names", []))
        formula = e.get("formula", {})
        all_metrics.update(formula.get("metrics", []))

    return json.dumps({
        "competition": competition,
        "experts_selected": len(matched_experts),
        "experts": [
            {
                "name": e.get("expert_name"),
                "title": e.get("title"),
                "capabilities": e.get("capabilities", []),
            }
            for e in matched_experts
        ],
        "recommended_skills": sorted(all_skills),
        "metrics": sorted(all_metrics),
        "strategy": "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit",
        "relevance_scores": relevance[:5] if relevance else [],
    }, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context.

    Generates contextual guidance from expert knowledge
    for the rdagent data_science competition workflow.

    Args:
        competition: The Kaggle competition slug.
        description: Competition description for better context.

    Returns:
        JSON with rdagent context and recommended commands.
    """
    registry = _get_registry()
    experts = registry.list_experts()

    context_parts = []
    skills_list = set()

    for e in experts:
        context_parts.append(
            f"- {e.get('title', 'Unknown')}: {', '.join(e.get('capabilities', [])[:3])}"
        )
        skills_list.update(e.get("skill_names", []))

    ml_context = "\n".join(context_parts) if context_parts else "No ML context available."

    return json.dumps({
        "competition": competition,
        "ml_context": ml_context,
        "recommended_skills": sorted(skills_list),
        "rdagent_command": f"rdagent --competition {competition} --context ml_principles",
        "strategy": "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit",
        "notes": "Use extract-knowledge first to populate ML Principles context.",
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the extraction progress for all chapters.

    Returns:
        JSON array of extraction status per chapter.
    """
    db = _get_db()
    statuses = db.get_extraction_status()
    return json.dumps(statuses, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with counts of chapters, experts, embeddings, and loop states.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        store = _get_embeddings()
        embed_stats = store.get_stats()
    except ImportError:
        embed_stats = {"total_chunks": 0, "note": "embedding libraries not installed"}

    return json.dumps({
        "database": db_stats,
        "embeddings": embed_stats,
    }, indent=2)


@mcp.tool()
def ask_expert(expert_name: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_name: The expert identifier (e.g., '08_ml_systems').
        question: The question to ask the expert.

    Returns:
        JSON with expert's capabilities, strategy, and relevant concepts.
    """
    registry = _get_registry()
    defn = registry.get_expert(expert_name)

    if not defn:
        all_experts = registry.list_experts()
        names = [e.get("expert_name", "") for e in all_experts]
        for n in names:
            if expert_name.lower() in n.lower():
                defn = registry.get_expert(n)
                break

    if not defn:
        return json.dumps({
            "error": f"Expert '{expert_name}' not found",
            "available_experts": [e.get("expert_name") for e in registry.list_experts()],
        })

    return json.dumps({
        "expert_name": defn.get("expert_name"),
        "title": defn.get("title"),
        "question": question,
        "capabilities": defn.get("capabilities", []),
        "strategy": defn.get("strategy"),
        "formula": defn.get("formula"),
        "recommended_skills": defn.get("skill_names", []),
        "concepts": defn.get("concepts", []),
        "answer_context": (
            f"As an expert in {defn.get('title', 'ML')}, "
            f"I recommend focusing on: {', '.join(defn.get('capabilities', [])[:3])}. "
            f"Key skills: {', '.join(defn.get('skill_names', [])[:5])}."
        ),
    }, indent=2)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
