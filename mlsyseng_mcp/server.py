"""FastMCP server for MLSysEng MoE - Mixture of Experts system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all_chapters, scan_chapter_folders
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import ConvergenceLoop, default_step_fn

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
        _registry = ExpertRegistry(db=_get_db())
    return _registry


# ── Knowledge Extraction ──────────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders in ML Principles, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Force re-extraction even if already indexed (default: False)

    Returns:
        JSON with extraction results including chapters processed and experts created.
    """
    db = _get_db()

    results = extract_all_chapters(db=db, force=force_reindex)

    extracted_count = sum(1 for r in results if r.get("status") == "extracted")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    failed_count = sum(1 for r in results if r.get("status") == "failed")

    embed_stats = {"indexed": 0, "chunks": 0, "skipped": 0}
    if extracted_count > 0:
        try:
            store = _get_embeddings()
            embed_stats = store.index_all_chapters(db)
        except ImportError as e:
            embed_stats = {"error": str(e)}

    registry = _get_registry()
    experts = registry.create_experts_from_db()

    return json.dumps({
        "status": "completed",
        "extraction": {
            "extracted": extracted_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "total": len(results),
        },
        "embeddings": embed_stats,
        "experts_created": len(experts),
        "expert_names": [e["expert_name"] for e in experts],
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """
    Run the convergence loop for a competition, evolving expert selection.

    Extracts knowledge if needed, then runs the state convergence loop
    with exit condition ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition name/slug (default: "titanic")
        max_iterations: Maximum number of iterations (default: 10)

    Returns:
        JSON with convergence results.
    """
    db = _get_db()
    registry = _get_registry()

    experts = registry.list_experts()
    if not experts:
        extract_knowledge(force_reindex=False)
        experts = registry.list_experts()

    if not experts:
        return json.dumps({
            "status": "error",
            "message": "No experts available. Ensure ML Principles PDFs are accessible.",
        })

    try:
        store = _get_embeddings()
        recommendations = store.infer_skills(competition)
        matched = registry.get_expert_for_competition(competition, recommendations)
        if matched:
            experts = matched
    except (ImportError, Exception) as e:
        logger.warning("Could not use RAG for expert selection: %s", e)

    loop = ConvergenceLoop(
        competition=competition,
        max_iterations=max_iterations,
        db=db,
    )

    step_fn = default_step_fn(experts, competition)
    result = loop.run(step_fn)

    return json.dumps(result, indent=2)


# ── Search & Query ────────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with matching content, chapters, and similarity scores.
    """
    try:
        store = _get_embeddings()
        hits = store.search(query, n_results=n_results)
        return json.dumps({
            "query": query,
            "results": hits,
            "count": len(hits),
        }, indent=2)
    except ImportError as e:
        return json.dumps({
            "status": "error",
            "message": f"Embedding dependencies not available: {e}",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
        })


# ── Expert Management ─────────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts with their capabilities, skills, and strategies.

    Returns:
        JSON array of expert definitions.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """
    Query a specific chapter expert for advice on a topic.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems")
        question: Question to ask the expert

    Returns:
        JSON with expert's capabilities, relevant skills, and strategy context.
    """
    registry = _get_registry()
    expert = registry.get_expert(expert_slug)

    if not expert:
        available = registry.list_experts()
        slugs = [e["slug"] for e in available]
        return json.dumps({
            "status": "error",
            "message": f"Expert '{expert_slug}' not found",
            "available_experts": slugs,
        })

    try:
        store = _get_embeddings()
        context = store.search(question, n_results=3, chapter_filter=expert["expert_name"])
    except Exception:
        context = []

    return json.dumps({
        "expert": expert["expert_name"],
        "slug": expert["slug"],
        "capabilities": expert.get("capabilities", []),
        "skills": expert.get("skills", []),
        "strategy": expert.get("strategy", ""),
        "formula": expert.get("formula", {}),
        "relevant_context": context,
        "question": question,
    }, indent=2)


# ── Competition Entry ─────────────────────────────────────────────────


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a Kaggle competition entry using expert knowledge and RAG-informed
    skill selection.

    Args:
        competition: Competition name/slug (default: "titanic")

    Returns:
        JSON with selected experts, skills, strategy, and notebook structure.
    """
    db = _get_db()
    registry = _get_registry()

    experts = registry.list_experts()
    if not experts:
        extract_knowledge(force_reindex=False)
        experts = registry.list_experts()

    selected_experts = experts
    try:
        store = _get_embeddings()
        recommendations = store.infer_skills(competition)
        matched = registry.get_expert_for_competition(competition, recommendations)
        if matched:
            selected_experts = matched
    except Exception:
        pass

    all_skills = set()
    all_capabilities = []
    for exp in selected_experts:
        for skill in exp.get("skills", []):
            all_skills.add(skill)
        for cap in exp.get("capabilities", []):
            if cap not in all_capabilities:
                all_capabilities.append(cap)

    strategy_steps = (
        selected_experts[0].get("strategy", "").split(" → ")
        if selected_experts
        else ["Baseline", "EDA", "Feature Engineering", "Model Selection", "Submit"]
    )

    notebook_cells = []
    for exp in selected_experts:
        notebook_cells.append({
            "expert": exp["expert_name"],
            "cell_type": "strategy",
            "content": f"# Expert: {exp['expert_name']}\n"
                       f"## Strategy: {exp.get('strategy', '')}\n"
                       f"## Capabilities: {', '.join(exp.get('capabilities', []))}\n",
        })

    output_dir = os.path.expanduser(f"~/{competition}")

    return json.dumps({
        "competition": competition,
        "experts_selected": len(selected_experts),
        "experts": [
            {
                "name": e["expert_name"],
                "slug": e["slug"],
                "relevance": e.get("relevance_score", 1.0),
            }
            for e in selected_experts
        ],
        "skills": sorted(all_skills),
        "capabilities": all_capabilities,
        "strategy_steps": strategy_steps,
        "notebook_structure": notebook_cells,
        "output_dir": output_dir,
    }, indent=2)


# ── RDAgent Integration ──────────────────────────────────────────────


@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
) -> str:
    """
    Prepare rdagent command with ML Principles context for a competition.

    Generates a context prompt enriched with expert knowledge for rdagent
    data_science competition runs.

    Args:
        competition: Competition name (default: "titanic")
        description: Competition description for context enrichment

    Returns:
        JSON with rdagent command and ML context.
    """
    registry = _get_registry()
    experts = registry.list_experts()

    context_parts = [f"Competition: {competition}"]
    if description:
        context_parts.append(f"Description: {description}")

    try:
        store = _get_embeddings()
        query = f"{competition} {description}"
        hits = store.search(query, n_results=5)
        for hit in hits:
            context_parts.append(
                f"[{hit['chapter']}] {hit['content'][:200]}"
            )
    except Exception:
        pass

    expert_context = []
    for exp in experts[:5]:
        expert_context.append({
            "expert": exp["expert_name"],
            "capabilities": exp.get("capabilities", []),
            "strategy": exp.get("strategy", ""),
        })

    context_prompt = "\n".join(context_parts)

    return json.dumps({
        "competition": competition,
        "command": f"rdagent data_science --competition {competition}",
        "context_prompt": context_prompt,
        "expert_guidance": expert_context,
        "ml_principles_indexed": len(experts) > 0,
    }, indent=2)


# ── Status & Statistics ──────────────────────────────────────────────


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check the status of PDF extraction progress.

    Returns:
        JSON with extraction statistics (total, extracted, pending, failed).
    """
    db = _get_db()
    status = db.get_extraction_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get overall system statistics including chapters, experts, and competitions.

    Returns:
        JSON with system statistics.
    """
    db = _get_db()
    stats = db.get_stats()
    return json.dumps(stats, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
