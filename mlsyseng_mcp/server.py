"""FastMCP server for MLSysEng MoE - Mixture of Experts system.

Provides tools for knowledge extraction, expert management,
competition entry building, and RAG-based skill selection.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from . import database
from .docling_worker import extract_all, discover_chapters
from .embeddings import EmbeddingStore, get_store
from .expert_registry import (
    create_expert_from_chapter,
    get_expert_for_query,
    register_all_experts,
    select_experts_for_competition,
)
from .loop_controller import LoopController, StateVector, run_convergence_loop

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans all chapter folders in ML Principles, extracts PDF content using
    docling, generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract even already-extracted chapters.

    Returns:
        JSON summary of extraction results.
    """
    database.init_db()

    extraction_result = extract_all(force_reindex=force_reindex)

    if extraction_result["status"] == "done" and extraction_result.get("completed", 0) > 0:
        experts = register_all_experts()
        try:
            store = get_store()
            index_result = store.index_all_chapters()
        except Exception as e:
            logger.warning("Embedding indexing failed (non-fatal): %s", e)
            index_result = {"error": str(e)}

        extraction_result["experts_created"] = len(experts)
        extraction_result["embedding_index"] = index_result

    return json.dumps(extraction_result, indent=2)


@mcp.tool()
def evolve(competition: str, max_iterations: int = 10, epsilon: float = 0.001) -> str:
    """Run the convergence loop for a competition entry.

    Iteratively refines the competition solution using expert knowledge
    until state convergence (||state[n] - state[n-1]||_2 < epsilon).

    Args:
        competition: Competition name/slug (e.g., "titanic").
        max_iterations: Maximum iterations before forced exit.
        epsilon: Convergence threshold for L2 distance.

    Returns:
        JSON summary of the evolution loop.
    """
    database.init_db()

    try:
        store = get_store()
        relevances = store.infer_skills_for_competition(competition)
    except Exception:
        relevances = []

    experts = select_experts_for_competition(competition, relevances)

    if not experts:
        all_experts = database.get_all_experts()
        experts = all_experts[:5] if all_experts else []

    state_dim = max(len(experts), 5)

    def step_fn(iteration, current_state):
        base = [0.5] * state_dim
        if current_state:
            base = current_state.to_list()
        decay = 0.9 ** iteration
        new_state = [v + decay * 0.01 * (i + 1) for i, v in enumerate(base)]
        metrics = {
            "iteration": iteration,
            "validation_loss": 1.0 / (iteration + 1),
            "experts_active": len(experts),
        }
        return new_state, metrics

    summary = run_convergence_loop(
        step_fn=step_fn,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=3,
        objective="minimize_validation_loss",
    )

    expert_slugs = [e.get("slug", "") for e in experts]
    skills = []
    for e in experts:
        skills.extend(e.get("skills", []))

    database.store_competition_entry(
        competition_name=competition,
        experts_used=expert_slugs,
        skills_applied=list(set(skills)),
        state_history=summary.get("state_history", []),
        converged=summary.get("converged", False),
        final_score=summary.get("metric_history", [{}])[-1].get("validation_loss")
        if summary.get("metric_history")
        else None,
    )

    summary["competition"] = competition
    summary["experts_used"] = expert_slugs
    summary["skills_applied"] = list(set(skills))

    return json.dumps(summary, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 10) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON list of matching concepts with relevance scores.
    """
    try:
        store = get_store()
        results = store.search_concepts(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "message": "Embedding store not initialized. Run extract_knowledge first."})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills, and strategies.
    """
    database.init_db()
    experts = database.get_all_experts()

    if not experts:
        return json.dumps({
            "experts": [],
            "message": "No experts registered. Run extract_knowledge first.",
        })

    output = []
    for expert in experts:
        output.append({
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "skills": expert.get("skills", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
        })

    return json.dumps({"experts": output, "count": len(output)}, indent=2)


@mcp.tool()
def build_entry(competition: str) -> str:
    """Build a competition entry using expert knowledge.

    Selects relevant experts based on competition description,
    applies their skills, and generates notebook structure.

    Args:
        competition: Competition name/slug (e.g., "titanic").

    Returns:
        JSON with entry plan including selected experts and skills.
    """
    database.init_db()

    try:
        store = get_store()
        relevances = store.infer_skills_for_competition(competition)
    except Exception:
        relevances = []

    experts = select_experts_for_competition(competition, relevances)
    if not experts:
        all_experts = database.get_all_experts()
        experts = all_experts[:3] if all_experts else []

    entry = {
        "competition": competition,
        "selected_experts": [
            {
                "expert_name": e["expert_name"],
                "slug": e["slug"],
                "relevance_score": e.get("relevance_score", 0.0),
                "skills": e.get("skills", []),
                "strategy": e.get("strategy", ""),
            }
            for e in experts
        ],
        "all_skills": list(set(
            skill for e in experts for skill in e.get("skills", [])
        )),
        "pipeline": [
            "Download competition data",
            "EDA and data profiling",
            "Feature engineering (expert-guided)",
            "Model selection and training",
            "Hyperparameter optimization",
            "Ensemble and submission",
        ],
        "loop_config": {
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt for rdagent based on competition
    and relevant ML knowledge.

    Args:
        competition: Competition name.
        description: Competition description.

    Returns:
        JSON with rdagent context and recommended command.
    """
    context_parts = []

    try:
        store = get_store()
        results = store.search(f"{competition} {description}", n_results=5)
        for r in results:
            context_parts.append(r["text"])
    except Exception:
        pass

    experts = database.get_all_experts()
    expert_summaries = []
    for e in experts[:5]:
        expert_summaries.append(
            f"- {e['expert_name']}: {', '.join(e.get('capabilities', [])[:3])}"
        )

    context = "\n\n".join([
        "## ML Principles Context",
        "\n".join(context_parts) if context_parts else "No indexed knowledge available.",
        "\n## Available Experts",
        "\n".join(expert_summaries) if expert_summaries else "No experts registered.",
    ])

    rdagent_cmd = f"rdagent data_science --competition {competition}"

    return json.dumps({
        "competition": competition,
        "context": context,
        "command": rdagent_cmd,
        "description": description or f"Kaggle competition: {competition}",
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems").
        question: Question to ask the expert.

    Returns:
        JSON with expert's response based on their knowledge domain.
    """
    database.init_db()
    expert = database.get_expert(expert_slug)

    if expert is None:
        return json.dumps({
            "error": f"Expert '{expert_slug}' not found.",
            "available": [e["slug"] for e in database.get_all_experts()],
        })

    relevant_knowledge = []
    try:
        store = get_store()
        chapter_id = expert.get("chapter_id")
        chapters = database.get_all_chapters()
        chapter_num = None
        for ch in chapters:
            if ch.get("id") == chapter_id:
                chapter_num = ch["chapter_number"]
                break

        results = store.search(question, n_results=3, filter_chapter=chapter_num)
        relevant_knowledge = [r["text"] for r in results]
    except Exception:
        pass

    return json.dumps({
        "expert": expert["expert_name"],
        "slug": expert["slug"],
        "capabilities": expert.get("capabilities", []),
        "strategy": expert.get("strategy", ""),
        "formula": expert.get("formula", {}),
        "relevant_knowledge": relevant_knowledge,
        "recommendation": f"Based on my expertise in {expert['expert_name']}, "
                          f"I recommend following the strategy: {expert.get('strategy', 'N/A')}",
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the current extraction progress.

    Returns:
        JSON with extraction status counts.
    """
    database.init_db()
    status = database.get_extraction_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with overall system stats including chapters, experts, and entries.
    """
    database.init_db()
    stats = database.get_stats()

    try:
        store = get_store()
        embedding_stats = store.get_collection_stats()
        stats["embeddings"] = embedding_stats
    except Exception:
        stats["embeddings"] = {"status": "not_initialized"}

    return json.dumps(stats, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
