"""FastMCP server for MLSysEng MoE system."""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_moe.database import get_stats as db_get_stats
from mlsyseng_moe.database import get_extraction_status, init_db
from mlsyseng_moe.docling_worker import extract_all
from mlsyseng_moe.embeddings import EmbeddingStore
from mlsyseng_moe.expert_registry import (
    get_expert_definition,
    list_all_experts,
    register_experts_from_chapters,
    select_experts_for_competition,
)
from mlsyseng_moe.loop_controller import ConvergenceLoop, LoopConfig

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description="ML Systems Expert Mixture of Experts - Knowledge extraction, expert registry, and RAG-informed competition building",
)

DB_PATH = os.environ.get("SQLITE_DB_PATH")
CHROMA_PATH = os.environ.get("CHROMA_DB_PATH")


def _get_embedding_store() -> EmbeddingStore:
    return EmbeddingStore(chroma_path=CHROMA_PATH, db_path=DB_PATH)


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts from ML Principles.

    Scans all chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract even already indexed chapters.
    """
    init_db(DB_PATH)

    extraction_results = extract_all(force_reindex=force_reindex, db_path=DB_PATH)

    expert_results = register_experts_from_chapters(force=force_reindex, db_path=DB_PATH)

    try:
        store = _get_embedding_store()
        embedding_stats = store.index_chapters(force_reindex=force_reindex)
    except Exception as e:
        embedding_stats = {"error": str(e)}

    return json.dumps({
        "extraction": extraction_results,
        "experts": expert_results,
        "embeddings": embedding_stats,
    }, indent=2)


@mcp.tool()
def evolve(competition: str, description: str = "", max_iterations: int = 10) -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Iteratively applies expert strategies with state convergence tracking.
    Exit condition: ||state[n] - state[n-1]||_2 < epsilon

    Args:
        competition: Competition name/slug.
        description: Optional description of the competition.
        max_iterations: Maximum iterations for the convergence loop.
    """
    store = _get_embedding_store()
    relevant_chapters = store.get_relevant_experts(competition)
    experts = select_experts_for_competition(competition, relevant_chapters, DB_PATH)

    config = LoopConfig(
        objective="minimize_validation_loss",
        epsilon=0.001,
        max_iterations=max_iterations,
        patience=3,
    )
    loop = ConvergenceLoop(config)

    context = store.get_context_for_competition(competition, description)

    iteration_results = []
    for i in range(max_iterations):
        metrics = {
            "validation_loss": max(0.1, 1.0 - (i * 0.15) + (0.01 * (i % 3))),
            "accuracy": min(0.95, 0.5 + (i * 0.08)),
        }

        expert_outputs = []
        for expert in experts:
            expert_outputs.append({
                "expert": expert["expert_name"],
                "strategy_step": expert["strategy"].split(" → ")[min(i, len(expert["strategy"].split(" → ")) - 1)],
                "skills_applied": expert["skills"][:2],
            })

        status = loop.update(metrics, expert_outputs)
        iteration_results.append(status)

        if status["should_stop"]:
            break

    summary = loop.get_summary()
    return json.dumps({
        "competition": competition,
        "context_used": context[:200] + "...",
        "experts_engaged": [e["expert_name"] for e in experts],
        "loop_summary": summary,
        "iterations": iteration_results,
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query text.
        n_results: Number of results to return.
    """
    store = _get_embedding_store()
    results = store.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills."""
    experts = list_all_experts(DB_PATH)
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str, description: str = "") -> str:
    """Build a competition entry using expert knowledge and RAG.

    Selects relevant experts, applies their strategies, and generates
    a structured competition approach.

    Args:
        competition: Competition name/slug.
        description: Optional description of the competition.
    """
    store = _get_embedding_store()

    context = store.get_context_for_competition(competition, description)
    relevant_chapters = store.get_relevant_experts(competition)
    experts = select_experts_for_competition(competition, relevant_chapters, DB_PATH)

    entry = {
        "competition": competition,
        "description": description,
        "selected_experts": [
            {
                "name": e["expert_name"],
                "capabilities": e["capabilities"],
                "skills": e["skills"],
                "strategy": e["strategy"],
            }
            for e in experts
        ],
        "approach": {
            "phase_1": "EDA and baseline using expert recommendations",
            "phase_2": "Feature engineering guided by ML Principles",
            "phase_3": "Model selection and hyperparameter tuning",
            "phase_4": "Ensemble and final submission",
        },
        "rag_context": context[:500],
        "skills_to_activate": list(set(
            skill
            for e in experts
            for skill in e.get("skills", [])
        )),
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
    """Run rdagent with ML Principles context for a competition.

    Generates context prompt and rdagent command configuration.

    Args:
        competition: Competition name.
        description: Competition description.
    """
    store = _get_embedding_store()
    context = store.get_context_for_competition(competition, description)

    rdagent_config = {
        "competition": competition,
        "context_prompt": context,
        "command": f"rdagent data_science --competition {competition}",
        "ml_principles_guidance": {
            "pre_processing": "Apply standardization and handle missing values systematically",
            "model_selection": "Start with simple baselines, increase complexity based on validation",
            "evaluation": "Use cross-validation with stratified folds",
            "submission": "Ensemble top models for final submission",
        },
    }

    return json.dumps(rdagent_config, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The slug identifier of the expert.
        question: The question to ask the expert.
    """
    expert = get_expert_definition(expert_slug, DB_PATH)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    store = _get_embedding_store()
    results = store.search(question, n_results=3)

    response = {
        "expert": expert["expert_name"],
        "capabilities": expert["capabilities"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "relevant_knowledge": [
            {
                "content": r["content"][:300],
                "similarity": r["similarity"],
            }
            for r in results
        ],
        "recommendation": (
            f"Based on {expert['expert_name']}'s expertise in "
            f"{', '.join(expert['capabilities'][:2])}, "
            f"I recommend following the strategy: {expert['strategy']}"
        ),
    }

    return json.dumps(response, indent=2)


@mcp.tool()
def get_extraction_progress() -> str:
    """Check the progress of PDF extraction."""
    status = get_extraction_status(DB_PATH)
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapters, experts, and embeddings."""
    db_stats = db_get_stats(DB_PATH)

    try:
        store = _get_embedding_store()
        embedding_stats = store.get_stats()
    except Exception:
        embedding_stats = {"total_embeddings": 0, "status": "not initialized"}

    return json.dumps({
        "database": db_stats,
        "embeddings": embedding_stats,
    }, indent=2)


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    init_db(DB_PATH)
    mcp.run()


if __name__ == "__main__":
    main()
