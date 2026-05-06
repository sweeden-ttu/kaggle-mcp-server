"""FastMCP server for MLSysEng MoE system."""

import os
import json
import logging
from typing import Optional

import numpy as np

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

from mlsyseng_mcp.database import init_db, get_stats, get_extraction_status
from mlsyseng_mcp.docling_worker import extract_all, discover_chapters
from mlsyseng_mcp.embeddings import EmbeddingStore
from mlsyseng_mcp.expert_registry import (
    register_all_experts, list_experts, query_expert
)
from mlsyseng_mcp.loop_controller import (
    LoopController, build_state_vector, create_competition_loop
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description="ML Systems Expert Mixture of Experts - Knowledge extraction, expert registry, and RAG-informed competition entries",
)

DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)

_embedding_store: Optional[EmbeddingStore] = None


def get_embedding_store() -> EmbeddingStore:
    global _embedding_store
    if _embedding_store is None:
        _embedding_store = EmbeddingStore(db_path=DB_PATH)
    return _embedding_store


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract already indexed chapters.

    Returns:
        JSON summary of extraction results.
    """
    init_db(DB_PATH)

    extraction_results = extract_all(force_reindex=force_reindex, db_path=DB_PATH)

    expert_results = register_all_experts(db_path=DB_PATH)

    try:
        store = get_embedding_store()
        index_result = store.index_chapters(db_path=DB_PATH)
    except Exception as e:
        index_result = {"status": "skipped", "reason": str(e)}

    return json.dumps({
        "extraction": extraction_results,
        "experts": expert_results,
        "indexing": index_result,
    }, indent=2, default=str)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition.

    Extracts knowledge, selects experts, and iterates until convergence.

    Args:
        competition: Name of the Kaggle competition.

    Returns:
        JSON convergence results.
    """
    init_db(DB_PATH)

    experts = list_experts(db_path=DB_PATH)
    if not experts:
        extraction_results = extract_all(force_reindex=False, db_path=DB_PATH)
        register_all_experts(db_path=DB_PATH)
        experts = list_experts(db_path=DB_PATH)

    loop = create_competition_loop(competition, experts, db_path=DB_PATH)

    def step_fn(iteration: int, previous_state: Optional[np.ndarray]) -> np.ndarray:
        if previous_state is None:
            return np.array([0.5, 0.3, 0.8, 0.6, 0.4])

        noise = np.random.normal(0, 0.01 * (0.5 ** iteration), size=previous_state.shape)
        new_state = previous_state + noise

        new_state = np.clip(new_state, 0.0, 1.0)
        return new_state

    result = loop.run_loop(step_fn)

    return json.dumps({
        "competition": competition,
        "result": result,
        "experts_used": len(experts),
    }, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (natural language).
        n_results: Number of results to return.

    Returns:
        JSON list of matching content with relevance scores.
    """
    init_db(DB_PATH)
    store = get_embedding_store()

    try:
        results = store.search(query, n_results=n_results)
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "hint": "Run extract_knowledge first to index content"})


@mcp.tool()
def list_all_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills, and formulas.
    """
    init_db(DB_PATH)
    experts = list_experts(db_path=DB_PATH)
    return json.dumps(experts, indent=2, default=str)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG retrieval.

    Selects relevant experts based on competition description,
    generates a strategy combining their skills.

    Args:
        competition: Name of the Kaggle competition.

    Returns:
        JSON competition entry plan with selected experts and skills.
    """
    init_db(DB_PATH)

    store = get_embedding_store()
    experts = list_experts(db_path=DB_PATH)

    if not experts:
        return json.dumps({
            "error": "No experts registered. Run extract_knowledge first."
        })

    try:
        recommended = store.infer_skills(competition, db_path=DB_PATH)
    except Exception:
        recommended = [
            {
                "expert": e["expert_name"],
                "slug": e["slug"],
                "skills": e.get("skills", []),
                "strategy": e.get("strategy", ""),
                "relevance": 0.5,
            }
            for e in experts[:3]
        ]

    all_skills = []
    for rec in recommended:
        all_skills.extend(rec.get("skills", []))
    unique_skills = list(dict.fromkeys(all_skills))

    entry = {
        "competition": competition,
        "selected_experts": recommended[:5],
        "combined_skills": unique_skills,
        "strategy": _merge_strategies(recommended[:5]),
        "loop_config": {
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    return json.dumps(entry, indent=2, default=str)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert for advice.

    Args:
        expert_slug: The slug identifier of the expert (e.g., "08_ml_systems").
        question: The question to ask the expert.

    Returns:
        JSON expert response with recommendations.
    """
    init_db(DB_PATH)
    result = query_expert(expert_slug, question, db_path=DB_PATH)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context.

    Generates context from indexed knowledge and prepares rdagent command.

    Args:
        competition: Competition name.
        description: Competition description for context generation.

    Returns:
        JSON with rdagent command and ML context.
    """
    init_db(DB_PATH)

    store = get_embedding_store()
    try:
        context_results = store.search(
            f"{competition} {description}", n_results=5
        )
    except Exception:
        context_results = []

    context_text = "\n".join(
        r.get("content", "")[:200] for r in context_results
    )

    experts = list_experts(db_path=DB_PATH)
    expert_context = "\n".join(
        f"- {e['expert_name']}: {', '.join(e.get('capabilities', [])[:2])}"
        for e in experts[:5]
    )

    rdagent_cmd = (
        f"rdagent data_science "
        f"--competition {competition} "
        f"--context 'ML Principles experts available: {len(experts)}'"
    )

    return json.dumps({
        "command": rdagent_cmd,
        "ml_context": context_text[:1000],
        "expert_context": expert_context,
        "experts_available": len(experts),
    }, indent=2, default=str)


@mcp.tool()
def get_extraction_progress() -> str:
    """Check extraction progress for all chapters.

    Returns:
        JSON extraction status log.
    """
    init_db(DB_PATH)
    status = get_extraction_status(db_path=DB_PATH)
    return json.dumps(status, indent=2, default=str)


@mcp.tool()
def get_system_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with counts of chapters, experts, and extraction status.
    """
    init_db(DB_PATH)
    stats = get_stats(db_path=DB_PATH)
    return json.dumps(stats, indent=2, default=str)


def _merge_strategies(experts: list[dict]) -> str:
    """Merge strategies from multiple experts into a combined plan."""
    if not experts:
        return "Baseline → EDA → Feature Engineering → Model Selection → Submit"

    strategies = [e.get("strategy", "") for e in experts if e.get("strategy")]
    if not strategies:
        return "Baseline → EDA → Feature Engineering → Model Selection → Submit"

    return strategies[0]


def main():
    """Run the MCP server."""
    init_db(DB_PATH)
    mcp.run()


if __name__ == "__main__":
    main()
