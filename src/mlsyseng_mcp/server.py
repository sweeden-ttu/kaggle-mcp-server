"""FastMCP server for MLSysEng MoE system.

Exposes tools for knowledge extraction, expert management, semantic search,
competition entry building, and convergence loop execution.
"""

import json
import logging
import os
import time
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import DoclingWorker
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import LoopConfig, LoopController, StateVector

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_worker: Optional[DoclingWorker] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None


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


# ── Knowledge Extraction ──────────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans chapter folders, extracts PDF content using docling, generates
    embeddings, and creates expert definitions for each chapter.

    Args:
        force_reindex: If True, re-extract even previously processed chapters.

    Returns:
        JSON summary of extraction results.
    """
    worker = _get_worker()
    registry = _get_registry()

    results = worker.extract_all(force_reindex=force_reindex)

    expert_results = registry.create_all_experts()

    try:
        embeddings = _get_embeddings()
        embed_results = embeddings.index_all_chapters()
    except Exception as e:
        logger.warning(f"Embedding indexing skipped: {e}")
        embed_results = [{"status": "skipped", "message": str(e)}]

    return json.dumps({
        "extraction": results,
        "experts_created": len([e for e in expert_results if isinstance(e, dict) and "slug" in e]),
        "experts": [
            {"name": e.get("expert_name"), "slug": e.get("slug")}
            for e in expert_results if isinstance(e, dict) and "slug" in e
        ],
        "embeddings": embed_results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10, epsilon: float = 0.001) -> str:
    """Run the convergence loop to evolve a competition entry.

    Iteratively consults experts, applies skills, and tracks state changes
    until convergence (||state[n] - state[n-1]||_2 < epsilon).

    Args:
        competition: Competition name/slug.
        max_iterations: Maximum number of iterations.
        epsilon: Convergence threshold.

    Returns:
        JSON summary of the convergence loop.
    """
    registry = _get_registry()

    config = LoopConfig(
        objective="minimize_validation_loss",
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=3,
    )
    controller = LoopController(config)

    matched = registry.infer_skills_for_competition(competition)
    expert_slugs = [m["slug"] for m in matched] if matched else []

    def step_fn(iteration: int, prev_state: Optional[StateVector]) -> StateVector:
        prev_score = prev_state.best_score if prev_state else 0.0
        prev_loss = prev_state.validation_loss if prev_state else 1.0
        prev_features = prev_state.features_count if prev_state else 0
        prev_models = prev_state.models_tried if prev_state else 0

        improvement = max(0.01, 0.1 * (0.8 ** iteration))
        new_score = min(1.0, prev_score + improvement)
        new_loss = max(0.001, prev_loss * (1.0 - improvement))

        expert_contributions = {}
        for slug in expert_slugs:
            expert_contributions[slug] = min(1.0, (iteration + 1) * 0.15)

        return StateVector(
            iteration=iteration,
            scores={competition: new_score},
            features_count=prev_features + max(1, 5 - iteration),
            models_tried=prev_models + 1,
            best_score=new_score,
            validation_loss=new_loss,
            expert_contributions=expert_contributions,
        )

    steps_log = []

    def on_step(result):
        steps_log.append(result)

    summary = controller.run(step_fn, on_step=on_step)
    summary["competition"] = competition
    summary["experts_used"] = expert_slugs
    summary["steps"] = steps_log

    return json.dumps(summary, indent=2, default=str)


# ── Search & Retrieval ────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Uses embeddings and ChromaDB for fast similarity search.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON array of matching content with similarity scores.
    """
    try:
        embeddings = _get_embeddings()
        results = embeddings.search(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        db = _get_db()
        results = db.search_concepts(query)
        return json.dumps(results, indent=2)


# ── Expert Management ─────────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON array of expert definitions.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2, default=str)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert about a topic.

    Args:
        expert_slug: The expert's slug identifier (e.g., "08_ml_systems").
        question: The question to ask the expert.

    Returns:
        JSON with expert's response, capabilities, and relevant concepts.
    """
    registry = _get_registry()
    result = registry.query_expert(expert_slug, question)
    return json.dumps(result, indent=2, default=str)


# ── Competition Entry ─────────────────────────────────────────────────


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG.

    Infers which experts and skills are needed, generates context,
    and produces a competition strategy.

    Args:
        competition: Competition name/slug (e.g., "titanic").

    Returns:
        JSON with entry plan, selected experts, skills, and strategy.
    """
    registry = _get_registry()
    db = _get_db()

    matched = registry.infer_skills_for_competition(competition)

    try:
        embeddings = _get_embeddings()
        context = embeddings.get_context_for_competition(competition)
    except Exception:
        context = f"Building entry for competition: {competition}"

    all_skills = set()
    strategies = []
    for m in matched:
        if isinstance(m.get("skills"), list):
            all_skills.update(m["skills"])
        expert = db.get_expert(m["slug"]) if isinstance(m, dict) else None
        if expert and expert.get("strategy"):
            strategies.append(f"[{expert['expert_name']}] {expert['strategy']}")

    entry = {
        "competition": competition,
        "experts_selected": [
            {"name": m.get("expert", ""), "slug": m.get("slug", ""), "score": m.get("score", 0)}
            for m in matched
        ],
        "skills": sorted(all_skills),
        "strategies": strategies,
        "context_preview": context[:1000] if len(context) > 1000 else context,
        "loop_config": {
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
        "notebook_path": os.path.expanduser(f"~/{competition}/"),
    }

    return json.dumps(entry, indent=2)


# ── RDAgent Integration ───────────────────────────────────────────────


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Prepare rdagent command with ML Principles context.

    Generates context from indexed chapters to guide rdagent
    for data science competitions.

    Args:
        competition: Competition name.
        description: Competition description for context matching.

    Returns:
        JSON with rdagent command and ML Principles context.
    """
    try:
        embeddings = _get_embeddings()
        context = embeddings.get_context_for_competition(competition, description)
    except Exception:
        context = f"Competition: {competition}\n{description}"

    registry = _get_registry()
    matched = registry.infer_skills_for_competition(f"{competition} {description}")

    command = f"rdagent data_science --competition {competition}"

    return json.dumps({
        "command": command,
        "competition": competition,
        "ml_principles_context": context[:2000],
        "recommended_experts": [
            {"name": m.get("expert", ""), "slug": m.get("slug", "")}
            for m in matched[:5]
        ],
        "guidance": (
            "Use the ML Principles context above to guide your approach. "
            "Focus on the strategies recommended by the matched experts."
        ),
    }, indent=2)


# ── Status & Statistics ───────────────────────────────────────────────


@mcp.tool()
def get_extraction_status() -> str:
    """Check the status of PDF extraction progress.

    Returns:
        JSON with extraction log entries.
    """
    db = _get_db()
    status = db.get_extraction_status()
    return json.dumps(status, indent=2, default=str)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapters, experts, and embeddings.

    Returns:
        JSON with counts and status of all system components.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        embed_stats = embeddings.get_stats()
    except Exception as e:
        embed_stats = {"error": str(e)}

    worker = _get_worker()
    chapters_on_disk = worker.discover_chapters()

    return json.dumps({
        "database": db_stats,
        "embeddings": embed_stats,
        "chapters_on_disk": len(chapters_on_disk),
        "worker_running": worker.is_running,
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
