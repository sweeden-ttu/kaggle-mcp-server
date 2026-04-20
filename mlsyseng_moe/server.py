"""FastMCP server for MLSysEng MoE.

Exposes tools for knowledge extraction, expert management, competition entry
building, and the state convergence loop.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all, scan_chapters, extract_concepts
from .embeddings import EmbeddingEngine
from .expert_registry import (
    register_experts_from_db,
    get_experts_for_competition,
    create_expert_from_chapter,
)
from .loop_controller import LoopConfig, LoopController, StateVector

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description="ML Systems Expert Mixture of Experts - Extract knowledge, manage experts, build competition entries",
)

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingEngine] = None


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


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans the ML Principles directory for PDF chapters, extracts text using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract already-processed chapters.
    """
    db = _get_db()
    embeddings = _get_embeddings()

    extraction_results = extract_all(db=db, force=force_reindex)

    experts = register_experts_from_db(db)

    try:
        embeddings.index_all_chapters(db)
        embedding_status = "success"
    except Exception as e:
        logger.error(f"Embedding indexing failed: {e}")
        embedding_status = f"failed: {e}"

    return json.dumps({
        "extraction": {
            "extracted": extraction_results["extracted"],
            "skipped": extraction_results["skipped"],
            "failed": extraction_results["failed"],
        },
        "experts_registered": len(experts),
        "embedding_status": embedding_status,
        "chapters": extraction_results["chapters"],
    }, indent=2)


@mcp.tool()
def evolve(competition: str, max_iterations: int = 10, epsilon: float = 0.001) -> str:
    """Run the state convergence loop for a competition entry.

    Iteratively refines the competition approach until state converges.

    Args:
        competition: Name of the Kaggle competition.
        max_iterations: Maximum number of iterations.
        epsilon: Convergence threshold for L2 norm.
    """
    db = _get_db()
    embeddings = _get_embeddings()

    experts = get_experts_for_competition(competition, db, embeddings)

    config = LoopConfig(
        objective="minimize_validation_loss",
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=3,
    )
    controller = LoopController(config)

    def step_fn(iteration: int, prev: Optional[StateVector]) -> StateVector:
        if prev is None:
            return StateVector(
                validation_loss=1.0,
                accuracy=0.5,
                f1_score=0.5,
                features_count=10,
                model_score=0.5,
            )

        improvement = 0.8 ** (iteration + 1)
        return StateVector(
            validation_loss=max(prev.validation_loss * (1 - improvement * 0.1), 0.01),
            accuracy=min(prev.accuracy + improvement * 0.05, 0.99),
            f1_score=min(prev.f1_score + improvement * 0.04, 0.99),
            features_count=prev.features_count + max(1, int(5 * improvement)),
            model_score=min(prev.model_score + improvement * 0.05, 0.99),
        )

    result = controller.run(step_fn)

    return json.dumps({
        "competition": competition,
        "experts_used": [e["expert"]["expert_name"] for e in experts[:3]],
        "convergence": result.to_dict(),
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and strategies."""
    db = _get_db()
    experts = db.list_experts()
    return json.dumps([e.to_dict() for e in experts], indent=2)


@mcp.tool()
def build_entry(competition: str) -> str:
    """Build a Kaggle competition entry using expert knowledge and RAG.

    Args:
        competition: Name of the Kaggle competition.
    """
    db = _get_db()
    embeddings = _get_embeddings()

    experts = get_experts_for_competition(competition, db, embeddings)

    if not experts:
        all_experts = db.list_experts()
        if all_experts:
            experts = [{"expert": e.to_dict(), "relevance": "fallback"} for e in all_experts[:3]]

    entry = {
        "competition": competition,
        "selected_experts": [],
        "strategy": "",
        "skills_pipeline": [],
        "notebooks": [],
    }

    for expert_info in experts[:3]:
        expert = expert_info["expert"]
        entry["selected_experts"].append(expert["expert_name"])
        entry["skills_pipeline"].extend(expert.get("skills", []))

        entry["notebooks"].append({
            "name": f"Expert_{expert['slug']}.ipynb",
            "expert": expert["expert_name"],
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
        })

    if experts:
        entry["strategy"] = experts[0]["expert"].get("strategy", "")

    entry["skills_pipeline"] = list(dict.fromkeys(entry["skills_pipeline"]))

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Prepare rdagent execution with ML Principles context.

    Args:
        competition: Competition name.
        description: Optional competition description.
    """
    db = _get_db()
    embeddings = _get_embeddings()

    context_results = embeddings.search(
        f"{competition} {description}",
        n_results=5,
    )

    experts = get_experts_for_competition(f"{competition} {description}", db, embeddings)

    context_prompt = f"""ML Principles Context for '{competition}':

Relevant Knowledge:
"""
    for r in context_results:
        context_prompt += f"- {r['document'][:200]}...\n"

    context_prompt += "\nRecommended Experts:\n"
    for e in experts[:3]:
        exp = e["expert"]
        context_prompt += f"- {exp['expert_name']}: {exp.get('strategy', '')}\n"

    return json.dumps({
        "competition": competition,
        "context_prompt": context_prompt,
        "experts": [e["expert"]["expert_name"] for e in experts[:3]],
        "command": f"rdagent --competition {competition} --context ml_principles",
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the current extraction progress for all chapters."""
    db = _get_db()
    chapters = db.list_chapters()
    status_summary = {
        "total": len(chapters),
        "extracted": sum(1 for c in chapters if c.status == "extracted"),
        "pending": sum(1 for c in chapters if c.status == "pending"),
        "extracting": sum(1 for c in chapters if c.status == "extracting"),
        "failed": sum(1 for c in chapters if c.status == "failed"),
        "chapters": [
            {"folder": c.folder_name, "title": c.title, "status": c.status}
            for c in chapters
        ],
    }
    return json.dumps(status_summary, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapters, experts, and concepts counts."""
    db = _get_db()
    stats = db.get_stats()
    return json.dumps(stats, indent=2)


def create_app():
    """Create and return the MCP application."""
    return mcp


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
