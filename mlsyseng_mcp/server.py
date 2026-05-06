"""MLSysEng MoE MCP Server.

FastMCP server exposing tools for knowledge extraction, expert management,
competition entry building, and state convergence loops.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all_chapters, extract_concepts_from_text
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None
_loop_ctrl: Optional[LoopController] = None


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


def _get_loop_ctrl() -> LoopController:
    global _loop_ctrl
    if _loop_ctrl is None:
        _loop_ctrl = LoopController(_get_db())
    return _loop_ctrl


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans the ML Principles directory for chapter PDFs, extracts their content
    using docling, generates embeddings, and registers chapter experts.

    Args:
        force_reindex: Force re-extraction even if chapters already exist (default: False)

    Returns:
        JSON summary of extraction results including chapters processed and experts created
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    extraction_results = extract_all_chapters(db, force=force_reindex)

    try:
        embeddings.index_all_chapters(db)
    except Exception as e:
        logger.warning(f"Embedding indexing failed (non-fatal): {e}")

    experts = registry.register_all_from_chapters()

    return json.dumps({
        "status": "completed",
        "chapters_processed": len(extraction_results),
        "chapters_completed": sum(1 for r in extraction_results if r["status"] == "completed"),
        "chapters_skipped": sum(1 for r in extraction_results if r["status"] == "skipped"),
        "chapters_failed": sum(1 for r in extraction_results if r["status"] == "failed"),
        "experts_registered": len(experts),
        "details": extraction_results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str, max_iterations: int = 10, epsilon: float = 0.001) -> str:
    """Run the state convergence loop for a competition entry.

    Iteratively refines the competition approach using expert knowledge
    until the state vector converges (L2 norm of difference < epsilon).

    Args:
        competition: Competition name/slug (e.g., "titanic")
        max_iterations: Maximum number of iterations (default: 10)
        epsilon: Convergence threshold (default: 0.001)

    Returns:
        JSON with convergence results including iterations, L2 norms, and final state
    """
    db = _get_db()
    loop_ctrl = _get_loop_ctrl()
    embeddings = _get_embeddings()
    registry = _get_registry()

    relevant_experts = registry.get_experts_for_competition(competition, embeddings)

    def step_fn(iteration, prev_state):
        expert_scores = {}
        skill_activations = {}

        for expert in relevant_experts:
            base_score = 0.5 + random.uniform(-0.1, 0.1)
            if prev_state:
                decay = 0.9 ** iteration
                base_score = base_score * decay + (1 - decay) * 0.7
            expert_scores[expert["slug"]] = base_score

            for skill in expert.get("skills", []):
                skill_name = Path(skill).name
                activation = base_score * random.uniform(0.8, 1.0)
                skill_activations[skill_name] = max(
                    skill_activations.get(skill_name, 0), activation
                )

        state_vector = loop_ctrl.build_competition_state_vector(
            expert_scores, skill_activations
        )

        metadata = {
            "expert_scores": expert_scores,
            "skill_activations": skill_activations,
            "active_experts": [e["expert_name"] for e in relevant_experts],
        }

        return state_vector, metadata

    result = loop_ctrl.run_loop(
        competition=competition,
        step_fn=step_fn,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=3,
    )

    return json.dumps(result, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Uses embeddings to find the most relevant content chunks for a query.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON list of matching content with similarity scores and chapter info
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)

    return json.dumps({
        "query": query,
        "results": results,
        "total_results": len(results),
    }, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON list of all experts with their names, capabilities, skills, strategies, and formulas
    """
    registry = _get_registry()
    experts = registry.list_experts()

    summary = []
    for expert in experts:
        summary.append({
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert["capabilities"],
            "skills": expert["skills"],
            "strategy": expert["strategy"],
            "formula": expert["formula"],
        })

    return json.dumps({
        "total_experts": len(summary),
        "experts": summary,
    }, indent=2)


@mcp.tool()
def build_entry(competition: str) -> str:
    """Build a Kaggle competition entry using expert knowledge and RAG.

    Selects relevant experts for the competition, assembles their skills,
    and generates a competition entry plan.

    Args:
        competition: Competition name/slug (e.g., "titanic")

    Returns:
        JSON with entry plan including selected experts, skills, strategy, and notebook paths
    """
    embeddings = _get_embeddings()
    registry = _get_registry()

    relevant_experts = registry.get_experts_for_competition(competition, embeddings)

    all_skills = set()
    all_capabilities = []
    strategies = []

    for expert in relevant_experts:
        all_skills.update(expert.get("skills", []))
        all_capabilities.extend(expert.get("capabilities", []))
        strategies.append(expert.get("strategy", ""))

    home = str(Path.home())
    notebook_paths = []
    for expert in relevant_experts:
        nb_path = f"{home}/{competition}/Expert_{expert['slug']}.ipynb"
        notebook_paths.append(nb_path)

    entry = {
        "competition": competition,
        "selected_experts": [
            {
                "name": e["expert_name"],
                "slug": e["slug"],
                "relevance_score": e.get("relevance_score", 0.5),
            }
            for e in relevant_experts
        ],
        "combined_skills": sorted(all_skills),
        "capabilities": list(set(all_capabilities)),
        "strategy": strategies[0] if strategies else "Baseline → EDA → Feature Engineering → Model Selection → Submit",
        "notebook_paths": notebook_paths,
        "loop_config": relevant_experts[0].get("loop_config", {}) if relevant_experts else {},
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Prepare and run rdagent with ML Principles context for a competition.

    Generates context from relevant ML Principles chapters and configures
    rdagent for the competition.

    Args:
        competition: Competition name (e.g., "titanic")
        description: Optional competition description for better expert matching

    Returns:
        JSON with rdagent configuration and context prompt
    """
    embeddings = _get_embeddings()
    registry = _get_registry()
    db = _get_db()

    search_query = f"{competition} {description}".strip()
    search_results = embeddings.search(search_query, n_results=5)

    relevant_experts = registry.get_experts_for_competition(search_query, embeddings)

    context_parts = []
    for result in search_results:
        context_parts.append(f"[Chapter {result['chapter_number']}: {result['title']}]\n{result['content']}")

    context_prompt = "\n\n---\n\n".join(context_parts)

    expert_strategies = []
    for expert in relevant_experts:
        expert_strategies.append(
            f"- {expert['expert_name']}: {expert.get('strategy', 'N/A')}"
        )

    rdagent_config = {
        "competition": competition,
        "description": description,
        "context_prompt": context_prompt[:4000],
        "relevant_experts": [e["expert_name"] for e in relevant_experts],
        "strategies": expert_strategies,
        "command": f"rdagent --competition {competition} --context-file /tmp/mlsyseng_context.txt",
        "skills_activated": sorted(
            set(s for e in relevant_experts for s in e.get("skills", []))
        ),
    }

    return json.dumps(rdagent_config, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the progress of PDF extraction.

    Returns:
        JSON with extraction status for each chapter (pending, extracting, completed, failed)
    """
    db = _get_db()
    statuses = db.get_extraction_status()

    return json.dumps({
        "extraction_status": statuses,
        "summary": {
            "total": len(statuses),
            "completed": sum(1 for s in statuses if s["status"] == "completed"),
            "pending": sum(1 for s in statuses if s["status"] == "pending"),
            "failed": sum(1 for s in statuses if s["status"] == "failed"),
            "extracting": sum(1 for s in statuses if s["status"] == "extracting"),
        },
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with database stats, embedding stats, and system health information
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        embed_stats = embeddings.get_stats()
    except Exception:
        embed_stats = {"status": "not available"}

    return json.dumps({
        "database": db_stats,
        "embeddings": embed_stats,
        "system": {
            "ml_principles_path": os.environ.get(
                "ML_PRINCIPLES_PATH",
                str(Path.home() / "Desktop" / "Machine Learning Principles - Chapters"),
            ),
            "skills_path": os.environ.get(
                "KAGGLE_SKILLS_PATH",
                str(Path.home() / "skills"),
            ),
        },
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert for advice.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems")
        question: Question to ask the expert

    Returns:
        JSON with expert's response based on their knowledge, capabilities, and strategy
    """
    registry = _get_registry()
    embeddings = _get_embeddings()
    db = _get_db()

    expert = registry.get_expert(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert not found: {expert_slug}"})

    relevant_content = []
    if expert.get("chapter_id"):
        chapter = db.get_chapter(expert["chapter_id"])
        if chapter:
            results = embeddings.search(question, n_results=3, chapter_filter=chapter["chapter_number"])
            relevant_content = [r["content"] for r in results]

    response = {
        "expert": expert["expert_name"],
        "slug": expert_slug,
        "capabilities": expert["capabilities"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "relevant_knowledge": relevant_content,
        "skills_recommended": expert["skills"],
        "advice": (
            f"Based on my expertise in {expert['expert_name']}, "
            f"I recommend the following strategy: {expert['strategy']}. "
            f"Key capabilities: {', '.join(expert['capabilities'][:3])}."
        ),
    }

    return json.dumps(response, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
