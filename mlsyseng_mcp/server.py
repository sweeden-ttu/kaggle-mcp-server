"""FastMCP server for MLSysEng MoE system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import MoEDatabase
from mlsyseng_mcp.docling_worker import run_extraction
from mlsyseng_mcp.embeddings import EmbeddingManager
from mlsyseng_mcp.expert_registry import (
    register_experts_from_chapters,
    get_expert_for_query,
)
from mlsyseng_mcp.loop_controller import LoopController, default_step_fn

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MoEDatabase] = None
_emb: Optional[EmbeddingManager] = None


def _get_db() -> MoEDatabase:
    global _db
    if _db is None:
        _db = MoEDatabase()
    return _db


def _get_emb() -> EmbeddingManager:
    global _emb
    if _emb is None:
        _emb = EmbeddingManager()
    return _emb


# ── Knowledge extraction ──────────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    This will:
    1. Scan chapter folders for PDFs
    2. Extract text content using docling
    3. Generate embeddings for semantic search
    4. Create expert definitions from chapters

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON with extraction results including chapters processed and experts created.
    """
    db = _get_db()
    emb = _get_emb()

    extraction_result = run_extraction(db, force_reindex=force_reindex)

    if extraction_result.get("status") in ("no_chapters_found", "already_indexed"):
        if extraction_result.get("status") == "already_indexed":
            experts = register_experts_from_chapters(db)
            try:
                emb.index_all_chapters(db)
            except Exception as e:
                logger.warning("Embedding indexing skipped: %s", e)
            return json.dumps(
                {
                    **extraction_result,
                    "experts_registered": len(experts),
                },
                indent=2,
            )
        return json.dumps(extraction_result, indent=2)

    experts = register_experts_from_chapters(db)
    try:
        embedding_stats = emb.index_all_chapters(db)
    except Exception as e:
        logger.warning("Embedding indexing failed: %s", e)
        embedding_stats = {"error": str(e)}

    return json.dumps(
        {
            **extraction_result,
            "experts_registered": len(experts),
            "embedding_stats": embedding_stats,
        },
        indent=2,
    )


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """Run the state convergence loop for a competition.

    Executes iterative optimization until state converges
    (||state[n] - state[n-1]||_2 < epsilon) or max iterations reached.

    Args:
        competition: Competition identifier (default: "titanic").
        max_iterations: Maximum optimization iterations (default: 10).

    Returns:
        JSON with convergence results including iterations, deltas, and final metrics.
    """
    db = _get_db()
    emb = _get_emb()

    relevant_experts = []
    try:
        relevant_experts = emb.infer_experts_for_competition(competition, db)
    except Exception:
        pass

    initial_state = [1.0] * max(1, len(relevant_experts))
    initial_metrics = {
        "competition": competition,
        "expert_count": len(relevant_experts),
        "experts": [e["expert_name"] for e in relevant_experts[:5]],
    }

    controller = LoopController(db, max_iterations=max_iterations)
    result = controller.run_loop(
        competition=competition,
        step_fn=default_step_fn,
        initial_state=initial_state,
        initial_metrics=initial_metrics,
    )

    return json.dumps(result, indent=2)


# ── Search ────────────────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language query (e.g., "neural network optimization").
        n_results: Number of results to return (default: 5).

    Returns:
        JSON list of relevant text passages with metadata and relevance scores.
    """
    emb = _get_emb()
    try:
        results = emb.search(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Expert management ─────────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills, strategy, and formulas.
    """
    db = _get_db()
    experts = db.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The expert's slug identifier (e.g., "08_ml_systems").
        question: The question to ask the expert.

    Returns:
        JSON with expert's response including relevant concepts and recommendations.
    """
    db = _get_db()
    emb = _get_emb()

    expert = db.get_expert_by_slug(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    chapter = db.get_chapter(expert.get("chapter_id", ""))

    context_hits = []
    try:
        context_hits = emb.search(
            question,
            n_results=3,
            chapter_filter=expert.get("chapter_id"),
        )
    except Exception:
        pass

    response = {
        "expert": expert["expert_name"],
        "slug": expert["slug"],
        "capabilities": expert.get("capabilities", []),
        "strategy": expert.get("strategy", ""),
        "relevant_context": [
            {
                "text": hit["document"][:300],
                "relevance": round(hit["relevance"], 4),
            }
            for hit in context_hits
        ],
        "recommended_skills": expert.get("skills", []),
        "formula": expert.get("formula", {}),
    }
    return json.dumps(response, indent=2)


# ── Competition entry ─────────────────────────────────────────────────


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Uses RAG to select the most relevant experts and their skills,
    then produces a structured plan for the competition.

    Args:
        competition: Competition name/slug (default: "titanic").

    Returns:
        JSON with entry plan including selected experts, skills, and strategy.
    """
    db = _get_db()
    emb = _get_emb()

    relevant_experts = []
    try:
        relevant_experts = emb.infer_experts_for_competition(competition, db)
    except Exception:
        experts = db.list_experts()
        relevant_experts = [
            {
                "expert_name": e["expert_name"],
                "slug": e["slug"],
                "relevance_score": 0.5,
                "skills": e.get("skills", []),
                "strategy": e.get("strategy", ""),
            }
            for e in experts[:3]
        ]

    all_skills: set = set()
    strategies = []
    for expert in relevant_experts:
        all_skills.update(expert.get("skills", []))
        if expert.get("strategy"):
            strategies.append(
                f"{expert['expert_name']}: {expert['strategy']}"
            )

    entry = {
        "competition": competition,
        "selected_experts": relevant_experts,
        "combined_skills": sorted(all_skills),
        "strategies": strategies,
        "execution_plan": [
            "1. Download competition data",
            "2. EDA with expert-guided feature analysis",
            "3. Feature engineering using recommended skills",
            "4. Model training with expert strategies",
            "5. Cross-validation and hyperparameter tuning",
            "6. Ensemble top models",
            "7. Generate submission",
        ],
        "convergence_config": {
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    return json.dumps(entry, indent=2)


# ── RDAgent integration ──────────────────────────────────────────────


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context for a competition.

    Generates context from indexed ML Principles knowledge and prepares
    rdagent command with expert guidance.

    Args:
        competition: Competition name (default: "titanic").
        description: Competition description for better context matching.

    Returns:
        JSON with rdagent context and command configuration.
    """
    db = _get_db()
    emb = _get_emb()

    query = f"{competition} {description}" if description else competition

    context_passages = []
    try:
        hits = emb.search(query, n_results=5)
        context_passages = [
            {
                "text": hit["document"][:500],
                "chapter": hit["metadata"].get("title", ""),
                "relevance": round(hit["relevance"], 4),
            }
            for hit in hits
        ]
    except Exception:
        pass

    relevant_experts = []
    try:
        relevant_experts = emb.infer_experts_for_competition(query, db)
    except Exception:
        pass

    context_prompt = f"Competition: {competition}\n"
    if description:
        context_prompt += f"Description: {description}\n"
    context_prompt += "\nRelevant ML Principles:\n"
    for passage in context_passages:
        context_prompt += f"- [{passage['chapter']}] {passage['text'][:200]}\n"

    return json.dumps(
        {
            "competition": competition,
            "context_prompt": context_prompt,
            "relevant_passages": context_passages,
            "recommended_experts": relevant_experts[:3],
            "rdagent_command": f"rdagent --competition {competition}",
        },
        indent=2,
    )


# ── Status & stats ───────────────────────────────────────────────────


@mcp.tool()
def get_extraction_status(job_id: str = "") -> str:
    """Check extraction job progress.

    Args:
        job_id: Job ID to check. If empty, returns latest job status.

    Returns:
        JSON with extraction job status and progress.
    """
    db = _get_db()
    if job_id:
        job = db.get_extraction_job(job_id)
        if job:
            return json.dumps(job, indent=2)
        return json.dumps({"error": f"Job {job_id} not found"})

    chapters = db.list_chapters()
    return json.dumps(
        {
            "total_chapters_indexed": len(chapters),
            "chapters": [
                {"id": c["chapter_id"], "title": c["title"]}
                for c in chapters
            ],
        },
        indent=2,
    )


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with database stats, embedding stats, and expert counts.
    """
    db = _get_db()
    emb = _get_emb()

    db_stats = db.get_stats()

    try:
        emb_stats = emb.get_stats()
    except Exception as e:
        emb_stats = {"error": str(e)}

    return json.dumps(
        {
            "database": db_stats,
            "embeddings": emb_stats,
        },
        indent=2,
    )


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
