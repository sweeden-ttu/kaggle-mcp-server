"""FastMCP server for the MLSysEng MoE system.

Exposes tools for knowledge extraction, expert management, semantic search,
competition entry building, and convergence loop execution.
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MoEDatabase
from .docling_worker import (
    discover_chapters,
    extract_all_chapters,
    get_extraction_status,
)
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import ConvergenceLoop

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-mcp")

_db: Optional[MoEDatabase] = None
_registry: Optional[ExpertRegistry] = None
_embeddings: Optional[EmbeddingEngine] = None
_loop: Optional[ConvergenceLoop] = None


def _get_db() -> MoEDatabase:
    global _db
    if _db is None:
        _db = MoEDatabase()
    return _db


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


def _get_embeddings() -> EmbeddingEngine:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingEngine(_get_db())
    return _embeddings


def _get_loop() -> ConvergenceLoop:
    global _loop
    if _loop is None:
        _loop = ConvergenceLoop(_get_db())
    return _loop


# ── Knowledge Extraction ──────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts, and generate embeddings.

    This is the main 'evolve' entry point. It:
    1. Scans all chapter folders in ML Principles
    2. Extracts PDF content using docling
    3. Generates embeddings
    4. Creates expert definitions

    Args:
        force_reindex: Re-extract and re-embed even if already done.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()
    registry = _get_registry()

    chapters = extract_all_chapters(db, force=force_reindex)

    experts_created = 0
    for ch in chapters:
        if ch.get("status") == "extracted":
            try:
                registry.register_expert_from_chapter(
                    chapter_id=ch["chapter_id"],
                    chapter_title=ch["title"],
                    concepts=ch.get("concepts", []),
                )
                experts_created += 1
            except Exception as exc:
                logger.error("Expert creation failed for %s: %s", ch["title"], exc)

    embedding_results = {}
    try:
        engine = _get_embeddings()
        embedding_results = engine.embed_all_chapters(force=force_reindex)
    except ImportError as exc:
        embedding_results = {"error": str(exc)}
    except Exception as exc:
        embedding_results = {"error": str(exc)}

    return json.dumps({
        "status": "complete",
        "chapters_processed": len(chapters),
        "chapters_extracted": sum(1 for c in chapters if c.get("status") == "extracted"),
        "experts_created": experts_created,
        "embedding_results": embedding_results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "", force_reindex: bool = False) -> str:
    """Run the full evolve pipeline: extract knowledge, then optionally
    build a competition entry with convergence loop.

    Args:
        competition: Optional competition name to build entry for.
        force_reindex: Re-extract even if already done.

    Returns:
        JSON summary including extraction and optional competition results.
    """
    extract_result = json.loads(extract_knowledge(force_reindex=force_reindex))

    if not competition:
        return json.dumps({
            "extraction": extract_result,
            "message": "Knowledge extracted. Provide a competition name to build an entry.",
        }, indent=2)

    entry_result = json.loads(build_entry(competition=competition))
    return json.dumps({
        "extraction": extract_result,
        "entry": entry_result,
    }, indent=2)


# ── Search & Query ────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles content.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON list of search hits with text, chapter, and similarity score.
    """
    try:
        engine = _get_embeddings()
        hits = engine.search(query, n_results=n_results)
        return json.dumps(hits, indent=2)
    except ImportError as exc:
        return json.dumps({
            "error": f"Search requires sentence-transformers and chromadb: {exc}"
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Expert Management ─────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills, and strategies.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The expert's slug (e.g., "08_ml_systems").
        question: The question to ask.

    Returns:
        JSON with expert info and relevant knowledge from their chapter.
    """
    registry = _get_registry()
    expert = registry.get_expert(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    context_hits = []
    try:
        engine = _get_embeddings()
        chapter_id = expert.get("chapter_id", "")
        if chapter_id:
            context_hits = engine.search(
                question, n_results=3, chapter_filter=chapter_id
            )
    except Exception:
        pass

    return json.dumps({
        "expert": expert,
        "question": question,
        "relevant_knowledge": context_hits,
        "recommendation": {
            "strategy": expert.get("strategy", ""),
            "skills": expert.get("skills", []),
            "capabilities": expert.get("capabilities", []),
        },
    }, indent=2)


# ── Competition Entry Building ────────────────────────────────────


@mcp.tool()
def build_entry(competition: str) -> str:
    """Build a competition entry using expert knowledge and RAG.

    Infers which experts and skills are needed, then runs the
    convergence loop to refine the entry.

    Args:
        competition: Competition name or description (e.g., "titanic").

    Returns:
        JSON with selected experts, skills, convergence results, and entry metadata.
    """
    db = _get_db()
    registry = _get_registry()

    inference = {}
    try:
        engine = _get_embeddings()
        inference = engine.infer_skills_for_competition(competition)
    except Exception as exc:
        logger.warning("RAG inference failed, using all experts: %s", exc)

    experts = [r["expert"] for r in inference.get("recommended_experts", [])]
    if not experts:
        experts = registry.list_experts()

    skills_used = inference.get("skills", [])
    if not skills_used:
        for exp in experts:
            skills_used.extend(exp.get("skills", []))
        skills_used = sorted(set(skills_used))

    loop = _get_loop()
    convergence_result = loop.run_competition_loop(
        competition=competition,
        experts=experts,
    )

    entry_id = f"entry_{competition}_{uuid.uuid4().hex[:6]}"
    entry = db.upsert_entry({
        "entry_id": entry_id,
        "competition": competition,
        "experts_used": [e.get("slug", "") for e in experts],
        "skills_used": skills_used,
        "notebook_path": f"~/{competition}/Expert_entry.ipynb",
        "status": "built",
    })

    return json.dumps({
        "entry": entry,
        "experts_selected": len(experts),
        "skills_selected": len(skills_used),
        "convergence": {
            "converged": convergence_result["converged"],
            "iterations": convergence_result["total_iterations"],
            "final_delta": convergence_result["final_delta"],
            "final_metrics": convergence_result["final_metrics"],
        },
    }, indent=2)


# ── Convergence Loop ──────────────────────────────────────────────


@mcp.tool()
def run_convergence_loop(
    competition: str = "default",
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> str:
    """Run the state convergence loop standalone.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon

    Args:
        competition: Competition identifier for the loop.
        epsilon: Convergence threshold.
        max_iterations: Maximum number of iterations.
        patience: Consecutive converging iterations before exit.

    Returns:
        JSON with full loop history and convergence status.
    """
    db = _get_db()
    loop = ConvergenceLoop(
        db=db,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )
    registry = _get_registry()
    experts = registry.list_experts() or []

    result = loop.run_competition_loop(
        competition=competition,
        experts=experts,
    )
    return json.dumps(result, indent=2)


# ── RDAgent Integration ───────────────────────────────────────────


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_context_results: int = 5,
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt from the knowledge base and returns
    the rdagent command with embedded context.

    Args:
        competition_name: Name of the Kaggle competition.
        description: Optional competition description.
        n_context_results: Number of context chunks to include.

    Returns:
        JSON with context prompt and suggested rdagent command.
    """
    context = ""
    try:
        engine = _get_embeddings()
        context = engine.get_rdagent_context(
            competition_name, description, n_results=n_context_results
        )
    except Exception as exc:
        context = f"[Context generation failed: {exc}]"

    command = (
        f"rdagent kaggle "
        f"--competition {competition_name} "
        f"--context-file /tmp/ml_principles_context.md"
    )

    return json.dumps({
        "context": context,
        "command": command,
        "instructions": (
            "1. Save the context to /tmp/ml_principles_context.md\n"
            "2. Run the rdagent command\n"
            "3. rdagent will use the ML Principles context to guide its approach"
        ),
    }, indent=2)


# ── Status & Stats ────────────────────────────────────────────────


@mcp.tool()
def get_extraction_status_tool() -> str:
    """Check extraction progress.

    Returns:
        JSON with chapter counts, status breakdown, and embedding stats.
    """
    db = _get_db()
    status = get_extraction_status(db)
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with counts of chapters, experts, embeddings, and entries.
    """
    db = _get_db()
    stats = db.get_stats()
    return json.dumps(stats, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
