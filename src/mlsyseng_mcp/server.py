"""
MLSysEng MoE FastMCP Server.

Provides MCP tools for:
- extract-knowledge / evolve: Extract PDFs, index chapters, create experts
- search-concepts: Semantic search over ML Principles
- list-experts: List all chapter experts
- build-entry: Build competition entry using expert knowledge
- run-rdagent: Run rdagent with ML Principles context
- ask_<expert>: Query specific chapter expert
- get-extraction-status: Check extraction progress
- get-stats: System statistics
"""

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MLSysEngDatabase
from .docling_worker import DoclingWorker
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MLSysEngDatabase] = None
_worker: Optional[DoclingWorker] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None


def _get_db() -> MLSysEngDatabase:
    global _db
    if _db is None:
        _db = MLSysEngDatabase()
    return _db


def _get_worker() -> DoclingWorker:
    global _worker
    if _worker is None:
        _worker = DoclingWorker(_get_db())
    return _worker


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


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract knowledge from ML Principles PDFs.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed (default: False)

    Returns:
        JSON string with extraction results including chapters found/processed and any errors
    """
    db = _get_db()
    worker = _get_worker()

    result = worker.run(force_reindex=force_reindex)

    if result["chapters_processed"] > 0:
        registry = _get_registry()
        experts = registry.register_all_chapters()
        result["experts_registered"] = len(experts)

        try:
            embeddings = _get_embeddings()
            chapters = db.list_chapters()
            total_chunks = 0
            for ch in chapters:
                chunks = embeddings.index_chapter(
                    ch.chapter_id,
                    ch.chapter_name,
                    ch.content_md,
                    ch.concepts,
                )
                total_chunks += chunks
            result["chunks_indexed"] = total_chunks
        except Exception as e:
            result["embedding_warning"] = str(e)

    return json.dumps(result, indent=2)


@mcp.tool()
def evolve(
    competition: str = "titanic",
    max_iterations: int = 10,
    epsilon: float = 0.001,
    patience: int = 3,
) -> str:
    """
    Run the convergence loop for a competition.

    Iteratively selects experts, applies strategies, and evaluates until
    the state vector converges (||state[n] - state[n-1]||_2 < epsilon).

    Args:
        competition: Competition name/slug (default: "titanic")
        max_iterations: Maximum number of improvement iterations (default: 10)
        epsilon: Convergence threshold for L2 norm (default: 0.001)
        patience: Number of consecutive converging iterations before stopping (default: 3)

    Returns:
        JSON string with convergence results, iteration history, and final state
    """
    db = _get_db()
    registry = _get_registry()

    experts_data = registry.export_all_experts_json()

    try:
        embeddings = _get_embeddings()
        matched = registry.get_experts_for_competition(competition, embeddings)
        if matched:
            experts_data = [m["expert"] for m in matched]
    except Exception:
        pass

    loop = LoopController(db, epsilon=epsilon, max_iterations=max_iterations, patience=patience)
    step_fn = LoopController.default_step_fn(experts_data)
    result = loop.run(competition, step_fn)

    result["experts_used"] = [
        {"name": e.get("expert_name", e.get("expert_id", "unknown")), "slug": e.get("slug", "")}
        for e in experts_data
    ]

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Uses sentence-transformer embeddings and ChromaDB for fast similarity search.

    Args:
        query: Natural language search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON string with search results including content snippets, similarity scores, and metadata
    """
    try:
        embeddings = _get_embeddings()
        results = embeddings.search(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "message": "Ensure chapters are indexed first"})


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts.

    Each expert has capabilities, skills, strategy, formula, and loop configuration.

    Returns:
        JSON string with list of all experts and their full definitions
    """
    registry = _get_registry()
    experts = registry.export_all_experts_json()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(
    competition: str = "titanic",
    description: str = "",
) -> str:
    """
    Build a competition entry using expert knowledge.

    Selects relevant experts via RAG, combines their strategies and skills,
    and produces a structured competition plan.

    Args:
        competition: Competition name/slug (default: "titanic")
        description: Optional description of the competition task

    Returns:
        JSON string with selected experts, combined strategy, skills to apply, and notebook suggestions
    """
    registry = _get_registry()

    search_text = description or competition
    try:
        embeddings = _get_embeddings()
        matched = registry.get_experts_for_competition(search_text, embeddings)
    except Exception:
        matched = registry.get_experts_for_competition(search_text)

    combined_skills = set()
    combined_capabilities = []
    strategies = []

    for item in matched:
        expert = item["expert"]
        for s in expert.get("skills", []):
            combined_skills.add(s)
        combined_capabilities.extend(expert.get("capabilities", []))
        strategies.append(
            {
                "expert": expert["expert_name"],
                "strategy": expert.get("strategy", ""),
                "formula": expert.get("formula", {}),
            }
        )

    notebooks = [
        {
            "path": f"~/{competition}/Expert_{item['expert']['slug']}.ipynb",
            "expert": item["expert"]["expert_name"],
            "focus": ", ".join(item["expert"].get("capabilities", [])[:3]),
        }
        for item in matched
    ]

    entry = {
        "competition": competition,
        "experts_selected": len(matched),
        "experts": [
            {
                "name": item["expert"]["expert_name"],
                "slug": item["expert"]["slug"],
                "relevance": item.get("relevance_score", 1.0),
                "capabilities": item["expert"].get("capabilities", []),
            }
            for item in matched
        ],
        "combined_skills": sorted(combined_skills),
        "combined_capabilities": list(dict.fromkeys(combined_capabilities)),
        "strategies": strategies,
        "suggested_notebooks": notebooks,
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
) -> str:
    """
    Prepare rdagent context with ML Principles knowledge.

    Generates a context prompt enriched with relevant chapter knowledge
    for guiding rdagent data_science competitions.

    Args:
        competition_name: Name of the competition
        description: Description of the competition task

    Returns:
        JSON string with rdagent context prompt, relevant chapters, and command suggestion
    """
    search_text = description or competition_name

    context_chunks = []
    try:
        embeddings = _get_embeddings()
        results = embeddings.search(search_text, n_results=8)
        for r in results:
            context_chunks.append(
                {
                    "chapter": r["metadata"]["chapter_name"],
                    "content": r["content"][:500],
                    "relevance": r["similarity"],
                }
            )
    except Exception:
        pass

    context_prompt = (
        f"Competition: {competition_name}\n"
        f"Description: {description}\n\n"
        "Relevant ML Principles:\n"
    )
    for chunk in context_chunks:
        context_prompt += (
            f"\n--- {chunk['chapter']} (relevance: {chunk['relevance']:.2f}) ---\n"
            f"{chunk['content']}\n"
        )

    return json.dumps(
        {
            "competition": competition_name,
            "context_prompt": context_prompt,
            "relevant_chapters": context_chunks,
            "rdagent_command": (
                f"rdagent --competition {competition_name} "
                f"--context-file mlsyseng_context.txt"
            ),
        },
        indent=2,
    )


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """
    Query a specific chapter expert.

    Retrieves expert knowledge relevant to the question using semantic search
    scoped to that expert's chapter.

    Args:
        expert_slug: The expert's slug identifier (e.g., "08_ml_systems")
        question: The question to ask the expert

    Returns:
        JSON string with expert info and relevant knowledge snippets
    """
    registry = _get_registry()
    expert = registry.get_expert(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    relevant_content = []
    try:
        embeddings = _get_embeddings()
        results = embeddings.search(
            question,
            n_results=5,
            chapter_filter=expert.chapter_id,
        )
        relevant_content = [
            {"content": r["content"][:500], "similarity": r["similarity"]}
            for r in results
        ]
    except Exception:
        pass

    return json.dumps(
        {
            "expert": {
                "name": expert.expert_name,
                "slug": expert.slug,
                "capabilities": expert.capabilities,
                "strategy": expert.strategy,
                "formula": expert.formula,
            },
            "question": question,
            "relevant_knowledge": relevant_content,
        },
        indent=2,
    )


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check the current extraction progress.

    Returns:
        JSON string with extraction status including total chapters, processed count,
        current status, and per-chapter progress
    """
    db = _get_db()
    worker = _get_worker()

    status = db.get_status("extraction_status") or "not_started"
    total = db.get_status("extraction_total") or "0"
    processed = db.get_status("extraction_processed") or "0"

    return json.dumps(
        {
            "status": status,
            "total_chapters": int(total),
            "chapters_processed": int(processed),
            "per_chapter_progress": worker.progress,
        },
        indent=2,
    )


@mcp.tool()
def get_stats() -> str:
    """
    Get system statistics.

    Returns:
        JSON string with database stats (chapters, experts, words) and
        embedding collection stats (chunks, model info)
    """
    db = _get_db()
    stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        stats["embeddings"] = embeddings.get_collection_stats()
    except Exception as e:
        stats["embeddings"] = {"status": "unavailable", "reason": str(e)}

    return json.dumps(stats, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
