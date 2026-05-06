"""FastMCP server for MLSysEng MoE - exposes tools for knowledge extraction, expert management, and competition building."""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_moe.mlsyseng_mcp.database import Database
from mlsyseng_moe.mlsyseng_mcp.docling_worker import DoclingWorker
from mlsyseng_moe.mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_moe.mlsyseng_mcp.expert_registry import ExpertRegistry
from mlsyseng_moe.mlsyseng_mcp.loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

db = Database()
docling_worker = DoclingWorker()
embedding_engine = EmbeddingEngine()
expert_registry = ExpertRegistry(db)
loop_controller = LoopController(db)


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extracts already completed chapters.

    Returns:
        JSON summary of extraction results.
    """
    results = docling_worker.extract_all(force_reindex=force_reindex, db=db)

    for result in results:
        embedding_engine.index_chunks(result["chunks"], result["slug"])

    experts = expert_registry.register_experts_from_extraction(results)

    return json.dumps(
        {
            "chapters_extracted": len(results),
            "experts_registered": len(experts),
            "chapters": [r["name"] for r in results],
        },
        indent=2,
    )


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the state convergence loop for a competition.

    Selects relevant experts using RAG, then iterates until state converges
    or max iterations reached.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon

    Args:
        competition: Competition slug/name.

    Returns:
        JSON summary of convergence loop results.
    """
    experts = expert_registry.select_experts_for_competition(
        competition, embedding_engine
    )

    if not experts:
        all_experts_raw = expert_registry.list_experts()
        experts = all_experts_raw[:3] if all_experts_raw else []

    result = loop_controller.run_loop(
        competition_slug=competition,
        experts=experts,
    )

    summary = {
        "competition": competition,
        "status": result["status"],
        "iterations": result["iterations"],
        "experts_used": result["experts_used"],
        "final_metrics": result["final_metrics"],
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language search query.
        n_results: Number of results to return.

    Returns:
        JSON array of matching knowledge chunks with similarity scores.
    """
    results = embedding_engine.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON array of expert summaries.
    """
    experts = expert_registry.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Uses RAG to find relevant experts, generates context from ML Principles,
    and outlines a competition strategy.

    Args:
        competition: Competition name/slug.

    Returns:
        JSON competition entry plan.
    """
    experts = expert_registry.select_experts_for_competition(
        competition, embedding_engine
    )

    context = embedding_engine.get_context_for_competition(competition)

    if not experts:
        all_experts_raw = expert_registry.list_experts()
        experts = all_experts_raw[:3] if all_experts_raw else []

    entry = {
        "competition": competition,
        "selected_experts": [
            {
                "name": e.get("expert_name", e.get("slug", "unknown")),
                "relevance": e.get("relevance_score", 0),
            }
            for e in experts
        ],
        "strategy": "Baseline → EDA → Feature Engineering → Model Selection → Submit",
        "skills_to_use": [],
        "context_summary": context[:2000],
    }

    all_skills = set()
    for e in experts:
        if isinstance(e.get("skills"), list):
            all_skills.update(e["skills"])
    entry["skills_to_use"] = sorted(all_skills)

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context.

    Generates a context-enriched prompt for rdagent data_science mode.

    Args:
        competition: Competition name.
        description: Competition description.

    Returns:
        JSON with rdagent command and context.
    """
    context = embedding_engine.get_context_for_competition(competition, description)

    command = f"rdagent data_science --competition {competition}"

    return json.dumps(
        {
            "command": command,
            "context": context[:3000],
            "experts_consulted": [
                e.get("slug", "") for e in
                expert_registry.select_experts_for_competition(
                    f"{competition} {description}", embedding_engine
                )
            ],
        },
        indent=2,
    )


@mcp.tool()
def get_extraction_status() -> str:
    """Check the current extraction progress.

    Returns:
        JSON with extraction status details.
    """
    status = docling_worker.get_extraction_status(db)
    return json.dumps(status, indent=2, default=str)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with database and embedding stats.
    """
    db_stats = db.get_stats()
    embedding_stats = embedding_engine.get_stats()

    return json.dumps(
        {
            "database": db_stats,
            "embeddings": embedding_stats,
        },
        indent=2,
    )


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The expert's slug identifier.
        question: The question to ask.

    Returns:
        JSON with expert's response context.
    """
    expert = expert_registry.get_expert_full(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    relevant_chunks = embedding_engine.search(
        question, n_results=5, chapter_filter=expert_slug
    )

    return json.dumps(
        {
            "expert": expert.get("expert_name"),
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "relevant_knowledge": [
                {"content": c["content"][:300], "similarity": c["similarity"]}
                for c in relevant_chunks
            ],
        },
        indent=2,
    )


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
