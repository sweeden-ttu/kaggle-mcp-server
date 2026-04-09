"""FastMCP server for MLSysEng MoE system."""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all_chapters, discover_chapter_folders
from .expert_registry import (
    build_expert_from_chapter,
    save_expert_json,
    load_all_experts_from_disk,
)
from .loop_controller import LoopController, initial_state_vector

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")
_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON summary of extraction results.
    """
    db = get_db()

    def on_progress(folder, status, message):
        db.log_extraction(folder, status, message)

    chapters = extract_all_chapters(force_reindex=force_reindex, on_progress=on_progress)

    results = []
    for chapter in chapters:
        chapter_id = db.upsert_chapter(
            folder_name=chapter["folder_name"],
            title=chapter["title"],
            markdown=chapter["markdown"],
            concepts=chapter["concepts"],
        )

        expert_def = build_expert_from_chapter(chapter, chapter_id)
        db.upsert_expert(expert_def)
        save_expert_json(expert_def)

        try:
            from .embeddings import index_chapter

            chunks = index_chapter(
                chapter_id=chapter_id,
                folder_name=chapter["folder_name"],
                title=chapter["title"],
                markdown=chapter["markdown"],
                concepts=chapter["concepts"],
            )
        except RuntimeError as e:
            chunks = 0
            logger.warning("Skipping embeddings: %s", e)

        results.append(
            {
                "folder": chapter["folder_name"],
                "title": chapter["title"],
                "concepts": len(chapter["concepts"]),
                "chunks_indexed": chunks,
            }
        )

    return json.dumps({"chapters_extracted": len(results), "details": results}, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the full extraction + convergence loop for a competition.

    1. Extract knowledge (if not already done)
    2. Find relevant experts via RAG
    3. Run state convergence loop for each expert

    Args:
        competition: Competition name/slug.

    Returns:
        JSON summary of the evolution run.
    """
    db = get_db()
    stats = db.get_stats()
    if stats["chapters"] == 0:
        extract_knowledge(force_reindex=False)

    experts = db.list_experts()
    if not experts:
        experts = load_all_experts_from_disk()

    results = []
    for expert in experts:
        controller = LoopController.from_expert_config(expert)
        state = initial_state_vector(expert)

        def step_fn(iteration: int, current: list) -> list:
            decay = 0.7 ** iteration
            return [
                current[0] * decay,
                min(1.0, current[1] + 0.1 * (1 - decay)),
                min(1.0, current[2] + 0.05 * (1 - decay)),
                iteration / controller.max_iterations,
            ]

        loop_result = controller.run(step_fn, state)
        entry = {
            "competition": competition,
            "expert_slug": expert["slug"],
            "state_vector": loop_result["history"][-1] if loop_result["history"] else [],
            "iteration": loop_result["total_iterations"],
            "converged": loop_result["converged"],
        }
        db.save_competition_entry(entry)
        results.append(
            {
                "expert": expert["slug"],
                "converged": loop_result["converged"],
                "iterations": loop_result["total_iterations"],
                "reason": loop_result["reason"],
                "final_state": loop_result["history"][-1] if loop_result["history"] else [],
            }
        )

    return json.dumps({"competition": competition, "experts_run": len(results), "results": results}, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles content.

    Args:
        query: Natural language query.
        n_results: Number of results to return.

    Returns:
        JSON list of matching chunks with metadata and similarity scores.
    """
    try:
        from .embeddings import search

        hits = search(query, n_results=n_results)
        return json.dumps(hits, indent=2, default=str)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of expert definitions.
    """
    db = get_db()
    experts = db.list_experts()
    if not experts:
        experts = load_all_experts_from_disk()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG-informed skill selection.

    Args:
        competition: Competition name/slug.

    Returns:
        JSON with selected experts, skills, and strategy for the competition.
    """
    db = get_db()
    experts = db.list_experts()
    if not experts:
        experts = load_all_experts_from_disk()

    try:
        from .embeddings import infer_experts_for_competition

        relevant = infer_experts_for_competition(competition, experts, n_results=3)
    except RuntimeError:
        relevant = experts[:3]

    all_skills: list = []
    all_capabilities: list = []
    for expert in relevant:
        all_skills.extend(expert.get("skills", []))
        all_capabilities.extend(expert.get("capabilities", []))

    entry = {
        "competition": competition,
        "experts": [
            {
                "slug": e["slug"],
                "name": e.get("expert_name", e["slug"]),
                "relevance": e.get("relevance_score", 0.5),
                "strategy": e.get("strategy", ""),
            }
            for e in relevant
        ],
        "skills": sorted(set(all_skills)),
        "capabilities": sorted(set(all_capabilities)),
        "strategy": relevant[0].get("strategy", "") if relevant else "",
        "formula": relevant[0].get("formula", {}) if relevant else {},
    }
    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context for a competition.

    Args:
        competition: Competition name.
        description: Competition description for context retrieval.

    Returns:
        JSON with rdagent configuration and ML context.
    """
    db = get_db()
    experts = db.list_experts()

    try:
        from .embeddings import search

        context_hits = search(
            f"{competition} {description}" if description else competition,
            n_results=5,
        )
        context_chunks = [h["document"] for h in context_hits]
    except RuntimeError:
        context_chunks = []

    expert_summaries = []
    for expert in experts:
        expert_summaries.append(
            {
                "name": expert.get("expert_name", expert["slug"]),
                "capabilities": expert.get("capabilities", []),
                "strategy": expert.get("strategy", ""),
            }
        )

    return json.dumps(
        {
            "competition": competition,
            "description": description,
            "ml_context": context_chunks[:5],
            "experts_available": expert_summaries,
            "suggested_command": f"rdagent --competition {competition}",
        },
        indent=2,
    )


@mcp.tool()
def get_extraction_status() -> str:
    """Check the progress of PDF extraction.

    Returns:
        JSON list of extraction log entries.
    """
    db = get_db()
    return json.dumps(db.get_extraction_status(), indent=2, default=str)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics.

    Returns:
        JSON with counts of chapters, experts, and competition entries.
    """
    db = get_db()
    return json.dumps(db.get_stats(), indent=2)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
