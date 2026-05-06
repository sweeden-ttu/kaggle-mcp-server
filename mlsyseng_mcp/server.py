"""FastMCP server for MLSysEng MoE system.

Exposes MCP tools for knowledge extraction, expert management, RAG search,
competition entry building, and convergence loop control.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import MLSysEngDatabase
from mlsyseng_mcp.docling_worker import extract_all_chapters, scan_chapters
from mlsyseng_mcp.embeddings import EmbeddingStore
from mlsyseng_mcp.expert_registry import (
    build_expert_definition,
    register_experts_from_chapters,
    save_expert_json,
    select_experts_for_competition,
)
from mlsyseng_mcp.loop_controller import LoopConfig, LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MLSysEngDatabase] = None
_embeddings: Optional[EmbeddingStore] = None
_loop_controller: Optional[LoopController] = None


def _get_db() -> MLSysEngDatabase:
    global _db
    if _db is None:
        _db = MLSysEngDatabase()
    return _db


def _get_embeddings() -> EmbeddingStore:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingStore()
    return _embeddings


def _get_loop_controller() -> LoopController:
    global _loop_controller
    if _loop_controller is None:
        _loop_controller = LoopController(db=_get_db())
    return _loop_controller


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans the ML Principles directory, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract and re-index even if already done (default: False)

    Returns:
        JSON summary of extraction results
    """
    db = _get_db()
    emb = _get_embeddings()

    results = extract_all_chapters(force_reindex=force_reindex, db=db)

    indexed_count = 0
    for chap in results:
        if chap.get("status") == "extracted" or force_reindex:
            content = chap.get("content_md", "")
            if not content:
                stored = db.get_chapter(chap["chapter_id"])
                content = stored.get("content_md", "") if stored else ""

            if content:
                try:
                    count = emb.index_chapter(
                        chapter_id=chap["chapter_id"],
                        title=chap.get("title", ""),
                        content=content,
                        concepts=chap.get("concepts", []),
                        force=force_reindex,
                    )
                    indexed_count += count
                except Exception as e:
                    logger.error("Failed to index chapter %s: %s", chap["chapter_id"], e)

    chapters = db.list_chapters()
    experts = register_experts_from_chapters(chapters, db=db)

    for expert in experts:
        try:
            save_expert_json(expert)
        except Exception as e:
            logger.error("Failed to save expert JSON for %s: %s", expert.get("slug"), e)

    return json.dumps({
        "status": "complete",
        "chapters_scanned": len(results),
        "chapters_extracted": sum(1 for r in results if r.get("status") == "extracted"),
        "chapters_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "chunks_indexed": indexed_count,
        "experts_created": len(experts),
        "details": [{
            "chapter_id": r.get("chapter_id"),
            "title": r.get("title"),
            "status": r.get("status"),
        } for r in results],
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Iteratively refines the competition entry using expert recommendations
    and checks for state convergence.

    Args:
        competition: Competition name (default: "titanic")
        max_iterations: Maximum number of iterations (default: 10)

    Returns:
        JSON summary of convergence loop execution
    """
    db = _get_db()
    emb = _get_embeddings()
    lc = _get_loop_controller()
    lc.config.max_iterations = max_iterations

    experts = db.list_experts()
    if not experts:
        return json.dumps({"error": "No experts registered. Run extract_knowledge first."})

    selected = select_experts_for_competition(competition, experts, emb)

    def step_fn(iteration: int, current_state: List[float]) -> List[float]:
        new_state = list(current_state)
        decay = 0.5 ** iteration
        for i, expert in enumerate(selected):
            if i < len(new_state):
                new_state[i] += decay * (1.0 - new_state[i])
            else:
                new_state.append(decay)
        return new_state

    initial = [0.0] * max(len(selected), 1)
    summary = lc.run_loop(competition, step_fn, initial_state=initial)

    summary["selected_experts"] = [
        {"name": e.get("expert_name"), "slug": e.get("slug")}
        for e in selected
    ]

    return json.dumps(summary, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON list of matching content with metadata and relevance scores
    """
    emb = _get_embeddings()
    hits = emb.search_concepts(query, n_results=n_results)
    return json.dumps(hits, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of all experts with their capabilities, skills, and configuration
    """
    db = _get_db()
    experts = db.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG-informed skill selection.

    Selects the most relevant experts, gathers their skills and strategies,
    and produces a structured competition entry plan.

    Args:
        competition: Competition name (default: "titanic")

    Returns:
        JSON competition entry plan with expert recommendations
    """
    db = _get_db()
    emb = _get_embeddings()

    experts = db.list_experts()
    if not experts:
        return json.dumps({"error": "No experts registered. Run extract_knowledge first."})

    selected = select_experts_for_competition(competition, experts, emb)

    context = emb.get_rdagent_context(competition)

    all_skills = set()
    all_capabilities = []
    strategies = []
    for expert in selected:
        skills = expert.get("skills", [])
        if isinstance(skills, str):
            skills = json.loads(skills)
        all_skills.update(skills)
        caps = expert.get("capabilities", [])
        if isinstance(caps, str):
            caps = json.loads(caps)
        all_capabilities.extend(caps)
        strategies.append({
            "expert": expert.get("expert_name"),
            "strategy": expert.get("strategy", ""),
        })

    entry = {
        "competition": competition,
        "selected_experts": [
            {
                "name": e.get("expert_name"),
                "slug": e.get("slug"),
                "capabilities": e.get("capabilities", []),
            }
            for e in selected
        ],
        "combined_skills": sorted(all_skills),
        "combined_capabilities": list(set(all_capabilities)),
        "strategies": strategies,
        "loop_config": selected[0].get("loop_config", {}) if selected else {},
        "context_snippet": context[:500] if context else "",
        "recommended_pipeline": [
            "1. Download competition data",
            "2. EDA with expert-recommended techniques",
            "3. Feature engineering using mapped skills",
            "4. Build baseline model",
            "5. Iterate with convergence loop",
            "6. Ensemble top models",
            "7. Generate submission",
        ],
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Run rdagent with ML Principles context for a competition.

    Generates a context prompt from relevant ML chapters and prepares
    rdagent configuration.

    Args:
        competition: Competition name
        description: Optional competition description for better context retrieval

    Returns:
        JSON with rdagent context and configuration
    """
    emb = _get_embeddings()
    context = emb.get_rdagent_context(competition, description)

    return json.dumps({
        "competition": competition,
        "context": context,
        "rdagent_config": {
            "scenario": "kaggle",
            "competition": competition,
            "ml_principles_context": True,
        },
        "command": f"rdagent --scenario kaggle --competition {competition}",
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the status of knowledge extraction.

    Returns:
        JSON with extraction progress details
    """
    db = _get_db()
    chapters = db.list_chapters()
    log_entries = db.get_extraction_log()

    return json.dumps({
        "total_chapters": len(chapters),
        "chapters": [
            {
                "chapter_id": c.get("chapter_id"),
                "title": c.get("title"),
                "extracted_at": c.get("extracted_at"),
                "concept_count": len(c.get("concepts", [])),
            }
            for c in chapters
        ],
        "recent_events": log_entries[-10:] if log_entries else [],
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with database stats, embedding stats, and system info
    """
    db = _get_db()
    emb = _get_embeddings()

    db_stats = db.get_stats()
    try:
        emb_stats = emb.get_stats()
    except Exception:
        emb_stats = {"error": "Embedding store not initialized"}

    return json.dumps({
        "database": db_stats,
        "embeddings": emb_stats,
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str, n_results: int = 3) -> str:
    """Query a specific chapter expert for recommendations.

    Args:
        expert_slug: Slug of the expert to query (e.g., "08_ml_systems")
        question: Question to ask the expert
        n_results: Number of relevant passages to retrieve (default: 3)

    Returns:
        JSON with expert info and relevant knowledge passages
    """
    db = _get_db()
    emb = _get_embeddings()

    expert = db.get_expert_by_slug(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    chapter_id = expert.get("chapter_id", "")
    hits = emb.search(question, n_results=n_results, chapter_filter=chapter_id)

    return json.dumps({
        "expert": {
            "name": expert.get("expert_name"),
            "slug": expert.get("slug"),
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
        },
        "relevant_passages": hits,
        "recommendation": f"Based on {expert.get('expert_name', 'unknown')} knowledge, "
                          f"consider the following approach: {expert.get('strategy', '')}",
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
