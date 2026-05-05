"""
FastMCP server for the MLSysEng Mixture of Experts system.

Provides tools for:
- Knowledge extraction from ML Principles PDFs
- Semantic search over ML concepts
- Expert listing and querying
- Competition entry building with RAG-informed skill selection
- State convergence loop execution
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import MoEDatabase
from mlsyseng_mcp.docling_worker import run_extraction
from mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_mcp.expert_registry import (
    get_experts_for_competition,
    register_experts_from_db,
)
from mlsyseng_mcp.loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MoEDatabase] = None
_embeddings: Optional[EmbeddingEngine] = None
_loop_controller: Optional[LoopController] = None


def _get_db() -> MoEDatabase:
    global _db
    if _db is None:
        _db = MoEDatabase()
    return _db


def _get_embeddings() -> EmbeddingEngine:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingEngine()
    return _embeddings


def _get_loop_controller() -> LoopController:
    global _loop_controller
    if _loop_controller is None:
        _loop_controller = LoopController(_get_db())
    return _loop_controller


# ── Knowledge Extraction ─────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if already done (default: False)

    Returns:
        JSON with extraction results
    """
    db = _get_db()
    emb = _get_embeddings()

    extraction_result = run_extraction(db, force_reindex=force_reindex)

    chapters = db.list_chapters()
    indexed_chunks = 0
    for ch in chapters:
        if ch.get("status") == "extracted" and ch.get("markdown"):
            count = emb.index_chapter(
                chapter_id=ch["chapter_id"],
                title=ch["title"],
                markdown=ch["markdown"],
                concepts=ch.get("concepts"),
            )
            indexed_chunks += count

    experts = register_experts_from_db(db)

    return json.dumps(
        {
            "status": "success",
            "extraction": extraction_result,
            "indexed_chunks": indexed_chunks,
            "experts_registered": len(experts),
        },
        indent=2,
    )


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """
    Run the full knowledge extraction and convergence loop for a competition.

    Equivalent to extract_knowledge + build_entry + run convergence loop.

    Args:
        competition: Competition name (default: "titanic")

    Returns:
        JSON with evolution results including convergence state
    """
    db = _get_db()
    emb = _get_embeddings()
    lc = _get_loop_controller()

    run_extraction(db, force_reindex=False)

    chapters = db.list_chapters()
    for ch in chapters:
        if ch.get("status") == "extracted" and ch.get("markdown"):
            emb.index_chapter(
                chapter_id=ch["chapter_id"],
                title=ch["title"],
                markdown=ch["markdown"],
                concepts=ch.get("concepts"),
            )

    register_experts_from_db(db)
    experts = get_experts_for_competition(db, emb, competition, top_k=3)

    if not experts:
        return json.dumps({
            "status": "no_experts",
            "message": "No experts available. Run extract_knowledge first.",
        })

    def refine_fn(
        iteration: int,
        expert_list: List[Dict[str, Any]],
        current_state: List[float],
    ) -> List[float]:
        new_state = []
        for i, expert in enumerate(expert_list):
            base = current_state[i] if i < len(current_state) else 0.0
            relevance = expert.get("relevance_score", 0.5)
            increment = relevance * (0.5 ** (iteration + 1))
            new_state.append(base + increment)
        return new_state

    loop_result = lc.run_loop(competition, experts, refine_fn)

    return json.dumps(
        {
            "status": "success",
            "competition": competition,
            "experts_used": [
                {
                    "name": e["expert_name"],
                    "slug": e["slug"],
                    "relevance": e.get("relevance_score"),
                }
                for e in experts
            ],
            "convergence": {
                "loop_id": loop_result["loop_id"],
                "iterations": loop_result["iteration"],
                "converged": loop_result["converged"],
                "final_l2_norm": loop_result["l2_norm"],
                "state_vector": loop_result["state_vector"],
            },
        },
        indent=2,
    )


# ── Search ────────────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with ranked search results
    """
    emb = _get_embeddings()
    results = emb.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


# ── Experts ───────────────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON array of expert definitions
    """
    db = _get_db()
    experts = db.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """
    Query a specific chapter expert by slug.

    Args:
        expert_slug: The expert's slug identifier (e.g., "08_ml_systems")
        question: The question to ask

    Returns:
        JSON with expert's context and related knowledge
    """
    db = _get_db()
    emb = _get_embeddings()

    expert = db.get_expert_by_slug(expert_slug)
    if expert is None:
        return json.dumps({
            "error": f"Expert '{expert_slug}' not found",
            "available": [e["slug"] for e in db.list_experts()],
        })

    search_results = emb.search(
        question,
        n_results=5,
        chapter_filter=expert.get("chapter_id"),
    )

    return json.dumps(
        {
            "expert": {
                "name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert["capabilities"],
                "strategy": expert["strategy"],
            },
            "question": question,
            "relevant_knowledge": search_results,
        },
        indent=2,
    )


# ── Competition Entry ─────────────────────────────────────────────


@mcp.tool()
def build_entry(competition: str = "titanic", top_k: int = 3) -> str:
    """
    Build a competition entry using expert knowledge and RAG-informed skill selection.

    Selects the most relevant experts for the competition, infers skills,
    and generates an entry plan.

    Args:
        competition: Competition name/description (default: "titanic")
        top_k: Number of experts to use (default: 3)

    Returns:
        JSON with entry plan, selected experts, and recommended skills
    """
    db = _get_db()
    emb = _get_embeddings()

    experts = get_experts_for_competition(db, emb, competition, top_k=top_k)

    if not experts:
        return json.dumps({
            "status": "no_experts",
            "message": "No experts available. Run extract-knowledge first.",
        })

    all_skills: List[str] = []
    all_concepts: List[str] = []
    for e in experts:
        for s in e.get("skills", []):
            if s not in all_skills:
                all_skills.append(s)
        for c in e.get("matched_concepts", []):
            if c not in all_concepts:
                all_concepts.append(c)

    entry = {
        "status": "success",
        "competition": competition,
        "experts": [
            {
                "name": e["expert_name"],
                "slug": e["slug"],
                "relevance_score": e.get("relevance_score", 0.0),
                "capabilities": e["capabilities"],
                "matched_concepts": e.get("matched_concepts", []),
            }
            for e in experts
        ],
        "recommended_skills": all_skills,
        "matched_concepts": all_concepts,
        "entry_plan": {
            "strategy": experts[0].get("strategy", "") if experts else "",
            "formula": experts[0].get("formula", {}) if experts else {},
            "loop_config": experts[0].get("loop_config", {}) if experts else {},
            "steps": [
                "Download competition data",
                "Run EDA with expert-recommended techniques",
                "Apply feature engineering from matched concepts",
                "Train models using selected skills",
                "Evaluate with cross-validation",
                "Run convergence loop until epsilon convergence",
                "Generate submission",
            ],
        },
    }
    return json.dumps(entry, indent=2)


# ── RDAgent Integration ───────────────────────────────────────────


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """
    Run rdagent with ML Principles context for a Kaggle competition.

    Generates a context prompt enriched with expert knowledge and
    prepares the rdagent command.

    Args:
        competition: Competition name
        description: Optional competition description

    Returns:
        JSON with rdagent context prompt and command
    """
    db = _get_db()
    emb = _get_embeddings()

    query = f"{competition} {description}".strip()
    search_results = emb.search(query, n_results=10)
    experts = get_experts_for_competition(db, emb, query, top_k=3)

    context_parts: List[str] = [
        f"# ML Principles Context for: {competition}",
        "",
    ]

    if experts:
        context_parts.append("## Recommended Experts")
        for e in experts:
            context_parts.append(
                f"- **{e['expert_name']}**: {', '.join(e.get('capabilities', []))}"
            )
        context_parts.append("")

    if search_results:
        context_parts.append("## Relevant Knowledge")
        for hit in search_results[:5]:
            doc_preview = hit["document"][:200] + "..." if len(hit["document"]) > 200 else hit["document"]
            context_parts.append(f"- [{hit['title']}] (score={hit['score']:.3f}): {doc_preview}")
        context_parts.append("")

    context_prompt = "\n".join(context_parts)

    return json.dumps(
        {
            "status": "success",
            "competition": competition,
            "context_prompt": context_prompt,
            "experts": [e["expert_name"] for e in experts],
            "command": f'rdagent --competition "{competition}" --context-file ml_context.md',
        },
        indent=2,
    )


# ── Status & Stats ────────────────────────────────────────────────


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check extraction progress for ML Principles chapters.

    Returns:
        JSON with extraction status counts
    """
    db = _get_db()
    status = db.get_extraction_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with database and embedding statistics
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        emb = _get_embeddings()
        emb_stats = emb.get_stats()
    except Exception:
        emb_stats = {"error": "Embedding engine not available"}

    return json.dumps(
        {
            "database": db_stats,
            "embeddings": emb_stats,
        },
        indent=2,
    )


# ── Entry Point ───────────────────────────────────────────────────


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
