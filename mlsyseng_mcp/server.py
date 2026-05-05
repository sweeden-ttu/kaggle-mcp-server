"""FastMCP server exposing MLSysEng MoE tools."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import discover_chapter_folders, extract_chapter, extract_concepts
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import (
    ConvergenceResult,
    LoopController,
    build_competition_step_fn,
    default_metric_fn,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embed: Optional[EmbeddingStore] = None
_registry: Optional[ExpertRegistry] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_embed() -> EmbeddingStore:
    global _embed
    if _embed is None:
        _embed = EmbeddingStore()
    return _embed


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


# ── extract-knowledge / evolve ────────────────────────────────────────

@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapter already exists (default: False).

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()
    embed = _get_embed()
    registry = _get_registry()

    chapters = discover_chapter_folders()
    if not chapters:
        return json.dumps({"status": "no_chapters", "message": "No chapter folders found."})

    results = []
    for ch in chapters:
        slug = ch["slug"]
        existing = db.get_chapter(slug)
        if existing and existing["status"] == "extracted" and not force_reindex:
            results.append({"slug": slug, "action": "skipped"})
            continue

        db.log_event(None, "extraction_start", slug)
        try:
            markdown, concepts = extract_chapter(ch)
            chapter_id = db.upsert_chapter(
                slug=slug,
                title=ch["name"],
                source_path=ch["folder"],
                markdown=markdown,
                status="extracted",
            )

            for concept in concepts:
                db.add_concept(chapter_id, concept["term"], concept["definition"], concept["category"])

            embed.index_chapter(slug, ch["name"], markdown)

            concept_terms = [c["term"] for c in concepts]
            expert = registry.register_from_chapter(slug, ch["name"], concept_terms, chapter_id)

            db.log_event(chapter_id, "extraction_complete", f"{len(concepts)} concepts")
            results.append({
                "slug": slug,
                "action": "extracted",
                "concepts": len(concepts),
                "expert": expert["expert_name"],
            })
        except Exception as exc:
            db.upsert_chapter(slug=slug, title=ch["name"], source_path=ch["folder"], status="failed")
            db.log_event(None, "extraction_failed", f"{slug}: {exc}")
            results.append({"slug": slug, "action": "failed", "error": str(exc)})

    return json.dumps({"status": "complete", "chapters": results}, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the full MoE pipeline: extract → search → build entry → converge.

    Shorthand for extract-knowledge + build-entry + convergence loop.

    Args:
        competition: Kaggle competition slug (default: "titanic").

    Returns:
        JSON with extraction and convergence results.
    """
    extraction = json.loads(extract_knowledge(force_reindex=False))
    entry = json.loads(build_entry(competition=competition))

    loop_config = entry.get("loop_config", {})
    controller = LoopController(
        epsilon=loop_config.get("epsilon", 0.001),
        max_iterations=loop_config.get("max_iterations", 10),
        patience=loop_config.get("patience", 3),
    )

    skills = entry.get("skills", [])
    step_fn = build_competition_step_fn(skills, competition)
    initial_state = np.zeros(8)

    result = controller.run(initial_state, step_fn, default_metric_fn)

    return json.dumps(
        {
            "extraction": extraction,
            "entry": entry,
            "convergence": result.to_dict(),
        },
        indent=2,
    )


# ── search-concepts ───────────────────────────────────────────────────

@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles content.

    Args:
        query: Natural language search query.
        n_results: Number of results to return (default: 5).

    Returns:
        JSON array of search results with text and metadata.
    """
    embed = _get_embed()
    hits = embed.search(query, n_results=n_results)

    db = _get_db()
    db_hits = db.search_concepts(query, limit=n_results)

    combined = {
        "vector_results": hits,
        "keyword_results": db_hits,
    }
    return json.dumps(combined, indent=2, default=str)


# ── list-experts ──────────────────────────────────────────────────────

@mcp.tool()
def list_experts() -> str:
    """List all chapter experts with their capabilities, skills, and strategies.

    Returns:
        JSON array of expert definitions.
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2, default=str)


# ── build-entry ───────────────────────────────────────────────────────

@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Uses RAG-informed skill selection to pick the best experts
    and generate a competition plan.

    Args:
        competition: Kaggle competition slug (default: "titanic").

    Returns:
        JSON with selected experts, skills, strategy, and loop config.
    """
    embed = _get_embed()
    registry = _get_registry()

    hits = embed.search(competition, n_results=3)
    concept_terms = []
    for hit in hits:
        words = hit.get("text", "").split()[:10]
        concept_terms.extend(words)

    experts = registry.get_experts_for_concepts(concept_terms)
    if not experts:
        experts = registry.list_experts()[:3]

    all_skills = []
    all_capabilities = []
    for e in experts:
        all_skills.extend(e.get("skills", []))
        all_capabilities.extend(e.get("capabilities", []))

    unique_skills = sorted(set(all_skills))
    unique_caps = sorted(set(all_capabilities))

    loop_config = experts[0].get("loop_config", {}) if experts else {}

    entry = {
        "competition": competition,
        "experts_used": [{"slug": e["slug"], "name": e["expert_name"]} for e in experts],
        "skills": unique_skills,
        "capabilities": unique_caps,
        "strategy": experts[0].get("strategy", "") if experts else "",
        "formula": experts[0].get("formula", {}) if experts else {},
        "loop_config": loop_config,
    }

    return json.dumps(entry, indent=2, default=str)


# ── run-rdagent ───────────────────────────────────────────────────────

@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
    max_context_chunks: int = 5,
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt from indexed chapters for use with rdagent.

    Args:
        competition: Kaggle competition name.
        description: Short description of the competition.
        max_context_chunks: Max number of context chunks to include.

    Returns:
        JSON with rdagent command and context prompt.
    """
    embed = _get_embed()

    query = f"{competition} {description}".strip()
    hits = embed.search(query, n_results=max_context_chunks)

    context_parts = []
    for hit in hits:
        meta = hit.get("metadata", {})
        chapter = meta.get("chapter_title", "Unknown")
        context_parts.append(f"## From: {chapter}\n\n{hit['text']}")

    context_prompt = "\n\n---\n\n".join(context_parts)

    rdagent_cmd = (
        f"rdagent data_science "
        f"--competition {competition} "
        f'--context "{context_prompt[:200]}..."'
    )

    return json.dumps(
        {
            "competition": competition,
            "context_prompt": context_prompt,
            "rdagent_command": rdagent_cmd,
            "chunks_used": len(hits),
        },
        indent=2,
    )


# ── get-extraction-status ─────────────────────────────────────────────

@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress.

    Returns:
        JSON with counts of total, extracted, pending, and failed chapters.
    """
    db = _get_db()
    return json.dumps(db.get_extraction_status(), indent=2)


# ── get-stats ─────────────────────────────────────────────────────────

@mcp.tool()
def get_stats() -> str:
    """System statistics including chapters, experts, concepts, and embeddings.

    Returns:
        JSON with comprehensive system statistics.
    """
    db = _get_db()
    embed = _get_embed()

    db_stats = db.get_stats()
    try:
        embed_stats = embed.get_stats()
    except Exception:
        embed_stats = {"error": "embedding store not initialized"}

    return json.dumps(
        {"database": db_stats, "embeddings": embed_stats},
        indent=2,
        default=str,
    )


# ── ask_<expert> (dynamic expert query) ───────────────────────────────

@mcp.tool()
def ask_expert(expert_slug: str, question: str, n_results: int = 3) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Slug identifier of the expert to query.
        question: Question to ask the expert.
        n_results: Number of context chunks to retrieve.

    Returns:
        JSON with expert info and relevant context from their chapter.
    """
    registry = _get_registry()
    embed = _get_embed()

    expert = registry.get_expert(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    hits = embed.search(question, n_results=n_results, chapter_filter=expert_slug)

    return json.dumps(
        {
            "expert": {
                "name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert.get("capabilities", []),
                "strategy": expert.get("strategy", ""),
            },
            "context": hits,
            "question": question,
        },
        indent=2,
        default=str,
    )


# ── index_ml_chapters ─────────────────────────────────────────────────

@mcp.tool()
def index_ml_chapters(force_reindex: bool = False) -> str:
    """Index all PDF chapters (extract + embed + store).

    Alias for extract_knowledge for compatibility.

    Args:
        force_reindex: Re-extract even if already done.

    Returns:
        JSON summary.
    """
    return extract_knowledge(force_reindex=force_reindex)


# ── search_ml_principles ──────────────────────────────────────────────

@mcp.tool()
def search_ml_principles(query: str, n_results: int = 5) -> str:
    """Semantic search over indexed knowledge base.

    Args:
        query: Natural language query.
        n_results: Number of results to return.

    Returns:
        JSON array of matching passages.
    """
    embed = _get_embed()
    hits = embed.search(query, n_results=n_results)
    return json.dumps(hits, indent=2, default=str)


# ── get_rdagent_context ───────────────────────────────────────────────

@mcp.tool()
def get_rdagent_context(
    competition_name: str,
    description: str = "",
    n_results: int = 5,
) -> str:
    """Generate context prompt for rdagent from indexed chapters.

    Args:
        competition_name: Name of the competition.
        description: Short description of the competition task.
        n_results: Number of relevant chunks to include.

    Returns:
        JSON with context prompt for rdagent.
    """
    return run_rdagent(competition=competition_name, description=description, max_context_chunks=n_results)


# ── list_indexed_chapters ─────────────────────────────────────────────

@mcp.tool()
def list_indexed_chapters() -> str:
    """List all indexed chapters.

    Returns:
        JSON array of chapter summaries.
    """
    db = _get_db()
    chapters = db.list_chapters()
    return json.dumps(chapters, indent=2, default=str)


# ── get_indexing_stats ────────────────────────────────────────────────

@mcp.tool()
def get_indexing_stats() -> str:
    """Get indexing statistics.

    Returns:
        JSON with extraction and embedding stats.
    """
    return get_stats()


# ── entry point ───────────────────────────────────────────────────────

def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
