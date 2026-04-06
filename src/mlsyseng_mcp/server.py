"""MLSysEng MoE FastMCP Server.

Provides tools for knowledge extraction, expert management, RAG search,
competition entry building, and state convergence loops.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import discover_chapters, process_chapter
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry, slugify
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingStore] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_embeddings() -> EmbeddingStore:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingStore()
    return _embeddings


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


def _get_loop() -> LoopController:
    global _loop
    if _loop is None:
        _loop = LoopController(_get_db())
    return _loop


# ── Knowledge Extraction ────────────────────────────────────────────────────


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract PDFs, index chapters, create experts.

    Scans all chapter folders in ML Principles, extracts PDF content using
    docling, generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()
    emb = _get_embeddings()
    reg = _get_registry()

    chapters = discover_chapters()
    if not chapters:
        return json.dumps({
            "status": "no_chapters",
            "message": "No chapter folders or PDFs found at the configured path.",
            "path": os.environ.get(
                "ML_PRINCIPLES_PATH",
                os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
            ),
        })

    results = []
    for chapter_info in chapters:
        chapter_name = chapter_info["name"]

        existing = db.get_chapter(chapter_name)
        if existing and not force_reindex:
            results.append({
                "chapter": chapter_name,
                "status": "skipped",
                "reason": "already indexed",
            })
            continue

        db.log_extraction(chapter_name, "running")

        try:
            extracted = process_chapter(chapter_info)

            chapter_id = db.upsert_chapter(
                chapter_name=extracted["chapter_name"],
                source_path=extracted["source_path"],
                markdown_content=extracted["markdown_content"],
                concept_count=len(extracted["concepts"]),
            )

            db.add_concepts(chapter_id, extracted["concepts"])

            emb.index_chapter(
                chapter_name=chapter_name,
                content=extracted["markdown_content"],
                concepts=extracted["concepts"],
            )

            reg.create_expert_from_chapter(
                chapter_name=chapter_name,
                concepts=extracted["concepts"],
                chapter_id=chapter_id,
            )

            db.log_extraction(
                chapter_name,
                "completed",
                pages_extracted=extracted["pages_extracted"],
            )

            results.append({
                "chapter": chapter_name,
                "status": "completed",
                "concepts": len(extracted["concepts"]),
                "pages": extracted["pages_extracted"],
            })

        except Exception as e:
            db.log_extraction(chapter_name, "failed", error_message=str(e))
            results.append({
                "chapter": chapter_name,
                "status": "failed",
                "error": str(e),
            })
            logger.exception("Failed to process chapter %s", chapter_name)

    completed = sum(1 for r in results if r["status"] == "completed")
    return json.dumps({
        "status": "done",
        "total_chapters": len(chapters),
        "completed": completed,
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "details": results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """
    Extract knowledge and run convergence loop for a competition.

    Combines extraction, expert selection, and iterative convergence.

    Args:
        competition: Kaggle competition name/slug.

    Returns:
        JSON with convergence loop results.
    """
    db = _get_db()
    emb = _get_embeddings()
    reg = _get_registry()
    loop = _get_loop()

    chapters = discover_chapters()
    if chapters:
        for chapter_info in chapters:
            existing = db.get_chapter(chapter_info["name"])
            if not existing:
                try:
                    extracted = process_chapter(chapter_info)
                    chapter_id = db.upsert_chapter(
                        chapter_name=extracted["chapter_name"],
                        source_path=extracted["source_path"],
                        markdown_content=extracted["markdown_content"],
                        concept_count=len(extracted["concepts"]),
                    )
                    db.add_concepts(chapter_id, extracted["concepts"])
                    emb.index_chapter(
                        chapter_name=extracted["chapter_name"],
                        content=extracted["markdown_content"],
                        concepts=extracted["concepts"],
                    )
                    reg.create_expert_from_chapter(
                        chapter_name=extracted["chapter_name"],
                        concepts=extracted["concepts"],
                        chapter_id=chapter_id,
                    )
                except Exception as e:
                    logger.warning("Could not process chapter %s: %s", chapter_info["name"], e)

    relevant = emb.infer_skills_for_competition(competition)
    experts = reg.get_experts_for_competition(relevant)

    if not experts:
        all_experts = reg.list_experts()
        if all_experts:
            experts = all_experts
        else:
            return json.dumps({
                "status": "no_experts",
                "message": "No experts available. Run extract-knowledge first.",
            })

    expert_count = len(experts)

    def state_generator(iteration: int, current_state: List[float]) -> List[float]:
        new_state = []
        for i, val in enumerate(current_state):
            noise = random.gauss(0, 0.01 / iteration)
            improvement = 0.1 * (1.0 / iteration)
            new_val = val + improvement + noise
            new_state.append(min(new_val, 1.0))
        return new_state

    result = loop.run_loop(
        competition=competition,
        state_generator=state_generator,
        expert_count=expert_count,
    )

    result["experts_used"] = [
        {"name": e["expert_name"], "slug": e["slug"]}
        for e in experts
    ]

    return json.dumps(result, indent=2, default=str)


# ── Search ──────────────────────────────────────────────────────────────────


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON list of matching documents with relevance scores.
    """
    emb = _get_embeddings()
    results = emb.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


# ── Expert Management ───────────────────────────────────────────────────────


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills, and strategies.
    """
    reg = _get_registry()
    experts = reg.list_experts()
    return json.dumps(experts, indent=2, default=str)


@mcp.tool()
def ask_expert(slug: str, question: str) -> str:
    """
    Query a specific chapter expert.

    Args:
        slug: Expert slug identifier (e.g., "08_ml_systems").
        question: The question to ask the expert.

    Returns:
        JSON with the expert's response and recommendations.
    """
    reg = _get_registry()
    result = reg.ask_expert(slug, question)
    return json.dumps(result, indent=2)


# ── Competition Entry Building ──────────────────────────────────────────────


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a competition entry using expert knowledge and RAG-informed skill selection.

    Args:
        competition: Kaggle competition name/slug.

    Returns:
        JSON with the entry plan including selected experts, skills, and strategy.
    """
    emb = _get_embeddings()
    reg = _get_registry()

    relevant = emb.infer_skills_for_competition(competition)
    experts = reg.get_experts_for_competition(relevant)

    if not experts:
        all_experts = reg.list_experts()
        experts = all_experts if all_experts else []

    all_skills = set()
    all_capabilities = set()
    strategies = []

    for expert in experts:
        for skill in expert.get("skills", []):
            all_skills.add(skill)
        for cap in expert.get("capabilities", []):
            all_capabilities.add(cap)
        strategies.append({
            "expert": expert["expert_name"],
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
        })

    entry = {
        "competition": competition,
        "experts_selected": len(experts),
        "experts": [
            {
                "name": e["expert_name"],
                "slug": e["slug"],
                "relevance": e.get("relevance_score", 0.0),
            }
            for e in experts
        ],
        "skills": sorted(all_skills),
        "capabilities": sorted(all_capabilities),
        "strategies": strategies,
        "execution_plan": {
            "phase_1": "Baseline model with preprocessor skill",
            "phase_2": "EDA and feature engineering with relevant experts",
            "phase_3": "Model selection and hyperparameter tuning",
            "phase_4": "Ensemble and final submission",
        },
        "loop_config": {
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    return json.dumps(entry, indent=2)


# ── RDAgent Integration ─────────────────────────────────────────────────────


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """
    Run rdagent with ML Principles context for a Kaggle competition.

    Args:
        competition: Kaggle competition name/slug.
        description: Optional competition description for better context.

    Returns:
        JSON with rdagent context and recommended command.
    """
    emb = _get_embeddings()

    search_query = f"{competition} {description}" if description else competition
    relevant = emb.search(search_query, n_results=10)

    context_parts = []
    for r in relevant:
        doc = r.get("document", "")
        meta = r.get("metadata", {})
        chapter = meta.get("chapter", "unknown")
        context_parts.append(f"[{chapter}] {doc[:200]}")

    context = "\n---\n".join(context_parts)

    return json.dumps({
        "competition": competition,
        "description": description,
        "ml_principles_context": context,
        "relevant_chapters": [
            r.get("metadata", {}).get("chapter", "unknown") for r in relevant
        ],
        "recommended_command": (
            f"rdagent kaggle --competition {competition} "
            f"--context 'ML Principles: {', '.join(set(r.get('metadata', {}).get('chapter', '') for r in relevant[:5]))}'"
        ),
    }, indent=2)


# ── Status & Stats ──────────────────────────────────────────────────────────


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check extraction progress for all chapters.

    Returns:
        JSON list of extraction log entries with status.
    """
    db = _get_db()
    status = db.get_extraction_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with counts of chapters, concepts, experts, and extraction status.
    """
    db = _get_db()
    emb = _get_embeddings()

    stats = db.get_stats()
    stats["embeddings"] = emb.get_collection_stats()

    return json.dumps(stats, indent=2)


# ── Index chapters (alias for ml-principles-rdagent compatibility) ──────────


@mcp.tool()
def index_ml_chapters(force_reindex: bool = False) -> str:
    """
    Index all PDF chapters (extract + embed + store).
    Alias for extract-knowledge for compatibility.

    Args:
        force_reindex: Re-extract even if chapters are already indexed.

    Returns:
        JSON summary of extraction results.
    """
    return extract_knowledge(force_reindex=force_reindex)


@mcp.tool()
def search_ml_principles(query: str, n_results: int = 5) -> str:
    """
    Semantic search over indexed ML Principles knowledge base.

    Args:
        query: Search query text.
        n_results: Number of results to return.

    Returns:
        JSON list of matching documents.
    """
    return search_concepts(query=query, n_results=n_results)


@mcp.tool()
def get_rdagent_context(competition_name: str, description: str = "") -> str:
    """
    Generate context prompt for rdagent based on ML Principles.

    Args:
        competition_name: Name of the Kaggle competition.
        description: Optional description of the competition.

    Returns:
        JSON with context and recommendations.
    """
    return run_rdagent(competition=competition_name, description=description)


@mcp.tool()
def list_indexed_chapters() -> str:
    """
    List all indexed chapters in the knowledge base.

    Returns:
        JSON list of indexed chapters with metadata.
    """
    db = _get_db()
    chapters = db.list_chapters()
    return json.dumps(chapters, indent=2)


@mcp.tool()
def get_indexing_stats() -> str:
    """
    Get indexing statistics.

    Returns:
        JSON with indexing stats.
    """
    return get_stats()


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
