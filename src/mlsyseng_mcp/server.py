"""FastMCP server for MLSysEng MoE - Machine Learning Systems Expert Mixture of Experts."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import MLSysEngDB
from .docling_worker import discover_chapter_folders, extract_chapter
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import ConvergenceResult, LoopController, StateVector, default_step_fn

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[MLSysEngDB] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> MLSysEngDB:
    global _db
    if _db is None:
        _db = MLSysEngDB()
    return _db


def _get_embeddings() -> EmbeddingEngine:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingEngine()
    return _embeddings


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(db=_get_db())
    return _registry


def _get_loop() -> LoopController:
    global _loop
    if _loop is None:
        _loop = LoopController(db=_get_db())
    return _loop


@mcp.tool()
def extract_knowledge(force_reindex: bool = False, base_path: Optional[str] = None) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders, extracts PDF content using docling, generates embeddings,
    and creates expert definitions for each chapter.

    Args:
        force_reindex: Force re-extraction even if chapters are already indexed (default: False)
        base_path: Optional override for the ML Principles directory path

    Returns:
        JSON string with extraction results (chapters found, extracted, experts created)
    """
    db = _get_db()
    embeddings = _get_embeddings()
    registry = _get_registry()

    chapters = discover_chapter_folders(base_path)
    if not chapters:
        return json.dumps({
            "status": "no_chapters_found",
            "message": "No chapter folders found. Set ML_PRINCIPLES_PATH environment variable.",
            "searched_path": base_path or os.environ.get("ML_PRINCIPLES_PATH", "~/Desktop/Machine Learning Principles - Chapters"),
        })

    results = {"chapters_found": len(chapters), "extracted": 0, "experts_created": 0, "errors": []}

    for chapter_info in chapters:
        ch_num = chapter_info["chapter_num"]

        existing = db.get_chapter(ch_num)
        if existing and existing.get("status") == "extracted" and not force_reindex:
            logger.info("Chapter %d already extracted, skipping", ch_num)
            continue

        try:
            extracted = extract_chapter(chapter_info)
            chapter_id = db.upsert_chapter(
                chapter_num=ch_num,
                title=extracted["title"],
                source_path=extracted["source_path"],
                markdown_content=extracted["markdown_content"],
                page_count=extracted["page_count"],
                status="extracted",
            )

            for concept in extracted["concepts"]:
                db.add_concept(
                    chapter_id=chapter_id,
                    concept_name=concept["concept_name"],
                    description=concept.get("description", ""),
                    category=concept.get("category", "general"),
                    importance=concept.get("importance", 0.5),
                )

            concept_names = [c["concept_name"] for c in extracted["concepts"]]
            try:
                embeddings.index_chapter(ch_num, extracted["title"], extracted["markdown_content"], concept_names)
            except Exception as e:
                logger.warning("Embedding indexing failed for chapter %d: %s", ch_num, e)

            expert = registry.register_expert_from_chapter(
                chapter_num=ch_num,
                title=extracted["title"],
                chapter_id=chapter_id,
                concepts=extracted["concepts"],
            )
            results["extracted"] += 1
            results["experts_created"] += 1

        except Exception as e:
            logger.error("Failed to extract chapter %d: %s", ch_num, e)
            results["errors"].append({"chapter": ch_num, "error": str(e)})

    results["status"] = "complete"
    return json.dumps(results, indent=2)


@mcp.tool()
def evolve(
    competition: str = "titanic",
    max_iterations: int = 10,
    epsilon: float = 0.001,
    patience: int = 3,
) -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Iteratively improves competition entries until state convergence is achieved.
    Exit condition: ||state[n] - state[n-1]||_2 < epsilon

    Args:
        competition: Competition name/slug (default: "titanic")
        max_iterations: Maximum iterations before stopping (default: 10)
        epsilon: Convergence threshold (default: 0.001)
        patience: Required consecutive converging iterations (default: 3)

    Returns:
        JSON string with convergence results
    """
    loop = LoopController(
        db=_get_db(),
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    result = loop.run(competition, step_fn=default_step_fn)

    return json.dumps({
        "competition": competition,
        "converged": result.converged,
        "total_iterations": result.total_iterations,
        "final_l2_norm": result.final_l2_norm,
        "final_state": result.final_state.to_list() if result.final_state else [],
        "iterations": result.iterations,
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization", "regularization techniques")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON string with search results including relevance scores and content snippets
    """
    try:
        embeddings = _get_embeddings()
        hits = embeddings.search(query, n_results=n_results)
        return json.dumps({"query": query, "results": hits}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON string with list of experts and their definitions
    """
    registry = _get_registry()
    experts = registry.list_experts()

    summary = []
    for expert in experts:
        summary.append({
            "expert_name": expert.get("expert_name"),
            "slug": expert.get("slug"),
            "capabilities": expert.get("capabilities", []),
            "skills_count": len(expert.get("skills", [])),
            "strategy": expert.get("strategy"),
            "formula": expert.get("formula", {}),
        })

    return json.dumps({"experts": summary, "total": len(summary)}, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG-informed skill selection.

    Searches the knowledge base for relevant concepts, selects appropriate experts,
    and generates a competition entry plan.

    Args:
        competition: Competition name/slug (default: "titanic")

    Returns:
        JSON string with the entry plan, selected experts, and recommended skills
    """
    try:
        embeddings = _get_embeddings()
        registry = _get_registry()
        db = _get_db()

        inferred = embeddings.infer_skills(f"Kaggle {competition} competition machine learning")

        experts = registry.get_experts_for_competition(inferred)

        if not experts:
            all_experts = registry.list_experts()
            if all_experts:
                experts = all_experts[:3]

        entry = {
            "competition": competition,
            "relevant_chapters": inferred,
            "selected_experts": [],
            "recommended_skills": set(),
            "strategy_plan": [],
        }

        for expert in experts[:5]:
            entry["selected_experts"].append({
                "name": expert.get("expert_name"),
                "slug": expert.get("slug"),
                "relevance": expert.get("relevance_score", 0.0),
                "capabilities": expert.get("capabilities", []),
                "strategy": expert.get("strategy"),
            })
            for skill in expert.get("skills", []):
                entry["recommended_skills"].add(skill)
            entry["strategy_plan"].append(expert.get("strategy", ""))

        entry["recommended_skills"] = sorted(entry["recommended_skills"])

        return json.dumps(entry, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "competition": competition})


@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
    max_iterations: int = 5,
) -> str:
    """Prepare and run rdagent with ML Principles context for a Kaggle competition.

    Generates a context prompt from the knowledge base to guide rdagent.

    Args:
        competition: Competition name (default: "titanic")
        description: Optional competition description for better context matching
        max_iterations: Maximum rdagent iterations (default: 5)

    Returns:
        JSON string with rdagent context, recommended strategies, and run configuration
    """
    try:
        embeddings = _get_embeddings()
        registry = _get_registry()

        query = f"{competition} {description}".strip() or competition
        hits = embeddings.search(query, n_results=10)

        context_parts = []
        for hit in hits:
            context_parts.append(f"[Chapter {hit['metadata'].get('title', 'Unknown')}]\n{hit['document'][:500]}")

        context = "\n\n---\n\n".join(context_parts)

        experts = registry.list_experts()
        strategies = [e.get("strategy", "") for e in experts[:5]]

        return json.dumps({
            "competition": competition,
            "description": description,
            "context_prompt": context[:5000],
            "strategies": strategies,
            "rdagent_config": {
                "competition": competition,
                "max_iterations": max_iterations,
                "context_length": len(context),
                "relevant_chapters": len(hits),
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "competition": competition})


@mcp.tool()
def get_extraction_status() -> str:
    """Check the status of PDF extraction and indexing progress.

    Returns:
        JSON string with extraction status for each chapter
    """
    db = _get_db()
    chapters = db.get_all_chapters()

    status = {
        "total_chapters": len(chapters),
        "extracted": sum(1 for c in chapters if c.get("status") == "extracted"),
        "pending": sum(1 for c in chapters if c.get("status") == "pending"),
        "chapters": [
            {
                "chapter_num": c["chapter_num"],
                "title": c["title"],
                "status": c.get("status", "unknown"),
                "page_count": c.get("page_count", 0),
                "extracted_at": c.get("extracted_at"),
            }
            for c in chapters
        ],
    }

    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapters, experts, concepts, and embeddings.

    Returns:
        JSON string with comprehensive system statistics
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        embeddings = _get_embeddings()
        embed_stats = embeddings.get_stats()
    except Exception:
        embed_stats = {"error": "Embedding engine not available"}

    return json.dumps({
        "database": db_stats,
        "embeddings": embed_stats,
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
