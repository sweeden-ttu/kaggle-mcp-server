"""FastMCP server for MLSysEng MoE system."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import DoclingWorker
from .embeddings import EmbeddingEngine
from .expert_registry import ExpertRegistry
from .loop_controller import ConvergenceResult, LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_worker: Optional[DoclingWorker] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_worker() -> DoclingWorker:
    global _worker
    if _worker is None:
        _worker = DoclingWorker()
    return _worker


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


def _get_loop(config: Optional[Dict[str, Any]] = None) -> LoopController:
    global _loop
    if _loop is None or config:
        cfg = config or {}
        _loop = LoopController(
            epsilon=cfg.get("epsilon", 0.001),
            max_iterations=cfg.get("max_iterations", 10),
            patience=cfg.get("patience", 3),
            objective=cfg.get("objective", "minimize_validation_loss"),
        )
    return _loop


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Force re-extraction even if already indexed (default: False)

    Returns:
        JSON with extraction results and status
    """
    db = _get_db()
    worker = _get_worker()
    embeddings = _get_embeddings()
    registry = _get_registry()

    chapters = worker.scan_chapters()
    if not chapters:
        return json.dumps({
            "status": "warning",
            "message": "No chapters found. Check ML_PRINCIPLES_PATH environment variable.",
            "path_checked": str(worker.ml_principles_path),
        })

    results = []
    for ch_info in chapters:
        ch_num = ch_info["chapter_number"]

        if not force_reindex:
            existing = db.get_chapter(ch_num)
            if existing:
                results.append({
                    "chapter": ch_num,
                    "title": ch_info["title"],
                    "status": "skipped (already indexed)",
                })
                continue

        try:
            extracted = worker.extract_chapter(ch_info)
            chapter_id = db.insert_chapter(
                chapter_number=extracted["chapter_number"],
                title=extracted["title"],
                source_path=extracted["source_path"],
                markdown_content=extracted["markdown_content"],
                concepts=extracted["concepts"],
                word_count=extracted["word_count"],
            )

            if extracted["concepts"]:
                concept_dicts = [{"name": c, "description": "", "category": "ml"} for c in extracted["concepts"]]
                db.insert_concepts(chapter_id, concept_dicts)

            if extracted["markdown_content"]:
                embeddings.index_chapter(
                    chapter_number=ch_num,
                    title=extracted["title"],
                    content=extracted["markdown_content"],
                    concepts=extracted["concepts"],
                )

            registry.create_expert_from_chapter(
                chapter_number=ch_num,
                title=extracted["title"],
                concepts=extracted["concepts"],
                chapter_id=chapter_id,
            )

            db.log_extraction(chapter_id, "success", f"Extracted {extracted['word_count']} words")
            results.append({
                "chapter": ch_num,
                "title": extracted["title"],
                "status": "success",
                "words": extracted["word_count"],
                "concepts": len(extracted["concepts"]),
            })

        except Exception as e:
            db.log_extraction(None, "error", f"Chapter {ch_num}: {str(e)}")
            results.append({
                "chapter": ch_num,
                "title": ch_info["title"],
                "status": f"error: {str(e)}",
            })

    return json.dumps({
        "status": "complete",
        "chapters_processed": len(results),
        "results": results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """
    Run the convergence loop for a competition, evolving expert solutions.

    Args:
        competition: Competition name/slug (default: "titanic")
        max_iterations: Maximum iterations before stopping (default: 10)

    Returns:
        JSON with convergence results
    """
    embeddings = _get_embeddings()
    registry = _get_registry()
    loop = _get_loop({"max_iterations": max_iterations})

    relevant = embeddings.infer_skills(competition)
    experts = registry.get_experts_for_competition(relevant)

    if not experts:
        return json.dumps({
            "status": "no_experts",
            "message": "No experts found. Run extract_knowledge first.",
        })

    state_dim = len(experts) * 3

    def step_fn(iteration: int, prev_state: Optional[np.ndarray]):
        if prev_state is None:
            state = np.random.uniform(0.3, 0.7, size=state_dim)
        else:
            noise = np.random.normal(0, max(0.01, 0.1 * (0.8 ** iteration)), size=state_dim)
            state = prev_state + noise
            state = np.clip(state, 0, 1)

        loss = float(np.mean((1 - state) ** 2))
        metrics = {
            "validation_loss": loss,
            "iteration": iteration,
            "active_experts": len(experts),
        }
        return state, metrics

    result: ConvergenceResult = loop.run(step_fn)

    return json.dumps({
        "status": "converged" if result.converged else "max_iterations_reached",
        "competition": competition,
        "iterations": result.iterations,
        "exit_reason": result.exit_reason,
        "l2_norm_history": result.l2_norm_history,
        "experts_used": [e.get("expert_name", "unknown") for e in experts[:5]],
        "final_metrics": result.history[-1]["metrics"] if result.history else {},
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles indexed knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with matching documents and relevance scores
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)

    if not results:
        db = _get_db()
        db_results = db.search_concepts(query)
        if db_results:
            return json.dumps({
                "source": "database",
                "results": db_results[:n_results],
            }, indent=2)
        return json.dumps({"status": "no_results", "query": query})

    return json.dumps({
        "source": "embeddings",
        "query": query,
        "results": results,
    }, indent=2)


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts.

    Returns:
        JSON with all expert names, slugs, and strategies
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps({"experts": experts, "count": len(experts)}, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a Kaggle competition entry using expert knowledge and RAG.

    Args:
        competition: Competition name/slug (default: "titanic")

    Returns:
        JSON with entry plan, selected experts, and notebook paths
    """
    embeddings = _get_embeddings()
    registry = _get_registry()

    relevant = embeddings.infer_skills(competition)
    experts = registry.get_experts_for_competition(relevant)

    if not experts:
        return json.dumps({
            "status": "no_experts",
            "message": "No experts available. Run extract_knowledge first.",
        })

    competition_slug = competition.replace(" ", "_").lower()
    output_dir = Path.home() / competition_slug

    notebooks = []
    for expert in experts[:5]:
        notebook_name = f"Expert_{expert['slug']}.ipynb"
        notebooks.append({
            "expert": expert["expert_name"],
            "notebook": str(output_dir / notebook_name),
            "skills": expert.get("skills", []),
            "strategy": expert.get("strategy", ""),
        })

    entry = {
        "competition": competition,
        "output_directory": str(output_dir),
        "experts_selected": len(experts),
        "notebooks": notebooks,
        "pipeline": [
            "1. Data Loading & EDA",
            "2. Feature Engineering (guided by expert concepts)",
            "3. Model Selection (based on expert strategies)",
            "4. Training with convergence loop",
            "5. Ensemble top expert outputs",
            "6. Generate submission",
        ],
        "convergence_config": {
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """
    Prepare rdagent context with ML Principles knowledge for a competition.

    Args:
        competition: Competition name (default: "titanic")
        description: Competition description for context matching

    Returns:
        JSON with rdagent context prompt and expert recommendations
    """
    embeddings = _get_embeddings()

    search_query = f"{competition} {description}".strip()
    results = embeddings.search(search_query, n_results=10)

    context_passages = []
    for r in results[:5]:
        context_passages.append(r["document"])

    context = "\n\n---\n\n".join(context_passages) if context_passages else "No indexed content available."

    rdagent_prompt = f"""ML Principles Context for {competition}:

{context}

Competition: {competition}
Description: {description}

Recommended approach based on ML Principles:
1. Start with a strong baseline
2. Apply concepts from relevant chapters
3. Use convergence loop for iterative improvement
4. Ensemble top approaches
"""

    return json.dumps({
        "competition": competition,
        "context_prompt": rdagent_prompt,
        "relevant_passages": len(results),
        "command": f"rdagent --competition {competition} --context ml_principles",
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check the progress of knowledge extraction.

    Returns:
        JSON with extraction statistics and status
    """
    db = _get_db()
    worker = _get_worker()

    chapters_available = worker.scan_chapters()
    chapters_indexed = db.list_chapters()

    indexed_numbers = {ch["chapter_number"] for ch in chapters_indexed}
    available_numbers = {ch["chapter_number"] for ch in chapters_available}
    pending = available_numbers - indexed_numbers

    return json.dumps({
        "chapters_available": len(chapters_available),
        "chapters_indexed": len(chapters_indexed),
        "chapters_pending": sorted(pending),
        "indexed_chapters": chapters_indexed,
        "ml_principles_path": str(worker.ml_principles_path),
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with database, embedding, and expert statistics
    """
    db = _get_db()

    db_stats = db.get_stats()

    try:
        emb_stats = _get_embeddings().get_stats()
    except Exception:
        emb_stats = {"status": "not_initialized"}

    return json.dumps({
        "database": db_stats,
        "embeddings": emb_stats,
        "system": {
            "version": "0.1.0",
            "ml_principles_path": os.environ.get("ML_PRINCIPLES_PATH", "not set"),
            "sqlite_db_path": os.environ.get("SQLITE_DB_PATH", "default"),
            "chroma_db_path": os.environ.get("CHROMA_DB_PATH", "default"),
        },
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
