"""FastMCP server for the MLSysEng MoE system.

Provides tools for knowledge extraction, expert management,
competition entry building, and RAG-informed skill selection.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import extract_all_chapters, scan_chapters
from .embeddings import EmbeddingStore
from .expert_registry import (
    create_expert_from_chapter,
    infer_experts_for_competition,
    register_experts_from_db,
    save_expert_json,
)
from .loop_controller import LoopController, LoopState, build_competition_step

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-mcp")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingStore] = None


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


@mcp.tool(name="extract-knowledge")
def extract_knowledge(
    force_reindex: bool = False,
    principles_path: Optional[str] = None,
) -> str:
    """Extract PDFs, index chapters, create experts, and build embeddings.

    This performs the full knowledge extraction pipeline:
    1. Scan all chapter folders in ML Principles
    2. Extract PDF content using docling
    3. Generate embeddings
    4. Create expert definitions

    Args:
        force_reindex: Re-extract even if already indexed (default: False)
        principles_path: Override the ML Principles path (optional)

    Returns:
        JSON status of extraction with per-chapter results
    """
    db = _get_db()
    embeddings = _get_embeddings()

    extraction_results = extract_all_chapters(
        principles_path=principles_path, db=db, force_reindex=force_reindex
    )

    expert_results = register_experts_from_db(db)

    try:
        index_result = embeddings.index_chapters(db, force=force_reindex)
    except Exception as e:
        index_result = {"status": "error", "error": str(e)}

    return json.dumps(
        {
            "extraction": extraction_results,
            "experts_registered": expert_results,
            "embedding_index": index_result,
        },
        indent=2,
    )


@mcp.tool(name="evolve")
def evolve(
    competition: str = "titanic",
    max_iterations: int = 10,
    epsilon: float = 0.001,
    patience: int = 3,
    principles_path: Optional[str] = None,
) -> str:
    """Run the full evolve pipeline: extract knowledge, then run convergence loop.

    Args:
        competition: Kaggle competition name/slug (default: "titanic")
        max_iterations: Maximum convergence loop iterations (default: 10)
        epsilon: Convergence threshold (default: 0.001)
        patience: Consecutive converging iterations required (default: 3)
        principles_path: Override the ML Principles path (optional)

    Returns:
        JSON with extraction results and convergence loop output
    """
    db = _get_db()
    embeddings = _get_embeddings()

    extraction_results = extract_all_chapters(
        principles_path=principles_path, db=db, force_reindex=False
    )
    register_experts_from_db(db)
    try:
        embeddings.index_chapters(db, force=False)
    except Exception:
        pass

    experts = infer_experts_for_competition(competition, db, embeddings, top_k=3)
    if not experts:
        experts = db.list_experts()[:3]

    controller = LoopController(
        objective="minimize_validation_loss",
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    def step_fn(iteration: int, prev_state: Optional[List[float]]) -> LoopState:
        return build_competition_step(experts, competition, iteration, prev_state)

    result = controller.run(step_fn)

    return json.dumps(
        {
            "competition": competition,
            "extraction_summary": {
                "chapters_processed": len(extraction_results),
            },
            "convergence": controller.to_dict(result),
            "experts_used": [e.get("expert_name") for e in experts],
        },
        indent=2,
    )


@mcp.tool(name="search-concepts")
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON list of matching content chunks with similarity scores
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


@mcp.tool(name="list-experts")
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills.

    Returns:
        JSON list of all expert definitions
    """
    db = _get_db()
    experts = db.list_experts()
    summary = []
    for e in experts:
        summary.append({
            "expert_name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e.get("capabilities", []),
            "skills_count": len(e.get("skills", [])),
            "strategy": e.get("strategy", ""),
            "formula": e.get("formula", {}),
        })
    return json.dumps(summary, indent=2)


@mcp.tool(name="build-entry")
def build_entry(
    competition: str = "titanic",
    top_k_experts: int = 3,
    max_iterations: int = 10,
    epsilon: float = 0.001,
    patience: int = 3,
) -> str:
    """Build a competition entry using expert knowledge and convergence loop.

    Uses RAG-informed skill selection to pick the best experts,
    then runs the state convergence loop to refine the entry.

    Args:
        competition: Kaggle competition name (default: "titanic")
        top_k_experts: Number of experts to use (default: 3)
        max_iterations: Max convergence iterations (default: 10)
        epsilon: Convergence threshold (default: 0.001)
        patience: Patience for convergence (default: 3)

    Returns:
        JSON with selected experts, convergence results, and entry metadata
    """
    db = _get_db()
    embeddings = _get_embeddings()

    experts = infer_experts_for_competition(competition, db, embeddings, top_k=top_k_experts)
    if not experts:
        experts = db.list_experts()[:top_k_experts]

    controller = LoopController(
        objective="minimize_validation_loss",
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    def step_fn(iteration: int, prev_state: Optional[List[float]]) -> LoopState:
        return build_competition_step(experts, competition, iteration, prev_state)

    result = controller.run(step_fn)

    notebook_dir = os.path.expanduser(f"~/{competition}")
    os.makedirs(notebook_dir, exist_ok=True)

    expert_notebooks = []
    for expert in experts:
        slug = expert.get("slug", "unknown")
        nb_path = os.path.join(notebook_dir, f"Expert_{slug}.ipynb")
        _generate_notebook_stub(nb_path, expert, competition, result)
        expert_notebooks.append(nb_path)

    return json.dumps(
        {
            "competition": competition,
            "experts_selected": [
                {
                    "name": e.get("expert_name"),
                    "slug": e.get("slug"),
                    "skills": e.get("skills", [])[:5],
                }
                for e in experts
            ],
            "convergence": controller.to_dict(result),
            "notebooks": expert_notebooks,
        },
        indent=2,
    )


def _generate_notebook_stub(
    path: str, expert: Dict[str, Any], competition: str, loop_result
) -> None:
    """Generate a minimal Jupyter notebook stub for an expert's contribution."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {competition.title()} - Expert: {expert.get('expert_name', 'Unknown')}\n",
                f"\n**Strategy**: {expert.get('strategy', 'N/A')}\n",
                f"\n**Capabilities**: {', '.join(expert.get('capabilities', []))}\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\nimport numpy as np\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import accuracy_score, f1_score\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Competition: {competition}\n",
                f"# Expert: {expert.get('expert_name', 'Unknown')}\n",
                f"# Objective: {expert.get('formula', {}).get('objective', 'minimize_validation_loss')}\n",
                "\n",
                "# Load data\n",
                f"# train = pd.read_csv('{competition}/train.csv')\n",
                f"# test = pd.read_csv('{competition}/test.csv')\n",
            ],
        },
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    with open(path, "w") as f:
        json.dump(notebook, f, indent=2)


@mcp.tool(name="run-rdagent")
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
    n_context_results: int = 5,
) -> str:
    """Prepare rdagent context using ML Principles knowledge.

    Generates a context prompt for rdagent data_science competitions
    informed by the indexed ML Principles chapters.

    Args:
        competition: Kaggle competition name (default: "titanic")
        description: Brief competition description
        n_context_results: Number of RAG results to use (default: 5)

    Returns:
        JSON with rdagent context, recommended strategies, and command
    """
    db = _get_db()
    embeddings = _get_embeddings()

    query = f"{competition} {description}".strip()
    search_results = embeddings.search(query, n_results=n_context_results)

    experts = infer_experts_for_competition(query, db, embeddings, top_k=3)

    context_parts = []
    for r in search_results:
        context_parts.append(
            f"[Chapter {r.get('chapter_num', '?')}: {r.get('title', '')}] "
            f"{r.get('content', '')[:300]}"
        )

    strategies = []
    for e in experts:
        strategies.append({
            "expert": e.get("expert_name"),
            "strategy": e.get("strategy"),
            "formula": e.get("formula"),
        })

    rdagent_cmd = (
        f"rdagent data_science --competition {competition} "
        f"--context 'ML Principles guided: {len(search_results)} relevant sections found'"
    )

    return json.dumps(
        {
            "competition": competition,
            "context": context_parts,
            "recommended_strategies": strategies,
            "rdagent_command": rdagent_cmd,
        },
        indent=2,
    )


def _ask_expert(slug: str, question: str, n_results: int = 3) -> str:
    """Query a specific chapter expert."""
    db = _get_db()
    expert = db.get_expert(slug)
    if not expert:
        return json.dumps({"error": f"Expert '{slug}' not found"})

    embeddings = _get_embeddings()
    chapter_num = None
    chapter = None
    if expert.get("chapter_id"):
        chapters = db.list_chapters()
        for ch in chapters:
            if ch["id"] == expert["chapter_id"]:
                chapter_num = ch["chapter_num"]
                chapter = ch
                break

    search_results = embeddings.search(question, n_results=n_results, chapter_filter=chapter_num)

    return json.dumps(
        {
            "expert": expert["expert_name"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy"),
            "formula": expert.get("formula"),
            "relevant_content": search_results,
            "chapter_concepts": chapter.get("concepts", []) if chapter else [],
        },
        indent=2,
    )


def register_expert_tools(db: Database) -> None:
    """Dynamically register ask_<expert> tools for all known experts."""
    experts = db.list_experts()
    for expert in experts:
        slug = expert["slug"]
        tool_name = f"ask_{slug}"

        @mcp.tool(name=tool_name)
        def ask_tool(question: str, n_results: int = 3, _slug=slug) -> str:
            f"""Query the {expert['expert_name']} expert.

            Args:
                question: Your question about {expert.get('capabilities', ['ML'])[0]}
                n_results: Number of relevant content chunks to return (default: 3)

            Returns:
                JSON with expert info and relevant content
            """
            return _ask_expert(_slug, question, n_results)


@mcp.tool(name="get-extraction-status")
def get_extraction_status() -> str:
    """Check the progress of PDF extraction.

    Returns:
        JSON list of extraction status per chapter
    """
    db = _get_db()
    statuses = db.get_extraction_status()
    return json.dumps(statuses, indent=2)


@mcp.tool(name="get-stats")
def get_stats() -> str:
    """Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with database stats, embedding stats, and expert counts
    """
    db = _get_db()
    embeddings = _get_embeddings()

    db_stats = db.get_stats()
    try:
        emb_stats = embeddings.get_stats()
    except Exception as e:
        emb_stats = {"error": str(e)}

    return json.dumps(
        {
            "database": db_stats,
            "embeddings": emb_stats,
        },
        indent=2,
    )


@mcp.tool(name="index_ml_chapters")
def index_ml_chapters(force_reindex: bool = False) -> str:
    """Index all PDF chapters (extract + embed + store).

    Args:
        force_reindex: Re-extract and re-embed even if already done (default: False)

    Returns:
        JSON status of indexing operation
    """
    return extract_knowledge(force_reindex=force_reindex)


@mcp.tool(name="search_ml_principles")
def search_ml_principles(query: str, n_results: int = 5) -> str:
    """Semantic search over indexed ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results (default: 5)

    Returns:
        JSON list of matching content
    """
    return search_concepts(query, n_results)


@mcp.tool(name="get_rdagent_context")
def get_rdagent_context(
    competition_name: str,
    description: str = "",
    n_results: int = 5,
) -> str:
    """Generate context prompt for rdagent from ML Principles.

    Args:
        competition_name: Name of the competition
        description: Description of the competition
        n_results: Number of context results (default: 5)

    Returns:
        JSON with context for rdagent
    """
    return run_rdagent(competition=competition_name, description=description, n_context_results=n_results)


@mcp.tool(name="list_indexed_chapters")
def list_indexed_chapters() -> str:
    """List all indexed chapters from ML Principles.

    Returns:
        JSON list of indexed chapters with metadata
    """
    db = _get_db()
    chapters = db.list_chapters()
    summary = []
    for ch in chapters:
        summary.append({
            "chapter_num": ch["chapter_num"],
            "title": ch["title"],
            "concepts_count": len(ch.get("concepts", [])),
            "page_count": ch.get("page_count", 0),
            "extracted_at": ch.get("extracted_at"),
        })
    return json.dumps(summary, indent=2)


@mcp.tool(name="get_indexing_stats")
def get_indexing_stats() -> str:
    """Get indexing statistics for ML Principles chapters.

    Returns:
        JSON with indexing statistics
    """
    return get_stats()


def main():
    """Run the MLSysEng MoE MCP server."""
    try:
        db = _get_db()
        register_expert_tools(db)
    except Exception as e:
        logger.warning("Could not register expert tools at startup: %s", e)

    mcp.run()


if __name__ == "__main__":
    main()
