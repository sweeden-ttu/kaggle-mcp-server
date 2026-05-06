"""FastMCP server for the MLSysEng MoE system.

Exposes MCP tools for knowledge extraction, expert queries,
competition entry building, and convergence loops.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import database as db
from . import docling_worker
from . import embeddings
from . import expert_registry
from . import loop_controller

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description=(
        "Machine Learning Systems Expert Mixture of Experts — "
        "extracts knowledge from ML Principles PDFs, registers chapter experts, "
        "and builds Kaggle competition entries using RAG-informed skill selection "
        "with state convergence loops."
    ),
)

DB_PATH = os.environ.get("SQLITE_DB_PATH")
CHROMA_PATH = os.environ.get("CHROMA_DB_PATH")


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.
    """
    db.init_db(DB_PATH)

    extraction_results = docling_worker.extract_all(
        db_path=DB_PATH, force=force_reindex
    )

    expert_results = expert_registry.register_experts_from_chapters(db_path=DB_PATH)

    try:
        index_results = embeddings.index_chapters(
            db_path=DB_PATH, chroma_path=CHROMA_PATH, force=force_reindex
        )
    except ImportError as e:
        index_results = {
            "status": "skipped",
            "reason": f"Missing dependency: {e}",
        }

    return json.dumps({
        "extraction": extraction_results,
        "experts": expert_results,
        "indexing": index_results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition.

    Iteratively refines expert weights until state converges:
    ||state[n] - state[n-1]||_2 < epsilon
    """
    db.init_db(DB_PATH)
    result = loop_controller.run_convergence_loop(
        competition=competition,
        db_path=DB_PATH,
        chroma_path=CHROMA_PATH,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base."""
    try:
        results = embeddings.search(
            query, n_results=n_results, chroma_path=CHROMA_PATH
        )
    except Exception as e:
        return json.dumps({"error": str(e), "hint": "Run extract-knowledge first"})

    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def list_experts_tool() -> str:
    """List all registered chapter experts."""
    db.init_db(DB_PATH)
    experts = db.list_experts(DB_PATH)
    summary = []
    for e in experts:
        summary.append({
            "expert_name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e["capabilities"],
            "skills_count": len(e.get("skills", [])),
            "strategy": e["strategy"],
            "formula": e["formula"],
        })
    return json.dumps(summary, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Selects the most relevant experts, gathers their recommended
    skills and strategies, and produces a structured entry plan.
    """
    db.init_db(DB_PATH)
    result = loop_controller.build_competition_entry(
        competition=competition,
        db_path=DB_PATH,
        chroma_path=CHROMA_PATH,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt from relevant ML Principles chapters
    for use with rdagent data_science competitions.
    """
    try:
        context = embeddings.generate_context(
            competition, description=description, chroma_path=CHROMA_PATH
        )
    except Exception:
        context = f"No indexed content available for {competition}. Run extract-knowledge first."

    experts = db.list_experts(DB_PATH)
    expert_info = []
    for e in experts[:5]:
        expert_info.append({
            "name": e["expert_name"],
            "strategy": e["strategy"],
            "formula": e["formula"],
        })

    return json.dumps({
        "competition": competition,
        "context": context,
        "experts": expert_info,
        "rdagent_command": (
            f"rdagent data_science --competition {competition} "
            f"--context 'ML Principles guided approach'"
        ),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for PDF processing jobs."""
    db.init_db(DB_PATH)
    status = db.get_extraction_status(DB_PATH)
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics — chapters, experts, word counts."""
    db.init_db(DB_PATH)
    stats = db.get_stats(DB_PATH)

    try:
        from chromadb import PersistentClient
        path = CHROMA_PATH or embeddings.DEFAULT_CHROMA_PATH
        if os.path.exists(path):
            client = PersistentClient(path=path)
            try:
                coll = client.get_collection(embeddings.COLLECTION_NAME)
                stats["vector_chunks"] = coll.count()
            except Exception:
                stats["vector_chunks"] = 0
        else:
            stats["vector_chunks"] = 0
    except ImportError:
        stats["vector_chunks"] = "chromadb not installed"

    return json.dumps(stats, indent=2)


@mcp.tool()
def ask_expert(slug: str, question: str) -> str:
    """Query a specific chapter expert by slug.

    Example slugs: 08_ml_systems, 03_deep_learning
    """
    db.init_db(DB_PATH)
    result = expert_registry.query_expert(
        slug=slug,
        question=question,
        db_path=DB_PATH,
        chroma_path=CHROMA_PATH,
    )
    return json.dumps(result, indent=2, default=str)


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    db.init_db(DB_PATH)
    mcp.run()


if __name__ == "__main__":
    main()
