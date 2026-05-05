"""FastMCP server for the MLSysEng MoE system.

Provides MCP tools for knowledge extraction, expert management,
RAG search, competition entry building, and convergence loop execution.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import database
from . import docling_worker
from . import embeddings
from . import expert_registry
from .loop_controller import ConvergenceLoop, LoopConfig, run_convergence_loop

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    instructions="ML Systems Engineering Mixture of Experts - Knowledge extraction, expert routing, and competition entry building",
)

DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
)
CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/chroma_db"),
)


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.
    """
    database.init_db(DB_PATH)

    extraction_results = docling_worker.run_extraction(
        force_reindex=force_reindex,
        db_path=DB_PATH,
    )

    if not extraction_results:
        return json.dumps({
            "status": "no_chapters_found",
            "message": "No chapter folders found. Check ML_PRINCIPLES_PATH environment variable.",
        })

    indexing_result = embeddings.index_all_chapters(
        db_path=DB_PATH, persist_path=CHROMA_PATH
    )

    experts = expert_registry.register_all_experts(db_path=DB_PATH)

    return json.dumps({
        "status": "completed",
        "chapters_extracted": len(extraction_results),
        "chunks_indexed": indexing_result.get("total_chunks", 0),
        "experts_created": len(experts),
        "details": extraction_results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Cycles through relevant experts, applying their strategies iteratively
    until the state converges (||state[n] - state[n-1]||_2 < epsilon).
    """
    database.init_db(DB_PATH)

    relevant_experts = embeddings.infer_relevant_experts(
        competition, db_path=DB_PATH, persist_path=CHROMA_PATH
    )

    if not relevant_experts:
        all_experts = database.get_all_experts(DB_PATH)
        if all_experts:
            relevant_experts = all_experts[:3]
        else:
            return json.dumps({
                "status": "no_experts",
                "message": "No experts available. Run extract_knowledge first.",
            })

    config = LoopConfig(
        objective="minimize_validation_loss",
        epsilon=0.001,
        max_iterations=10,
        patience=3,
    )

    summary = run_convergence_loop(
        experts=relevant_experts,
        competition=competition,
        config=config,
    )

    return json.dumps({
        "status": "converged" if summary["converged"] else "max_iterations_reached",
        "competition": competition,
        "experts_used": [e.get("expert_name", "") for e in relevant_experts],
        "summary": summary,
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Uses sentence-transformers embeddings and ChromaDB for fast similarity search.
    """
    results = embeddings.search(query, n_results=n_results, persist_path=CHROMA_PATH)

    if not results:
        return json.dumps({
            "status": "no_results",
            "message": "No matching concepts found. Have you run extract_knowledge?",
        })

    formatted = []
    for r in results:
        formatted.append({
            "chapter": r["metadata"].get("title", "Unknown"),
            "chapter_number": r["metadata"].get("chapter_number"),
            "content": r["document"][:300],
            "relevance": 1.0 / (1.0 + (r.get("distance") or 1.0)),
        })

    return json.dumps({"results": formatted, "query": query}, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and strategies."""
    database.init_db(DB_PATH)
    experts = database.get_all_experts(DB_PATH)

    if not experts:
        return json.dumps({
            "status": "no_experts",
            "message": "No experts registered. Run extract_knowledge first.",
        })

    summary = []
    for expert in experts:
        summary.append({
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert["capabilities"],
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "skills_count": len(expert.get("skills", [])),
        })

    return json.dumps({"experts": summary, "total": len(summary)}, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG-informed skill selection.

    Identifies relevant experts, retrieves ML principles context,
    and generates a structured competition approach.
    """
    database.init_db(DB_PATH)

    relevant_experts = embeddings.infer_relevant_experts(
        competition, db_path=DB_PATH, persist_path=CHROMA_PATH
    )

    context = embeddings.get_rdagent_context(
        competition_name=competition,
        db_path=DB_PATH,
        persist_path=CHROMA_PATH,
    )

    if not relevant_experts:
        all_experts = database.get_all_experts(DB_PATH)
        relevant_experts = all_experts[:3] if all_experts else []

    entry = {
        "competition": competition,
        "experts_selected": [
            {
                "name": e.get("expert_name", ""),
                "strategy": e.get("strategy", ""),
                "skills": e.get("skills", []),
                "relevance_score": e.get("relevance_score", 0),
            }
            for e in relevant_experts
        ],
        "ml_principles_context": context[:2000],
        "recommended_approach": _build_approach(relevant_experts, competition),
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context for a competition.

    Generates context from the knowledge base and prepares an rdagent command.
    """
    context = embeddings.get_rdagent_context(
        competition_name=competition,
        description=description,
        db_path=DB_PATH,
        persist_path=CHROMA_PATH,
    )

    command = f"rdagent kaggle --competition {competition}"

    return json.dumps({
        "command": command,
        "context": context,
        "competition": competition,
        "instructions": (
            "Use the context above to guide your rdagent session. "
            "Apply the recommended strategies from ML Principles."
        ),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the current extraction progress and status."""
    database.init_db(DB_PATH)
    stats = database.get_stats(DB_PATH)
    chapters = database.get_all_chapters(DB_PATH)

    return json.dumps({
        "stats": stats,
        "chapters": [
            {
                "number": ch["chapter_number"],
                "title": ch["title"],
                "extracted_at": ch.get("extracted_at", ""),
            }
            for ch in chapters
        ],
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapter, concept, and expert counts."""
    database.init_db(DB_PATH)
    stats = database.get_stats(DB_PATH)
    return json.dumps(stats, indent=2)


def _build_approach(experts: list[dict], competition: str) -> dict:
    """Build a recommended approach from the selected experts."""
    if not experts:
        return {
            "steps": [
                "Run extract_knowledge to index ML Principles",
                "Use search_concepts to find relevant techniques",
                f"Build baseline for {competition}",
            ]
        }

    steps = []
    all_skills = set()

    for i, expert in enumerate(experts, 1):
        strategy = expert.get("strategy", "Baseline → Submit")
        steps.append(f"Step {i} ({expert.get('expert_name', 'Expert')}): {strategy}")
        for skill in expert.get("skills", []):
            all_skills.add(skill)

    formula = experts[0].get("formula", {}) if experts else {}

    return {
        "steps": steps,
        "skills_needed": list(all_skills),
        "objective": formula.get("objective", "minimize_validation_loss"),
        "convergence": {
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
        },
    }


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    database.init_db(DB_PATH)
    mcp.run()


if __name__ == "__main__":
    main()
