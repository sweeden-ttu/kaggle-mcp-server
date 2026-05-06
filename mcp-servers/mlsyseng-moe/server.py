"""FastMCP server for MLSysEng Mixture of Experts system.

Provides MCP tools for knowledge extraction, expert management,
RAG search, and competition entry building.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

try:
    from . import database as db
    from . import docling_worker
    from . import embeddings
    from . import expert_registry
    from . import loop_controller
except ImportError:
    import database as db
    import docling_worker
    import embeddings
    import expert_registry
    import loop_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

DB_PATH = os.environ.get("SQLITE_DB_PATH")
CHROMA_PATH = os.environ.get("CHROMA_DB_PATH")


def _init():
    db.init_db(DB_PATH)


_init()


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract already completed chapters
    """
    results = docling_worker.extract_all(
        force_reindex=force_reindex, db_path=DB_PATH
    )

    index_result = embeddings.index_all_chapters(
        db_path=DB_PATH, chroma_path=CHROMA_PATH
    )

    experts = expert_registry.register_all_experts(db_path=DB_PATH)

    return json.dumps(
        {
            "extraction": {
                "chapters_processed": len(results),
                "completed": sum(1 for r in results if r["status"] == "completed"),
                "skipped": sum(1 for r in results if r["status"] == "skipped"),
                "failed": sum(1 for r in results if r["status"] == "failed"),
            },
            "indexing": index_result,
            "experts_registered": len(experts),
        },
        indent=2,
    )


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Extracts knowledge if needed, selects relevant experts via RAG,
    and runs the state convergence loop until ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Kaggle competition slug (e.g., "titanic")
    """
    experts = db.get_all_experts(DB_PATH)

    if not experts:
        extract_knowledge(force_reindex=False)
        experts = db.get_all_experts(DB_PATH)

    if not experts:
        return json.dumps({"error": "No experts available after extraction"})

    inferred = embeddings.infer_skills_for_competition(
        competition, chroma_path=CHROMA_PATH, db_path=DB_PATH
    )

    if inferred:
        selected = [item["expert"] for item in inferred[:3]]
    else:
        selected = experts[:3]

    results = loop_controller.evolve_all_experts(competition, selected, db_path=DB_PATH)

    entry = expert_registry.build_competition_entry(
        competition,
        expert_slugs=[e["slug"] for e in selected],
        db_path=DB_PATH,
    )

    return json.dumps(
        {
            "competition": competition,
            "convergence_results": results,
            "entry": entry,
        },
        indent=2,
    )


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return
    """
    hits = embeddings.search(query, n_results=n_results, chroma_path=CHROMA_PATH)
    return json.dumps(hits, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their skills and strategies."""
    experts = db.get_all_experts(DB_PATH)
    summaries = []
    for e in experts:
        summaries.append({
            "expert_name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e.get("capabilities", []),
            "skills": e.get("skills", []),
            "strategy": e.get("strategy", ""),
            "formula": e.get("formula", {}),
        })
    return json.dumps(summaries, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Selects relevant experts via RAG and creates notebook outlines.

    Args:
        competition: Kaggle competition slug
    """
    inferred = embeddings.infer_skills_for_competition(
        competition, chroma_path=CHROMA_PATH, db_path=DB_PATH
    )

    if inferred:
        expert_slugs = [item["expert"]["slug"] for item in inferred[:3]]
    else:
        expert_slugs = None

    entry = expert_registry.build_competition_entry(
        competition, expert_slugs=expert_slugs, db_path=DB_PATH
    )
    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt from the knowledge base and prepares
    the rdagent command for a Kaggle competition.

    Args:
        competition_name: Competition name or slug
        description: Optional description of the competition
    """
    query = f"{competition_name} {description}".strip()
    hits = embeddings.search(query, n_results=5, chroma_path=CHROMA_PATH)

    context_parts = []
    for hit in hits:
        context_parts.append(
            f"[{hit['chapter_name']}] (similarity={hit['similarity']:.3f}): "
            f"{hit['text'][:200]}"
        )

    experts = db.get_all_experts(DB_PATH)
    expert_skills = []
    for e in experts[:5]:
        expert_skills.append(
            f"  - {e['expert_name']}: {e.get('strategy', 'N/A')}"
        )

    context = (
        f"ML Principles Context for '{competition_name}':\n\n"
        f"Relevant Knowledge:\n" + "\n".join(context_parts) + "\n\n"
        f"Available Expert Strategies:\n" + "\n".join(expert_skills) + "\n\n"
        f"Recommended approach: Use the most relevant expert strategies above "
        f"to guide feature engineering, model selection, and submission."
    )

    return json.dumps(
        {
            "competition": competition_name,
            "context_prompt": context,
            "rdagent_command": (
                f'rdagent --competition "{competition_name}" '
                f"--mode data_science --context-file /tmp/ml_context.txt"
            ),
            "relevant_chapters": [h["chapter_name"] for h in hits],
            "expert_count": len(experts),
        },
        indent=2,
    )


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for all chapters."""
    chapters = db.get_all_chapters(DB_PATH)
    status_counts = {"pending": 0, "completed": 0, "failed": 0}
    chapter_details = []

    for ch in chapters:
        status = ch.get("extraction_status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
        concepts = ch.get("concepts", [])
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        chapter_details.append({
            "name": ch["chapter_name"],
            "status": status,
            "concept_count": len(concepts) if concepts else 0,
        })

    return json.dumps(
        {
            "summary": status_counts,
            "chapters": chapter_details,
        },
        indent=2,
    )


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapter, expert, and entry counts."""
    stats = db.get_stats(DB_PATH)
    return json.dumps(stats, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The slug identifier of the expert (e.g., "08_ml_systems")
        question: The question to ask the expert
    """
    expert = db.get_expert_by_slug(expert_slug, DB_PATH)
    if not expert:
        available = db.get_all_experts(DB_PATH)
        slugs = [e["slug"] for e in available]
        return json.dumps({
            "error": f"Expert '{expert_slug}' not found",
            "available_experts": slugs,
        })

    hits = embeddings.search(question, n_results=3, chroma_path=CHROMA_PATH)

    relevant_knowledge = []
    for hit in hits:
        if hit.get("chapter_name", "").lower() in expert["expert_name"].lower() or \
           hit.get("similarity", 0) > 0.3:
            relevant_knowledge.append(hit["text"][:300])

    return json.dumps(
        {
            "expert": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "relevant_knowledge": relevant_knowledge,
            "recommendation": (
                f"Based on {expert['expert_name']}'s expertise, "
                f"the recommended approach is: {expert.get('strategy', 'N/A')}. "
                f"Key metrics to optimize: "
                f"{', '.join(expert.get('formula', {}).get('metrics', []))}"
            ),
        },
        indent=2,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
