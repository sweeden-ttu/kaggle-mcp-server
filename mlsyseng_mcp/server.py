"""FastMCP server for MLSysEng MoE system."""

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import (
    get_all_experts,
    get_extraction_status,
    get_stats,
    init_db,
)
from mlsyseng_mcp.docling_worker import run_extraction
from mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_mcp.expert_registry import (
    get_expert,
    list_experts,
    query_expert,
    register_all_experts,
)
from mlsyseng_mcp.loop_controller import (
    LoopController,
    build_competition_entry,
)

mcp = FastMCP(
    "mlsyseng-moe",
    instructions="Machine Learning Systems Expert Mixture of Experts - extracts knowledge from ML Principles PDFs, registers chapter experts, and builds Kaggle competition entries using RAG-informed skill selection with state convergence loops.",
)

_embedding_engine: Optional[EmbeddingEngine] = None
_loop_controllers: dict[str, LoopController] = {}


def _get_embedding_engine() -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, create experts.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract all chapters even if already indexed.
    """
    init_db()

    results = run_extraction(force_reindex=force_reindex)

    if results["chapters_processed"] > 0:
        experts = register_all_experts()
        results["experts_created"] = len(experts)

        engine = _get_embedding_engine()
        from mlsyseng_mcp.database import get_all_chapters

        chapters = get_all_chapters()
        chunks_added = 0
        for chapter in chapters:
            if chapter.get("content_md"):
                count = engine.add_chapter_content(
                    chapter_id=chapter["id"],
                    chapter_number=chapter["chapter_number"],
                    title=chapter["title"],
                    content=chapter["content_md"],
                )
                chunks_added += count
        results["chunks_embedded"] = chunks_added

    return json.dumps(results, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the full evolution pipeline: extract, build entry, run convergence loop.

    Args:
        competition: Name of the Kaggle competition to target.
    """
    init_db()
    engine = _get_embedding_engine()

    experts = get_all_experts()
    if not experts:
        extraction_results = run_extraction()
        experts_created = register_all_experts()
        experts = get_all_experts()

    entry = build_competition_entry(
        competition=competition,
        experts=experts,
        embedding_engine=engine,
    )

    loop_config = entry.get("loop_config", {})
    controller = LoopController.from_config(loop_config)
    _loop_controllers[competition] = controller

    import numpy as np

    state_dim = max(len(experts), 5)
    state = np.random.randn(state_dim).tolist()

    iteration_results = []
    for i in range(controller.max_iterations):
        noise = np.random.randn(state_dim) * (0.1 / (i + 1))
        state = (np.array(state) + noise).tolist()
        score = 1.0 / (1.0 + np.sum(np.array(state) ** 2))

        step_result = controller.step(
            state_vector=state,
            score=score,
            metrics={"iteration_score": score},
            expert_contributions=[e["expert_name"] for e in experts[:3]],
        )
        iteration_results.append(step_result)

        if step_result["should_stop"]:
            break

    return json.dumps({
        "competition": competition,
        "entry": {
            "experts_used": entry["experts_used"],
            "capabilities_count": len(entry["capabilities"]),
            "skills_count": len(entry["skills"]),
        },
        "convergence": controller.get_status(),
        "final_iteration": iteration_results[-1] if iteration_results else None,
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query about ML concepts.
        n_results: Number of results to return (default 5).
    """
    try:
        engine = _get_embedding_engine()
        results = engine.search(query, n_results=n_results)
    except ImportError as e:
        return json.dumps({
            "query": query,
            "results": [],
            "error": f"Embedding engine unavailable: {e}. Install sentence-transformers and chromadb.",
        }, indent=2)

    formatted = []
    for r in results:
        formatted.append({
            "content": r["content"][:300],
            "chapter": r["metadata"].get("title", "Unknown"),
            "chapter_number": r["metadata"].get("chapter_number"),
            "relevance": round(r["relevance"], 4),
        })

    return json.dumps({"query": query, "results": formatted}, indent=2)


@mcp.tool()
def list_all_experts() -> str:
    """List all registered chapter experts with their capabilities and strategies."""
    init_db()
    experts = list_experts()
    summary = []
    for e in experts:
        summary.append({
            "name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e["capabilities"][:3],
            "strategy": e["strategy"][:100],
            "formula_objective": e["formula"].get("objective", ""),
        })
    return json.dumps({"experts": summary, "total": len(summary)}, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG skill selection.

    Args:
        competition: Name of the Kaggle competition.
    """
    init_db()
    engine = _get_embedding_engine()
    experts = get_all_experts()

    if not experts:
        return json.dumps({"error": "No experts registered. Run extract_knowledge first."})

    entry = build_competition_entry(
        competition=competition,
        experts=experts,
        embedding_engine=engine,
    )

    return json.dumps(entry, indent=2, default=str)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Prepare rdagent context with ML Principles knowledge for a competition.

    Args:
        competition: Kaggle competition name.
        description: Brief description of the competition.
    """
    query = f"{competition} {description}"
    relevant = []
    try:
        engine = _get_embedding_engine()
        relevant = engine.search(query, n_results=5)
    except ImportError:
        pass

    context_parts = []
    for r in relevant:
        context_parts.append(
            f"[Ch.{r['metadata'].get('chapter_number', '?')} - {r['metadata'].get('title', 'Unknown')}]\n{r['content'][:200]}"
        )

    experts = get_all_experts()
    expert_context = []
    for e in experts[:3]:
        expert_context.append(
            f"Expert: {e['expert_name']}\n"
            f"Strategy: {e['strategy']}\n"
            f"Capabilities: {', '.join(e['capabilities'][:3])}"
        )

    rdagent_prompt = f"""Competition: {competition}
Description: {description}

## Relevant ML Principles Knowledge
{chr(10).join(context_parts)}

## Expert Recommendations
{chr(10).join(expert_context)}

## Suggested Approach
1. Start with baseline model using expert strategy
2. Apply feature engineering from relevant chapters
3. Use convergence loop to iterate
4. Submit best result
"""

    return json.dumps({
        "competition": competition,
        "rdagent_context": rdagent_prompt,
        "relevant_chapters": [r["metadata"].get("chapter_number") for r in relevant],
        "command": f"rdagent --competition {competition} --context ml_principles",
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The slug identifier of the expert (e.g., '08_ml_systems').
        question: The question to ask the expert.
    """
    init_db()
    result = query_expert(expert_slug, question)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_extraction_progress() -> str:
    """Check extraction progress and system status."""
    init_db()
    status = get_extraction_status()
    return json.dumps(status, indent=2, default=str)


@mcp.tool()
def get_system_stats() -> str:
    """Get system statistics including chapters, concepts, experts, and entries."""
    init_db()
    stats = get_stats()

    try:
        engine = _get_embedding_engine()
        embedding_stats = engine.get_collection_stats()
        stats["embeddings"] = embedding_stats
    except Exception:
        stats["embeddings"] = {"status": "not initialized"}

    return json.dumps(stats, indent=2)


def main():
    """Run the MCP server."""
    init_db()
    mcp.run()


if __name__ == "__main__":
    main()
