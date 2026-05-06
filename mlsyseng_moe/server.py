"""FastMCP server for MLSysEng MoE system."""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_moe import database, docling_worker, embeddings, expert_registry, loop_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description="Machine Learning Systems Expert Mixture of Experts - extracts knowledge from ML Principles PDFs, registers chapter experts, and builds Kaggle competition entries using RAG-informed skill selection with state convergence loops.",
)


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.
    """
    results = docling_worker.index_all_chapters(force=force_reindex)
    experts = expert_registry.register_experts_from_db()

    summary = {
        "extraction_results": results,
        "experts_created": len(experts),
        "expert_names": [e["expert_name"] for e in experts],
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def evolve(competition: str, epsilon: float = 0.001, max_iterations: int = 10, patience: int = 3) -> str:
    """
    Run the state convergence loop for a competition.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    Runs until convergence or max_iterations reached.
    """
    experts = database.get_all_experts()
    matched = embeddings.infer_skills_for_competition(competition, experts)

    all_skills = []
    for match in matched:
        all_skills.extend(match["expert"].get("skills", []))
    all_skills = list(set(all_skills))

    result = loop_controller.run_competition_loop(
        competition_name=competition,
        expert_skills=all_skills,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    output = {
        "competition": competition,
        "experts_used": [m["expert"]["expert_name"] for m in matched],
        "skills_applied": all_skills,
        "convergence": result,
    }
    return json.dumps(output, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base."""
    results = embeddings.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills."""
    experts = database.get_all_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str) -> str:
    """
    Build a competition entry using expert knowledge.

    Uses RAG to find relevant experts, then assembles a strategy
    with recommended skills and approach.
    """
    experts = database.get_all_experts()
    matched = embeddings.infer_skills_for_competition(competition, experts)

    entry = {
        "competition": competition,
        "experts_consulted": [],
        "recommended_skills": [],
        "strategy": "",
        "formula": {},
    }

    strategies = []
    all_skills = set()
    for match in matched:
        expert = match["expert"]
        entry["experts_consulted"].append({
            "name": expert["expert_name"],
            "relevance": match["relevance"],
            "capabilities": expert.get("capabilities", []),
        })
        all_skills.update(expert.get("skills", []))
        if expert.get("strategy"):
            strategies.append(expert["strategy"])
        if expert.get("formula"):
            entry["formula"] = expert["formula"]

    entry["recommended_skills"] = sorted(all_skills)
    entry["strategy"] = strategies[0] if strategies else expert_registry.DEFAULT_STRATEGY

    relevant_context = embeddings.search(competition, n_results=3)
    entry["relevant_knowledge"] = [
        {"chapter": r["chapter_title"], "excerpt": r["content"][:200]}
        for r in relevant_context
    ]

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """
    Prepare rdagent context with ML Principles knowledge.

    Generates context prompt for rdagent data_science competitions.
    """
    relevant = embeddings.search(f"{competition} {description}", n_results=10)
    experts = database.get_all_experts()
    matched = embeddings.infer_skills_for_competition(f"{competition} {description}", experts)

    context_parts = [
        f"# ML Principles Context for: {competition}",
        "",
        "## Relevant Knowledge",
    ]

    for r in relevant:
        context_parts.append(f"\n### From: {r['chapter_title']}")
        context_parts.append(r["content"][:500])

    context_parts.append("\n## Expert Recommendations")
    for match in matched:
        expert = match["expert"]
        context_parts.append(f"\n### {expert['expert_name']}")
        context_parts.append(f"Strategy: {expert.get('strategy', 'N/A')}")
        context_parts.append(f"Skills: {', '.join(expert.get('skills', []))}")

    context = "\n".join(context_parts)

    return json.dumps({
        "competition": competition,
        "context_prompt": context,
        "experts_matched": len(matched),
        "knowledge_chunks": len(relevant),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the extraction progress for all chapters."""
    chapters = database.get_all_chapters()
    stats = database.get_stats()
    vector_stats = {}
    try:
        vector_stats = embeddings.get_collection_stats()
    except Exception:
        vector_stats = {"error": "ChromaDB not initialized"}

    return json.dumps({
        "chapters": chapters,
        "stats": stats,
        "vector_store": vector_stats,
    }, indent=2, default=str)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics."""
    db_stats = database.get_stats()
    try:
        vector_stats = embeddings.get_collection_stats()
    except Exception:
        vector_stats = {"total_chunks": 0}

    return json.dumps({
        "database": db_stats,
        "vector_store": vector_stats,
        "config": {
            "ml_principles_path": os.environ.get("ML_PRINCIPLES_PATH", docling_worker.DEFAULT_ML_PRINCIPLES_PATH),
            "db_path": database.DEFAULT_DB_PATH,
            "chroma_path": embeddings.DEFAULT_CHROMA_PATH,
            "embedding_model": embeddings.MODEL_NAME,
        },
    }, indent=2)


def _register_expert_tools():
    """Dynamically register ask_<expert> tools for each registered expert."""
    try:
        experts = database.get_all_experts()
    except Exception:
        return

    for expert in experts:
        slug = expert["slug"]
        name = expert["expert_name"]

        def make_ask_fn(expert_slug: str, expert_name: str):
            def ask_expert(question: str) -> str:
                f"""Query the {expert_name} expert."""
                context = expert_registry.get_expert_context(expert_slug)
                if not context:
                    return json.dumps({"error": f"Expert {expert_slug} not found"})

                relevant = embeddings.search(question, n_results=3)
                return json.dumps({
                    "expert": context["expert"],
                    "concepts": context["concepts"],
                    "relevant_passages": relevant,
                }, indent=2)

            ask_expert.__name__ = f"ask_{expert_slug}"
            ask_expert.__doc__ = f"Query the {expert_name} expert with a question."
            return ask_expert

        fn = make_ask_fn(slug, name)
        mcp.tool()(fn)


database.init_db()

try:
    _register_expert_tools()
except Exception as e:
    logger.warning(f"Could not register expert tools: {e}")


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
