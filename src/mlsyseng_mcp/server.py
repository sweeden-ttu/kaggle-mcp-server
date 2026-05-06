"""FastMCP server for MLSysEng MoE system."""

import json
import logging
from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from .database import (
    init_db,
    get_stats as db_get_stats,
    get_extraction_status,
    get_chapters,
    get_chapter_content,
    store_chapter,
    store_concepts,
    update_extraction_status,
    store_competition_entry,
)
from .docling_worker import discover_chapters, extract_chapter
from .embeddings import EmbeddingStore
from .expert_registry import (
    create_expert_from_chapter,
    list_experts,
    get_expert,
    get_experts_for_competition,
)
from .loop_controller import ConvergenceLoop, LoopConfig, run_convergence_loop

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description="ML Systems Engineering Mixture of Experts - Knowledge extraction, expert registry, and RAG-informed competition building",
)

_embedding_store: Optional[EmbeddingStore] = None


def get_embedding_store() -> EmbeddingStore:
    global _embedding_store
    if _embedding_store is None:
        _embedding_store = EmbeddingStore()
    return _embedding_store


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans all chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If True, re-extract already indexed chapters.
    """
    init_db()
    chapters = discover_chapters()

    if not chapters:
        return json.dumps({
            "status": "no_chapters_found",
            "message": "No chapter folders found in ML Principles path. Set ML_PRINCIPLES_PATH environment variable.",
        })

    results = []
    embed_store = get_embedding_store()

    for chapter_info in chapters:
        chapter_num = chapter_info["chapter_number"]

        if not force_reindex:
            existing = get_chapter_content(chapter_num)
            if existing:
                results.append({"chapter": chapter_num, "status": "already_indexed"})
                continue

        update_extraction_status(chapter_num, "started")
        try:
            extracted = extract_chapter(chapter_info)

            chapter_id = store_chapter(
                chapter_number=extracted["chapter_number"],
                title=extracted["title"],
                source_path=extracted["source_path"],
                markdown_content=extracted["markdown_content"],
            )

            store_concepts(chapter_id, extracted["concepts"])

            num_indexed = embed_store.index_chapter(
                chapter_number=extracted["chapter_number"],
                title=extracted["title"],
                content=extracted["markdown_content"],
                concepts=extracted["concepts"],
            )

            create_expert_from_chapter(
                chapter_number=extracted["chapter_number"],
                title=extracted["title"],
                concepts=extracted["concepts"],
            )

            update_extraction_status(chapter_num, "completed")
            results.append({
                "chapter": chapter_num,
                "title": extracted["title"],
                "status": "indexed",
                "concepts": len(extracted["concepts"]),
                "embeddings": num_indexed,
            })

        except Exception as e:
            update_extraction_status(chapter_num, "failed", str(e))
            results.append({"chapter": chapter_num, "status": "failed", "error": str(e)})
            logger.error(f"Failed to extract chapter {chapter_num}: {e}")

    return json.dumps({
        "status": "complete",
        "chapters_processed": len(results),
        "results": results,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Selects relevant experts based on competition context, then iterates
    until state convergence: ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition name or description.
    """
    init_db()
    embed_store = get_embedding_store()

    relevant_chapters = embed_store.get_relevant_experts(competition)
    experts = get_experts_for_competition(competition, relevant_chapters)

    if not experts:
        all_experts = list_experts()
        experts = all_experts[:3] if all_experts else []

    if not experts:
        return json.dumps({
            "status": "no_experts",
            "message": "No experts registered. Run extract-knowledge first.",
        })

    config = LoopConfig(
        objective="minimize_validation_loss",
        epsilon=0.001,
        max_iterations=10,
        patience=3,
    )

    if experts and experts[0].get("loop_config"):
        lc = experts[0]["loop_config"]
        if isinstance(lc, dict):
            config = LoopConfig.from_dict(lc)

    report = run_convergence_loop(experts=experts, config=config)

    skills_used = []
    for expert in experts:
        skills_used.extend(expert.get("skills", []))

    store_competition_entry(
        competition_name=competition,
        experts_used=report.get("experts_used", []),
        skills_applied=list(set(skills_used)),
        state_history=report.get("state_history", []),
        converged=report.get("converged", False),
    )

    return json.dumps({
        "competition": competition,
        "converged": report["converged"],
        "total_iterations": report["total_iterations"],
        "final_delta": report["final_delta"],
        "epsilon": report["epsilon"],
        "experts_used": report["experts_used"],
        "skills_applied": list(set(skills_used)),
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return.
    """
    embed_store = get_embedding_store()
    results = embed_store.search(query, n_results=n_results)

    return json.dumps({
        "query": query,
        "results": results,
        "count": len(results),
    }, indent=2)


@mcp.tool()
def list_all_experts() -> str:
    """List all registered chapter experts with their capabilities and skills."""
    init_db()
    experts = list_experts()
    return json.dumps({
        "experts": experts,
        "count": len(experts),
    }, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge and RAG-informed skill selection.

    Args:
        competition: Competition name or slug.
    """
    init_db()
    embed_store = get_embedding_store()

    relevant_chapters = embed_store.get_relevant_experts(competition)
    experts = get_experts_for_competition(competition, relevant_chapters)

    if not experts:
        all_experts = list_experts()
        experts = all_experts[:3] if all_experts else []

    entry = {
        "competition": competition,
        "selected_experts": [],
        "recommended_skills": [],
        "strategy": "Baseline → EDA → Feature Engineering → Model Selection → Submit",
        "notebook_plan": [],
    }

    all_skills = set()
    for expert in experts:
        entry["selected_experts"].append({
            "name": expert.get("expert_name"),
            "slug": expert.get("slug"),
            "capabilities": expert.get("capabilities", []),
            "relevance": expert.get("relevance_score", 0.0),
        })
        for skill in expert.get("skills", []):
            all_skills.add(skill)

        entry["notebook_plan"].append({
            "expert": expert.get("slug"),
            "notebook": f"Expert_{expert.get('slug', 'unknown')}.ipynb",
            "focus": expert.get("capabilities", [])[:2],
        })

    entry["recommended_skills"] = sorted(all_skills)

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Generate rdagent context with ML Principles knowledge for a competition.

    Args:
        competition: Competition name.
        description: Competition description for context matching.
    """
    init_db()
    embed_store = get_embedding_store()

    query = f"{competition} {description}".strip()
    search_results = embed_store.search(query, n_results=8)

    context_sections = []
    for r in search_results:
        meta = r["metadata"]
        context_sections.append(
            f"[Chapter {meta.get('chapter_number', '?')}: {meta.get('title', 'Unknown')}]\n{r['content']}"
        )

    context_prompt = "\n\n---\n\n".join(context_sections)

    return json.dumps({
        "competition": competition,
        "context_prompt": context_prompt,
        "sources": len(search_results),
        "command": f"rdagent --competition {competition}",
        "note": "Use the context_prompt to guide rdagent's approach",
    }, indent=2)


@mcp.tool()
def get_extraction_progress() -> str:
    """Check the extraction status of all chapters."""
    init_db()
    status = get_extraction_status()
    chapters = get_chapters()
    return json.dumps({
        "extraction_status": status,
        "chapters_in_db": len(chapters),
    }, indent=2)


@mcp.tool()
def get_system_stats() -> str:
    """Get system statistics including chapters, experts, and embeddings."""
    init_db()
    stats = db_get_stats()

    try:
        embed_store = get_embedding_store()
        embed_stats = embed_store.get_stats()
        stats["embeddings"] = embed_stats
    except Exception as e:
        stats["embeddings"] = {"error": str(e)}

    return json.dumps(stats, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert with a question.

    Args:
        expert_slug: The slug of the expert to query (e.g., "08_ml_systems").
        question: The question to ask the expert.
    """
    init_db()
    expert = get_expert(expert_slug)

    if not expert:
        available = list_experts()
        slugs = [e.get("slug") for e in available]
        return json.dumps({
            "error": f"Expert '{expert_slug}' not found",
            "available_experts": slugs,
        })

    embed_store = get_embedding_store()
    chapter_id = expert.get("chapter_id")
    results = embed_store.search(question, n_results=5, filter_chapter=chapter_id)

    context = "\n".join([r["content"] for r in results])

    return json.dumps({
        "expert": expert["expert_name"],
        "slug": expert["slug"],
        "capabilities": expert.get("capabilities", []),
        "strategy": expert.get("strategy", ""),
        "relevant_context": context[:2000],
        "question": question,
        "note": "Use the relevant_context to formulate an answer based on this expert's knowledge",
    }, indent=2)


def main():
    """Run the MCP server."""
    init_db()
    mcp.run()


if __name__ == "__main__":
    main()
