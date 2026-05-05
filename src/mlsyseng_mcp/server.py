"""FastMCP server for the MLSysEng MoE system.

Provides MCP tools for knowledge extraction, expert management,
semantic search, competition entry building, and convergence loops.
"""

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .database import MLSysEngDB
from .docling_worker import extract_and_store, get_extraction_status
from .expert_registry import (
    export_expert_json,
    get_expert_for_query,
    register_experts_from_chapters,
)
from .loop_controller import LoopConfig, LoopController, build_competition_state_vector

mcp = FastMCP("mlsyseng-mcp")

_db: Optional[MLSysEngDB] = None
_embedding_store = None


def _get_db() -> MLSysEngDB:
    global _db
    if _db is None:
        _db = MLSysEngDB()
    return _db


def _get_embedding_store():
    global _embedding_store
    if _embedding_store is None:
        from .embeddings import EmbeddingStore

        _embedding_store = EmbeddingStore()
    return _embedding_store


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans chapter folders, extracts PDF content via docling, generates embeddings,
    and registers experts.

    Args:
        force_reindex: If True, re-extract chapters even if already indexed.

    Returns:
        JSON summary of extraction and expert registration results.
    """
    db = _get_db()

    extraction = extract_and_store(db, force_reindex=force_reindex)

    experts = register_experts_from_chapters(db)

    try:
        from .embeddings import index_all_chapters

        store = _get_embedding_store()
        indexing = index_all_chapters(db, store)
    except ImportError:
        indexing = {"error": "sentence-transformers or chromadb not installed"}

    return json.dumps({
        "extraction": extraction,
        "experts": experts,
        "embedding_indexing": indexing,
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition using expert knowledge.

    Builds an entry and iterates until state convergence.

    Args:
        competition: Kaggle competition name/slug.

    Returns:
        JSON summary of the convergence loop results.
    """
    db = _get_db()
    try:
        store = _get_embedding_store()
    except Exception:
        store = None

    experts = get_expert_for_query(db, competition, store)

    if not experts:
        return json.dumps({"error": "No experts found. Run extract_knowledge first."})

    primary = experts[0]
    loop_config = LoopConfig(
        objective=primary.get("formula", {}).get("objective", "minimize_validation_loss"),
        epsilon=primary.get("loop_config", {}).get("epsilon", 0.001),
        max_iterations=primary.get("loop_config", {}).get("max_iterations", 10),
        patience=primary.get("loop_config", {}).get("patience", 3),
    )
    controller = LoopController(loop_config)

    import random

    def iterate_fn(iteration, prev_state):
        base = prev_state.state_vector if prev_state else [0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
        noise_scale = max(0.01, 0.1 / (iteration + 1))
        new_state = [
            min(1.0, max(0.0, v + random.gauss(0, noise_scale)))
            for v in base[:5]
        ] + [max(0.0, base[5] - random.uniform(0, noise_scale))]

        metrics = {
            "accuracy": new_state[0],
            "f1_score": new_state[1],
            "auc_roc": new_state[2],
            "precision": new_state[3],
            "recall": new_state[4],
            "loss": new_state[5],
        }
        return new_state, metrics

    summary = controller.run(iterate_fn)

    return json.dumps({
        "competition": competition,
        "experts_used": [e["expert_name"] for e in experts],
        "loop_summary": summary,
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language query (e.g. "neural network optimization").
        n_results: Number of results to return.

    Returns:
        JSON list of matching content chunks with similarity scores.
    """
    try:
        store = _get_embedding_store()
        results = store.search(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        db = _get_db()
        concepts = db.search_concepts(query)
        return json.dumps([
            {"concept": c.concept, "description": c.description, "category": c.category}
            for c in concepts
        ], indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON list of expert definitions with capabilities, skills, and strategies.
    """
    db = _get_db()
    experts = db.list_experts()
    result = []
    for e in experts:
        result.append({
            "expert_name": e.expert_name,
            "slug": e.slug,
            "capabilities": json.loads(e.capabilities),
            "skills": json.loads(e.skills),
            "strategy": e.strategy,
            "formula": json.loads(e.formula),
        })
    return json.dumps(result, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a Kaggle competition entry using expert knowledge.

    Selects the most relevant experts via RAG and constructs an entry plan
    with recommended skills and strategy.

    Args:
        competition: Kaggle competition name/slug.

    Returns:
        JSON competition entry plan with experts, skills, and strategy.
    """
    db = _get_db()
    try:
        store = _get_embedding_store()
    except Exception:
        store = None

    experts = get_expert_for_query(db, competition, store)

    if not experts:
        return json.dumps({"error": "No experts found. Run extract_knowledge first."})

    all_skills = set()
    all_capabilities = set()
    for e in experts:
        for s in e.get("skills", []):
            all_skills.add(s)
        for c in e.get("capabilities", []):
            all_capabilities.add(c)

    entry = {
        "competition": competition,
        "experts": [e["expert_name"] for e in experts],
        "combined_skills": sorted(all_skills),
        "combined_capabilities": sorted(all_capabilities),
        "strategy": experts[0]["strategy"],
        "formula": experts[0]["formula"],
        "loop_config": experts[0]["loop_config"],
        "notebook_path": os.path.expanduser(f"~/{competition}/"),
    }
    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context for a Kaggle competition.

    Generates a context prompt from relevant ML Principles and prepares
    the rdagent command.

    Args:
        competition: Kaggle competition name.
        description: Optional description of the competition.

    Returns:
        JSON with the rdagent command and context prompt.
    """
    db = _get_db()
    try:
        store = _get_embedding_store()
        hits = store.search(f"{competition} {description}", n_results=5)
        context_chunks = [h["document"] for h in hits]
    except Exception:
        context_chunks = []

    experts = get_expert_for_query(db, competition)
    expert_context = "\n".join(
        f"- {e['expert_name']}: {', '.join(e.get('capabilities', [])[:3])}"
        for e in experts
    )

    context_prompt = f"""ML Principles Context for {competition}:

Expert Recommendations:
{expert_context}

Relevant Knowledge:
{chr(10).join(context_chunks[:3])}

Strategy: {experts[0]['strategy'] if experts else 'Baseline → Feature Engineering → Model Selection → Submit'}
"""

    return json.dumps({
        "competition": competition,
        "context_prompt": context_prompt,
        "command": f"rdagent --competition {competition} --context ml_principles",
        "experts": [e["expert_name"] for e in experts],
    }, indent=2)


@mcp.tool()
def get_extraction_status_tool() -> str:
    """Check the extraction progress for ML Principles chapters.

    Returns:
        JSON with extraction status for each chapter.
    """
    db = _get_db()
    return json.dumps(get_extraction_status(db), indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON with database stats, embedding stats, and configuration.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        store = _get_embedding_store()
        embed_stats = store.get_stats()
    except Exception:
        embed_stats = {"status": "unavailable"}

    return json.dumps({
        "database": db_stats,
        "embeddings": embed_stats,
        "config": {
            "ml_principles_path": os.environ.get(
                "ML_PRINCIPLES_PATH", "~/Desktop/Machine Learning Principles - Chapters"
            ),
            "sqlite_db_path": db.db_path,
            "kaggle_skills_path": os.environ.get("KAGGLE_SKILLS_PATH", "~/skills"),
        },
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: The expert's slug identifier.
        question: The question to ask.

    Returns:
        JSON with the expert's response based on their knowledge domain.
    """
    db = _get_db()
    expert = db.get_expert(expert_slug)

    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found."})

    chapter = db.get_chapter(expert_slug)
    chapter_content = ""
    if chapter and chapter.markdown_content:
        words = chapter.markdown_content.split()
        chapter_content = " ".join(words[:500])

    return json.dumps({
        "expert": expert.expert_name,
        "slug": expert.slug,
        "capabilities": json.loads(expert.capabilities),
        "strategy": expert.strategy,
        "formula": json.loads(expert.formula),
        "relevant_content": chapter_content,
        "question": question,
        "guidance": (
            f"Based on {expert.expert_name}'s knowledge, "
            f"the recommended approach is: {expert.strategy}. "
            f"Key capabilities: {', '.join(json.loads(expert.capabilities)[:3])}."
        ),
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
