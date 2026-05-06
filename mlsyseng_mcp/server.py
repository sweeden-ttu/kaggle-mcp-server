"""FastMCP server for the MLSysEng MoE system.

Exposes MCP tools for knowledge extraction, expert management,
competition entry building, and convergence loop execution.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.docling_worker import extract_all_chapters, discover_chapters
from mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_mcp.expert_registry import ExpertRegistry
from mlsyseng_mcp.loop_controller import LoopConfig, LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-mcp")

_db: Optional[Database] = None
_embedding_engine: Optional[EmbeddingEngine] = None
_expert_registry: Optional[ExpertRegistry] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_embedding_engine() -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine


def _get_expert_registry() -> ExpertRegistry:
    global _expert_registry
    if _expert_registry is None:
        _expert_registry = ExpertRegistry(_get_db())
    return _expert_registry


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans all chapter folders in ML Principles, extracts PDF content
    using docling (with fallbacks), generates embeddings, and creates
    expert definitions.

    Args:
        force_reindex: If True, re-extract even previously indexed chapters.

    Returns:
        JSON summary of extraction results.
    """
    db = _get_db()

    extraction_result = extract_all_chapters(db, force_reindex=force_reindex)

    try:
        engine = _get_embedding_engine()
        index_result = engine.index_all_chapters(db)
        extraction_result["chunks_indexed"] = sum(index_result.values())
    except Exception as e:
        extraction_result["embedding_error"] = str(e)
        extraction_result["chunks_indexed"] = 0

    registry = _get_expert_registry()
    experts = registry.create_experts_from_all_chapters()
    extraction_result["experts_created"] = len(experts)

    return json.dumps(extraction_result, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the convergence loop for a competition.

    Iteratively refines the solution using expert knowledge until
    the state vector converges (||state[n] - state[n-1]||_2 < epsilon).

    Args:
        competition: Competition name/slug (default: "titanic").

    Returns:
        JSON summary with convergence results.
    """
    db = _get_db()
    registry = _get_expert_registry()

    experts = registry.list_experts()
    if not experts:
        return json.dumps({
            "error": "No experts registered. Run extract_knowledge first.",
            "competition": competition,
        })

    controller = LoopController(db)
    result = controller.run_loop(competition, experts)
    return json.dumps(result, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles.

    Args:
        query: Search query (e.g., "neural network optimization").
        n_results: Number of results to return (default: 5).

    Returns:
        JSON array of matching chunks with chapter info and similarity scores.
    """
    try:
        engine = _get_embedding_engine()
        results = engine.search(query, n_results=n_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


@mcp.tool()
def list_experts() -> str:
    """List all chapter experts.

    Returns:
        JSON array of expert definitions with capabilities, skills,
        strategy, formula, and loop configuration.
    """
    registry = _get_expert_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build competition entry using expert knowledge.

    Performs RAG-informed skill selection to identify the best experts
    and their recommended skills for a given competition.

    Args:
        competition: Competition name (default: "titanic").

    Returns:
        JSON with recommended experts, skills, and entry plan.
    """
    db = _get_db()
    registry = _get_expert_registry()

    try:
        engine = _get_embedding_engine()
        rag_results = engine.infer_skills_for_competition(competition, db)
    except Exception:
        rag_results = []

    keyword_experts = registry.get_experts_for_competition(competition)

    all_experts = []
    seen_slugs = set()
    for item in rag_results:
        expert = item["expert"]
        if expert["slug"] not in seen_slugs:
            seen_slugs.add(expert["slug"])
            all_experts.append({
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert.get("capabilities", []),
                "skills": expert.get("skills", []),
                "relevance": item.get("relevance", 0),
                "source": "rag",
            })
    for expert in keyword_experts:
        if expert["slug"] not in seen_slugs:
            seen_slugs.add(expert["slug"])
            all_experts.append({
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert.get("capabilities", []),
                "skills": expert.get("skills", []),
                "relevance": 0.5,
                "source": "keyword",
            })

    all_skills = set()
    for expert in all_experts:
        for skill in expert.get("skills", []):
            all_skills.add(skill)

    entry = {
        "competition": competition,
        "experts": all_experts,
        "recommended_skills": sorted(all_skills),
        "strategy": "Baseline → EDA → Feature Engineering → Model Selection → Submit",
        "entry_plan": {
            "step_1": "Download competition data and explore",
            "step_2": "Apply expert preprocessing recommendations",
            "step_3": "Build baseline model with recommended architecture",
            "step_4": "Run convergence loop for iterative refinement",
            "step_5": "Generate submission notebook",
        },
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
) -> str:
    """Run rdagent with ML Principles context.

    Generates a context prompt from relevant ML principles for the
    given competition and returns the command configuration.

    Args:
        competition: Competition name (default: "titanic").
        description: Additional description of the competition task.

    Returns:
        JSON with rdagent configuration and context prompt.
    """
    try:
        engine = _get_embedding_engine()
        context = engine.generate_rdagent_context(competition, description)
    except Exception as e:
        context = f"[Embedding engine unavailable: {e}]"

    return json.dumps({
        "competition": competition,
        "description": description,
        "context_prompt": context,
        "rdagent_command": f"rdagent --competition {competition} --mode data_science",
        "instructions": (
            "Use the context_prompt above to guide rdagent's approach. "
            "The ML Principles content provides foundational knowledge "
            "for building competitive solutions."
        ),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress.

    Returns:
        JSON array of extraction status entries with chapter, status,
        timestamps, and page counts.
    """
    db = _get_db()
    statuses = db.get_extraction_status()
    return json.dumps(statuses, indent=2)


@mcp.tool()
def get_stats() -> str:
    """System statistics.

    Returns:
        JSON with chapter count, expert count, extraction stats,
        embedding stats, and database info.
    """
    db = _get_db()
    db_stats = db.get_stats()

    try:
        engine = _get_embedding_engine()
        emb_stats = engine.get_stats()
    except Exception:
        emb_stats = {"status": "unavailable"}

    chapters = discover_chapters()

    return json.dumps({
        "database": db_stats,
        "embeddings": emb_stats,
        "discovered_chapters": len(chapters),
        "chapter_folders": [c["chapter_number"] + " - " + c["title"] for c in chapters],
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems").
        question: Question to ask the expert.

    Returns:
        JSON with expert info and relevant knowledge for the question.
    """
    db = _get_db()
    registry = _get_expert_registry()

    expert = registry.get_expert(expert_slug)
    if not expert:
        available = [e["slug"] for e in registry.list_experts()]
        return json.dumps({
            "error": f"Expert '{expert_slug}' not found",
            "available_experts": available,
        })

    chapter_id = expert.get("chapter_id")
    relevant_content = ""
    if chapter_id:
        chapters = db.list_chapters()
        for ch in chapters:
            if ch["id"] == chapter_id:
                content = ch.get("content_md", "")
                question_lower = question.lower()
                paragraphs = content.split("\n\n")
                scored = []
                for para in paragraphs:
                    if not para.strip():
                        continue
                    score = sum(
                        1 for word in question_lower.split()
                        if word in para.lower() and len(word) > 3
                    )
                    if score > 0:
                        scored.append((score, para))
                scored.sort(key=lambda x: -x[0])
                relevant_content = "\n\n".join(p for _, p in scored[:3])
                break

    return json.dumps({
        "expert": {
            "name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
        },
        "question": question,
        "relevant_knowledge": relevant_content or "No specific content found. Run extract_knowledge first.",
        "recommendations": expert.get("skills", []),
    }, indent=2)


@mcp.tool()
def list_indexed_chapters() -> str:
    """List all indexed chapters.

    Returns:
        JSON array of chapter summaries with number, title, concepts, and page count.
    """
    db = _get_db()
    chapters = db.list_chapters()
    summaries = []
    for ch in chapters:
        concepts = ch.get("concepts", "[]")
        if isinstance(concepts, str):
            try:
                concepts = json.loads(concepts)
            except (json.JSONDecodeError, TypeError):
                concepts = []
        summaries.append({
            "chapter_number": ch["chapter_number"],
            "title": ch["title"],
            "concepts": concepts,
            "page_count": ch.get("page_count", 0),
            "has_content": bool(ch.get("content_md")),
        })
    return json.dumps(summaries, indent=2)


@mcp.tool()
def get_indexing_stats() -> str:
    """Get indexing statistics.

    Returns:
        JSON with detailed indexing and extraction statistics.
    """
    db = _get_db()
    stats = db.get_stats()
    statuses = db.get_extraction_status()

    status_counts = {}
    for s in statuses:
        st = s.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    return json.dumps({
        **stats,
        "extraction_status_counts": status_counts,
        "extraction_details": statuses,
    }, indent=2)


@mcp.tool()
def search_ml_principles(query: str, n_results: int = 5) -> str:
    """Semantic search over indexed knowledge base.

    Args:
        query: Natural language query.
        n_results: Number of results to return.

    Returns:
        JSON array of matching content with chapter and similarity info.
    """
    return search_concepts(query, n_results)


@mcp.tool()
def index_ml_chapters(force_reindex: bool = False) -> str:
    """Index all PDF chapters (extract + embed + store).

    Args:
        force_reindex: If True, re-extract previously indexed chapters.

    Returns:
        JSON summary of indexing results.
    """
    return extract_knowledge(force_reindex)


@mcp.tool()
def get_rdagent_context(
    competition_name: str,
    description: str = "",
    n_results: int = 5,
) -> str:
    """Generate context prompt for rdagent.

    Args:
        competition_name: Name of the competition.
        description: Description of the competition task.
        n_results: Number of relevant chunks to include.

    Returns:
        Context prompt string with relevant ML principles.
    """
    try:
        engine = _get_embedding_engine()
        return engine.generate_rdagent_context(
            competition_name, description, n_results
        )
    except Exception as e:
        return f"Error generating context: {e}"


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
