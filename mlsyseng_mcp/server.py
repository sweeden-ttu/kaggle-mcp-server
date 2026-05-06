"""FastMCP server for MLSysEng MoE.

Provides MCP tools for knowledge extraction, expert management,
RAG search, competition entry building, and convergence loops.
"""

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import database as db
from . import docling_worker
from . import embeddings
from . import expert_registry
from . import loop_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: If true, re-extract already-indexed chapters
    """
    db.init_db()

    extraction_results = docling_worker.extract_all(force_reindex=force_reindex)

    index_result = {}
    has_chapters = any(r.get("status") == "completed" for r in extraction_results)
    if has_chapters:
        try:
            index_result = embeddings.index_chapters()
        except Exception as e:
            index_result = {"status": "error", "message": str(e)}

    experts = expert_registry.register_experts_from_chapters()

    return json.dumps({
        "extraction": extraction_results,
        "indexing": index_result,
        "experts_registered": experts,
        "stats": db.get_stats(),
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic",
           epsilon: float = 0.001,
           max_iterations: int = 10,
           patience: int = 3) -> str:
    """Run the convergence loop for a competition.

    Iteratively refines competition entries using expert knowledge
    until state convergence (||state[n] - state[n-1]||_2 < epsilon).

    Args:
        competition: Competition name/slug
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations
        patience: Consecutive converging iterations required to exit
    """
    db.init_db()
    experts = db.get_all_experts()

    if not experts:
        return json.dumps({
            "status": "error",
            "message": "No experts registered. Run extract-knowledge first.",
        })

    result = loop_controller.run_convergence_loop(
        competition=competition,
        experts=experts,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5,
                    chapter: Optional[str] = None) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language search query
        n_results: Number of results to return
        chapter: Optional chapter folder name to filter by
    """
    try:
        results = embeddings.search(query, n_results=n_results,
                                    chapter_filter=chapter)
        return json.dumps({"query": query, "results": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e),
                           "hint": "Run extract-knowledge first to build the index"})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills."""
    db.init_db()
    experts = expert_registry.export_expert_definitions()
    return json.dumps({
        "count": len(experts),
        "experts": experts,
    }, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Selects relevant experts via RAG, applies their strategies,
    and produces a structured competition plan.

    Args:
        competition: Competition name/slug
    """
    db.init_db()

    try:
        relevant = embeddings.infer_skills(competition, n_results=5)
    except Exception:
        relevant = []

    if not relevant:
        experts = db.get_all_experts()
        relevant = [{
            "expert_slug": e["slug"],
            "expert_name": e["expert_name"],
            "relevance_score": 0.5,
            "capabilities": e.get("capabilities", []),
            "skills": e.get("skills", []),
            "strategy": e.get("strategy", ""),
        } for e in experts[:3]]

    plan = {
        "competition": competition,
        "selected_experts": relevant,
        "pipeline": [],
    }

    for i, expert_info in enumerate(relevant):
        stage = {
            "order": i + 1,
            "expert": expert_info["expert_slug"],
            "name": expert_info["expert_name"],
            "relevance": expert_info.get("relevance_score"),
            "strategy": expert_info.get("strategy", ""),
            "skills_to_apply": expert_info.get("skills", []),
            "capabilities": expert_info.get("capabilities", []),
        }
        plan["pipeline"].append(stage)

    return json.dumps(plan, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic",
                description: str = "") -> str:
    """Generate rdagent context from ML Principles for a competition.

    Searches the knowledge base for relevant principles and generates
    a context prompt that can be passed to rdagent.

    Args:
        competition: Competition name
        description: Optional competition description for better matching
    """
    query = f"{competition} {description}".strip() or competition

    try:
        search_results = embeddings.search(query, n_results=10)
    except Exception:
        search_results = []

    try:
        skill_matches = embeddings.infer_skills(query, n_results=5)
    except Exception:
        skill_matches = []

    context_parts = [
        f"# ML Principles Context for: {competition}",
        "",
    ]

    if search_results:
        context_parts.append("## Relevant Knowledge")
        for hit in search_results[:5]:
            context_parts.append(
                f"\n### From {hit.get('title', 'Unknown')} "
                f"(similarity: {hit.get('similarity', 'N/A')})"
            )
            context_parts.append(hit.get("text", ""))

    if skill_matches:
        context_parts.append("\n## Recommended Experts")
        for match in skill_matches:
            context_parts.append(
                f"- **{match['expert_name']}** "
                f"(relevance: {match.get('relevance_score', 'N/A')}): "
                f"{', '.join(match.get('capabilities', []))}"
            )

    context_prompt = "\n".join(context_parts)

    return json.dumps({
        "competition": competition,
        "context_prompt": context_prompt,
        "experts_matched": len(skill_matches),
        "knowledge_chunks": len(search_results),
        "rdagent_command": (
            f"rdagent data_science --competition {competition} "
            f"--context 'ML Principles guided'"
        ),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the extraction progress for all chapters."""
    db.init_db()
    status = db.get_extraction_status()
    return json.dumps({"chapters": status}, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics: chapters, experts, concepts, entries."""
    db.init_db()
    stats = db.get_stats()
    return json.dumps(stats, indent=2)


def _register_dynamic_ask_tools():
    """Register ask_<expert> tools for each registered expert.

    Called after experts are loaded to create per-expert query tools.
    """
    db.init_db()
    experts = db.get_all_experts()

    for expert in experts:
        slug = expert["slug"]
        name = expert["expert_name"]

        def _make_ask_fn(s, n):
            def ask_expert(question: str) -> str:
                f"""Query the {n} expert.

                Args:
                    question: Your question for this expert
                """
                exp = db.get_expert(s)
                if not exp:
                    return json.dumps({"error": f"Expert {s} not found"})

                try:
                    results = embeddings.search(question, n_results=3,
                                                chapter_filter=None)
                except Exception:
                    results = []

                return json.dumps({
                    "expert": n,
                    "slug": s,
                    "capabilities": exp.get("capabilities", []),
                    "strategy": exp.get("strategy", ""),
                    "formula": exp.get("formula", {}),
                    "relevant_knowledge": results,
                    "answer_context": (
                        f"As the {n} expert, I recommend: "
                        f"{exp.get('strategy', 'Apply systematic approach')}"
                    ),
                }, indent=2)
            ask_expert.__name__ = f"ask_{s}"
            ask_expert.__doc__ = f"Query the {n} expert about ML concepts."
            return ask_expert

        try:
            fn = _make_ask_fn(slug, name)
            mcp.tool()(fn)
        except Exception as e:
            logger.debug("Could not register ask_%s: %s", slug, e)


try:
    _register_dynamic_ask_tools()
except Exception:
    pass
