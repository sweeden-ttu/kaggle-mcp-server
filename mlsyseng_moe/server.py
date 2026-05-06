"""FastMCP server for MLSysEng MoE system.

Provides MCP tools for knowledge extraction, expert querying,
competition entry building, and convergence loop execution.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import database as db
from . import embeddings
from . import expert_registry
from .docling_worker import run_extraction
from .loop_controller import run_competition_loop

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "mlsyseng-moe",
    description="ML Systems Expert Mixture of Experts - extracts knowledge from ML Principles, "
    "registers chapter experts, builds Kaggle entries with RAG + convergence loops",
)


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, create experts.

    Scans chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.
    """
    db.init_db()

    results = run_extraction(force_reindex=force_reindex)

    experts = expert_registry.register_experts_from_extraction(results)

    indexing_result = embeddings.index_all_chapters()

    summary = {
        "extraction": {
            "chapters_processed": len(results),
            "successful": sum(1 for r in results if r.get("status") == "completed"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
        },
        "experts_created": len(experts),
        "indexing": indexing_result,
    }

    return json.dumps(summary, indent=2)


@mcp.tool()
def evolve(competition: str, description: str = "", epsilon: float = 0.001, max_iterations: int = 10, patience: int = 3) -> str:
    """Run the convergence loop for a competition entry.

    Iteratively applies expert transformations until state convergence:
    ||state[n] - state[n-1]||_2 < epsilon
    """
    db.init_db()

    experts = expert_registry.list_experts()
    if not experts:
        return json.dumps({"error": "No experts registered. Run extract_knowledge first."})

    context = embeddings.get_context_for_competition(competition, description)

    relevant = embeddings.infer_skills_for_competition(competition, description)
    selected_experts = [r["expert"] for r in relevant[:5]] if relevant else experts[:5]

    result = run_competition_loop(
        experts=selected_experts,
        competition_name=competition,
        competition_context=context,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base."""
    results = embeddings.search(query, n_results=n_results)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities."""
    experts = expert_registry.list_experts()
    summary = []
    for e in experts:
        summary.append({
            "expert_name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e["capabilities"],
            "skills": e["skills"],
            "strategy": e["strategy"],
            "formula": e["formula"],
        })
    return json.dumps(summary, indent=2)


@mcp.tool()
def build_entry(competition: str, description: str = "") -> str:
    """Build a competition entry using expert knowledge and RAG.

    Selects relevant experts, generates context, and produces
    a structured plan for the competition.
    """
    db.init_db()

    context = embeddings.get_context_for_competition(competition, description)

    relevant = embeddings.infer_skills_for_competition(competition, description)

    experts_selected = []
    all_skills = set()
    for r in relevant[:5]:
        expert = r["expert"]
        experts_selected.append({
            "expert_name": expert["expert_name"],
            "relevance": r["relevance"],
            "capabilities": expert["capabilities"],
            "strategy": expert["strategy"],
        })
        for skill in expert.get("skills", []):
            all_skills.add(skill)

    if not experts_selected:
        all_experts = expert_registry.list_experts()
        for expert in all_experts[:3]:
            experts_selected.append({
                "expert_name": expert["expert_name"],
                "relevance": 0.5,
                "capabilities": expert["capabilities"],
                "strategy": expert["strategy"],
            })
            for skill in expert.get("skills", []):
                all_skills.add(skill)

    entry = {
        "competition": competition,
        "description": description,
        "experts_selected": experts_selected,
        "skills_needed": sorted(all_skills),
        "strategy": "Baseline → EDA → Feature Engineering → Model Selection → Ensemble → Submit",
        "context_summary": context[:1000],
        "recommended_steps": [
            "1. Load and explore competition data",
            "2. Apply expert-recommended preprocessing",
            "3. Build baseline model using primary expert strategy",
            "4. Iterate feature engineering based on chapter knowledge",
            "5. Apply ensemble techniques from relevant experts",
            "6. Run convergence loop until ||state[n]-state[n-1]||₂ < ε",
            "7. Generate final submission",
        ],
    }

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Run rdagent with ML Principles context for a competition.

    Generates context from knowledge base and prepares rdagent invocation.
    """
    context = embeddings.get_context_for_competition(competition, description)
    experts = expert_registry.list_experts()

    expert_summary = "\n".join(
        f"- {e['expert_name']}: {', '.join(e['capabilities'][:2])}"
        for e in experts[:5]
    )

    rdagent_prompt = f"""Competition: {competition}
Description: {description}

## Available Expert Knowledge
{expert_summary}

## Relevant ML Principles
{context[:2000]}

## Strategy
1. Apply systematic feature engineering from ML Principles
2. Use ensemble methods where applicable
3. Monitor convergence of validation metrics
4. Submit when ||state[n] - state[n-1]||_2 < 0.001
"""

    return json.dumps({
        "competition": competition,
        "rdagent_context": rdagent_prompt,
        "command": f"rdagent --competition {competition} --context ml_principles",
        "experts_available": len(experts),
    }, indent=2)


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert with a question."""
    result = expert_registry.query_expert(expert_slug, question)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for all chapters."""
    db.init_db()
    status = db.get_extraction_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics: chapters, blocks, concepts, experts."""
    db.init_db()
    stats = db.get_stats()
    return json.dumps(stats, indent=2)


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
