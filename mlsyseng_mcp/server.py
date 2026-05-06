"""FastMCP server for MLSysEng MoE system.

Provides MCP tools for knowledge extraction, expert management,
competition entry building, and state convergence loops.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.docling_worker import DoclingWorker
from mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_mcp.expert_registry import ExpertRegistry
from mlsyseng_mcp.loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_worker: Optional[DoclingWorker] = None
_embeddings: Optional[EmbeddingEngine] = None
_registry: Optional[ExpertRegistry] = None
_loop: Optional[LoopController] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_worker() -> DoclingWorker:
    global _worker
    if _worker is None:
        _worker = DoclingWorker(_get_db())
    return _worker


def _get_embeddings() -> EmbeddingEngine:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingEngine(_get_db())
    return _embeddings


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(_get_db())
    return _registry


def _get_loop() -> LoopController:
    global _loop
    if _loop is None:
        _loop = LoopController(_get_db())
    return _loop


# ── Knowledge Extraction Tools ──


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, create experts.

    Scans chapter folders, extracts PDF content via docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if already indexed.
    """
    worker = _get_worker()
    embeddings = _get_embeddings()
    registry = _get_registry()

    chapters = worker.extract_all(force_reindex=force_reindex)

    embed_stats = {"chapters_indexed": 0, "total_chunks": 0}
    if chapters:
        embed_stats = embeddings.index_all_chapters()

    experts = registry.register_all_chapters()

    return json.dumps(
        {
            "status": "complete",
            "chapters_extracted": len(chapters),
            "embedding_stats": embed_stats,
            "experts_created": len(experts),
        },
        indent=2,
    )


@mcp.tool()
def evolve(competition: str = "titanic") -> str:
    """Run the full evolve pipeline: extract knowledge, select experts, run convergence loop.

    Args:
        competition: The Kaggle competition to optimize for.
    """
    worker = _get_worker()
    embeddings = _get_embeddings()
    registry = _get_registry()
    loop = _get_loop()

    chapters = worker.extract_all(force_reindex=False)
    if chapters:
        embeddings.index_all_chapters()
        registry.register_all_chapters()

    experts = registry.get_experts_for_competition(competition, embeddings)
    if not experts:
        return json.dumps({"error": "No experts available. Run extract-knowledge first."})

    result = loop.run_expert_loop(competition, experts)
    entry = registry.build_competition_entry(competition, experts)

    return json.dumps(
        {
            "status": "complete",
            "competition": competition,
            "convergence": result,
            "entry": entry,
        },
        indent=2,
    )


# ── Search Tools ──


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Natural language search query.
        n_results: Number of results to return.
    """
    embeddings = _get_embeddings()
    results = embeddings.search(query, n_results=n_results)
    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "chapter": r["metadata"]["chapter_name"],
                    "relevance": round(r["relevance"], 4),
                    "excerpt": r["document"][:300] + "..."
                    if len(r["document"]) > 300
                    else r["document"],
                }
                for r in results
            ],
        },
        indent=2,
    )


# ── Expert Tools ──


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and skills."""
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(
        {
            "experts": [
                {
                    "name": e.expert_name,
                    "slug": e.slug,
                    "capabilities": e.capabilities,
                    "skills": e.skills,
                    "strategy": e.strategy,
                    "metrics": e.formula.get("metrics", []),
                }
                for e in experts
            ],
            "total": len(experts),
        },
        indent=2,
    )


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert for advice on a topic.

    Args:
        expert_slug: The slug identifier of the expert (e.g., '08_ml_systems').
        question: The question to ask the expert.
    """
    registry = _get_registry()
    embeddings = _get_embeddings()

    expert = registry.get_expert(expert_slug)
    if not expert:
        return json.dumps({"error": f"Expert '{expert_slug}' not found"})

    context = embeddings.search(question, n_results=3, chapter_filter=expert.chapter_id)

    return json.dumps(
        {
            "expert": expert.expert_name,
            "capabilities": expert.capabilities,
            "strategy": expert.strategy,
            "formula": expert.formula,
            "relevant_context": [
                {
                    "excerpt": r["document"][:500],
                    "relevance": round(r["relevance"], 4),
                }
                for r in context
            ],
        },
        indent=2,
    )


# ── Competition Tools ──


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a Kaggle competition entry using expert knowledge.

    Selects the most relevant experts for the competition and
    combines their skills, capabilities, and strategies.

    Args:
        competition: The Kaggle competition name or description.
    """
    registry = _get_registry()
    embeddings = _get_embeddings()

    experts = registry.get_experts_for_competition(competition, embeddings)
    if not experts:
        return json.dumps({"error": "No experts available. Run extract-knowledge first."})

    entry = registry.build_competition_entry(competition, experts)
    return json.dumps(entry, indent=2)


# ── RDAgent Integration ──


@mcp.tool()
def run_rdagent(competition: str, description: str = "") -> str:
    """Prepare rdagent context with ML Principles knowledge.

    Generates a context prompt enriched with relevant ML concepts
    for guiding rdagent on a Kaggle data_science competition.

    Args:
        competition: The competition name.
        description: Optional description of the competition task.
    """
    embeddings = _get_embeddings()
    registry = _get_registry()

    query = f"{competition} {description}".strip()
    search_results = embeddings.search(query, n_results=10)

    experts = registry.get_experts_for_competition(query, embeddings)

    context_parts = [
        f"## ML Principles Context for: {competition}",
        "",
    ]

    if description:
        context_parts.append(f"**Task:** {description}")
        context_parts.append("")

    if experts:
        context_parts.append("### Recommended Experts")
        for expert in experts[:5]:
            context_parts.append(f"- **{expert.expert_name}**: {', '.join(expert.capabilities[:3])}")
        context_parts.append("")

    if search_results:
        context_parts.append("### Relevant ML Principles")
        for r in search_results[:5]:
            context_parts.append(
                f"- [{r['metadata']['chapter_name']}] (relevance: {r['relevance']:.3f}): "
                f"{r['document'][:200]}..."
            )
        context_parts.append("")

    if experts:
        all_skills = set()
        for e in experts[:5]:
            all_skills.update(e.skills)
        context_parts.append("### Skills to Apply")
        for skill in sorted(all_skills):
            context_parts.append(f"- {skill}")

    context_prompt = "\n".join(context_parts)

    return json.dumps(
        {
            "competition": competition,
            "context_prompt": context_prompt,
            "experts_selected": len(experts),
            "principles_found": len(search_results),
            "rdagent_command": (
                f"rdagent --competition {competition} "
                f"--context-file ml_principles_context.md"
            ),
        },
        indent=2,
    )


# ── Status Tools ──


@mcp.tool()
def get_extraction_status() -> str:
    """Check the status of PDF extraction and indexing."""
    worker = _get_worker()
    return json.dumps(worker.get_status(), indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapters, experts, and embeddings."""
    db = _get_db()
    embeddings = _get_embeddings()

    db_stats = db.get_stats()
    embed_stats = embeddings.get_stats()

    return json.dumps(
        {
            "database": db_stats,
            "embeddings": embed_stats,
        },
        indent=2,
    )


# ── Index/Search Tools (ML Principles RD-Agent compatibility) ──


@mcp.tool()
def index_ml_chapters(force_reindex: bool = False) -> str:
    """Index all PDF chapters (extract + embed + store).

    Alias for extract-knowledge for ML Principles RD-Agent compatibility.

    Args:
        force_reindex: Whether to re-extract already indexed chapters.
    """
    return extract_knowledge(force_reindex=force_reindex)


@mcp.tool()
def search_ml_principles(query: str, n_results: int = 5) -> str:
    """Semantic search over indexed ML Principles knowledge base.

    Args:
        query: Search query string.
        n_results: Number of results to return.
    """
    return search_concepts(query=query, n_results=n_results)


@mcp.tool()
def get_rdagent_context(competition_name: str, description: str = "") -> str:
    """Generate context prompt for rdagent with ML Principles knowledge.

    Args:
        competition_name: The competition name.
        description: Description of the competition task.
    """
    return run_rdagent(competition=competition_name, description=description)


@mcp.tool()
def list_indexed_chapters() -> str:
    """List all indexed chapters with their metadata."""
    db = _get_db()
    chapters = db.list_chapters()
    return json.dumps(
        {
            "chapters": [
                {
                    "id": c.chapter_id,
                    "name": c.chapter_name,
                    "word_count": c.word_count,
                    "concepts": c.concepts[:10],
                    "extracted_at": c.extracted_at,
                }
                for c in chapters
            ],
            "total": len(chapters),
        },
        indent=2,
    )


@mcp.tool()
def get_indexing_stats() -> str:
    """Get indexing statistics for the knowledge base."""
    return get_stats()


def main():
    """Run the MLSysEng MoE MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
