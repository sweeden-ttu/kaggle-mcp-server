"""FastMCP server for MLSysEng MoE - Mixture of Experts system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.docling_worker import (
    discover_chapters,
    extract_all_chapters,
    extract_concepts,
)
from mlsyseng_mcp.expert_registry import ExpertRegistry
from mlsyseng_mcp.loop_controller import LoopController, build_competition_entry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_registry: Optional[ExpertRegistry] = None
_embeddings: Optional[Any] = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(db=_get_db())
    return _registry


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from mlsyseng_mcp.embeddings import EmbeddingStore

            _embeddings = EmbeddingStore()
        except ImportError:
            logger.warning(
                "Embedding dependencies not available. "
                "Install sentence-transformers and chromadb for RAG features."
            )
            return None
    return _embeddings


@mcp.tool()
def extract_knowledge(
    force_reindex: bool = False,
    ml_principles_path: Optional[str] = None,
) -> str:
    """Extract knowledge from ML Principles PDFs, index chapters, and create experts.

    Scans chapter folders, extracts PDF content using docling, generates
    embeddings, and creates expert definitions.

    Args:
        force_reindex: Force re-extraction even if already indexed (default: False)
        ml_principles_path: Override path to ML Principles chapters (optional)

    Returns:
        JSON string with extraction results for each chapter
    """
    db = _get_db()
    registry = _get_registry()

    results = extract_all_chapters(
        base_path=ml_principles_path, db=db, force=force_reindex
    )

    experts = registry.register_all_from_db()

    emb_store = _get_embeddings()
    embedding_results = {}
    if emb_store:
        try:
            embedding_results = emb_store.index_all_chapters(db)
        except Exception as e:
            embedding_results = {"error": str(e)}

    return json.dumps(
        {
            "extraction": results,
            "experts_registered": len(experts),
            "embeddings_indexed": embedding_results,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def evolve(
    competition: str = "titanic",
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> str:
    """Run the convergence loop for a competition entry.

    Uses RAG-informed expert selection and iterates until the state
    converges: ||state[n] - state[n-1]||_2 < epsilon.

    Args:
        competition: Competition name (default: "titanic")
        epsilon: Convergence threshold (default: 0.001)
        max_iterations: Maximum loop iterations (default: 10)
        patience: Consecutive converging iterations needed (default: 3)

    Returns:
        JSON string with convergence loop results
    """
    db = _get_db()
    registry = _get_registry()
    experts = registry.list_experts()

    emb_store = _get_embeddings()
    if emb_store:
        try:
            relevant = emb_store.infer_relevant_experts(competition, db)
            if relevant:
                experts = relevant
        except Exception:
            pass

    controller = LoopController(
        competition=competition,
        experts=experts,
        db=db,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    result = controller.run()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON string with search results including text snippets and metadata
    """
    emb_store = _get_embeddings()
    if emb_store is None:
        db = _get_db()
        concepts = db.get_all_concepts()
        query_lower = query.lower()
        matched = [
            c
            for c in concepts
            if query_lower in c.get("concept_name", "").lower()
            or query_lower in c.get("description", "").lower()
        ]
        return json.dumps(
            {"results": matched[:n_results], "method": "keyword_fallback"},
            indent=2,
            default=str,
        )

    hits = emb_store.search(query, n_results=n_results)
    return json.dumps(
        {"results": hits, "method": "semantic_search"},
        indent=2,
        default=str,
    )


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts.

    Returns:
        JSON string with all expert definitions including capabilities, skills,
        strategies, and formulas
    """
    registry = _get_registry()
    experts = registry.list_experts()
    return json.dumps(experts, indent=2, default=str)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a competition entry using expert knowledge.

    Combines relevant experts' strategies, skills, and evaluation metrics
    into an execution plan for the competition.

    Args:
        competition: Competition name (default: "titanic")

    Returns:
        JSON string with the competition entry plan including experts used,
        skills, metrics, and execution pipeline
    """
    db = _get_db()
    registry = _get_registry()
    experts = registry.list_experts()

    emb_store = _get_embeddings()
    if emb_store:
        try:
            relevant = emb_store.infer_relevant_experts(competition, db)
            if relevant:
                experts = relevant
        except Exception:
            pass

    entry = build_competition_entry(competition, experts, db)
    return json.dumps(entry, indent=2, default=str)


@mcp.tool()
def run_rdagent(
    competition: str = "titanic",
    description: str = "",
) -> str:
    """Prepare rdagent command with ML Principles context.

    Generates the rdagent command and context prompt for a Kaggle
    data_science competition.

    Args:
        competition: Competition name (default: "titanic")
        description: Competition description for context matching

    Returns:
        JSON string with rdagent command, context, and expert recommendations
    """
    db = _get_db()
    registry = _get_registry()
    experts = registry.list_experts()

    search_text = f"{competition} {description}"

    emb_store = _get_embeddings()
    context_snippets = []
    if emb_store:
        try:
            hits = emb_store.search(search_text, n_results=5)
            context_snippets = [h["text"] for h in hits]
        except Exception:
            pass

    relevant_experts = []
    if emb_store:
        try:
            relevant_experts = emb_store.infer_relevant_experts(search_text, db)
        except Exception:
            relevant_experts = experts

    context_prompt = _build_rdagent_context(
        competition, description, context_snippets, relevant_experts
    )

    rdagent_cmd = (
        f"rdagent data_science --competition {competition} "
        f'--context "{context_prompt[:500]}"'
    )

    return json.dumps(
        {
            "command": rdagent_cmd,
            "context_prompt": context_prompt,
            "relevant_experts": [
                {
                    "name": e.get("expert_name"),
                    "slug": e.get("slug"),
                    "relevance": e.get("relevance_score", 1.0),
                }
                for e in relevant_experts[:5]
            ],
            "context_snippets": context_snippets[:3],
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def ask_expert(expert_slug: str, question: str) -> str:
    """Query a specific chapter expert about a topic.

    Args:
        expert_slug: Expert slug identifier (e.g., "08_ml_systems")
        question: Question to ask the expert

    Returns:
        JSON string with expert's context, capabilities, and answer guidance
    """
    registry = _get_registry()
    result = registry.ask_expert(expert_slug, question)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_extraction_status() -> str:
    """Check the status of PDF extraction progress.

    Returns:
        JSON string with extraction statistics and chapter list
    """
    db = _get_db()
    chapters = db.get_chapters()
    stats = db.get_stats()

    return json.dumps(
        {
            "chapters_extracted": len(chapters),
            "chapters": [
                {
                    "name": c["chapter_name"],
                    "words": c.get("word_count", 0),
                    "pages": c.get("page_count", 0),
                    "extracted_at": c.get("extracted_at"),
                }
                for c in chapters
            ],
            "database": stats,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def get_stats() -> str:
    """Get system statistics for the MLSysEng MoE system.

    Returns:
        JSON string with database stats, embedding stats, and expert count
    """
    db = _get_db()
    db_stats = db.get_stats()

    emb_stats = {}
    emb_store = _get_embeddings()
    if emb_store:
        try:
            emb_stats = emb_store.get_stats()
        except Exception as e:
            emb_stats = {"error": str(e)}

    return json.dumps(
        {
            "database": db_stats,
            "embeddings": emb_stats,
            "system": {
                "version": "0.1.0",
                "name": "MLSysEng MoE",
            },
        },
        indent=2,
        default=str,
    )


def _build_rdagent_context(
    competition: str,
    description: str,
    context_snippets: List[str],
    experts: List[Dict[str, Any]],
) -> str:
    """Build a context prompt for rdagent."""
    parts = [f"Competition: {competition}"]

    if description:
        parts.append(f"Description: {description}")

    if experts:
        expert_names = [e.get("expert_name", "") for e in experts[:5]]
        parts.append(f"Relevant ML Principles: {', '.join(expert_names)}")

        all_caps = []
        for e in experts[:3]:
            caps = e.get("capabilities", [])
            if isinstance(caps, str):
                try:
                    caps = json.loads(caps)
                except (json.JSONDecodeError, TypeError):
                    caps = []
            all_caps.extend(caps)
        if all_caps:
            parts.append(f"Capabilities: {'; '.join(all_caps[:5])}")

    if context_snippets:
        parts.append("Knowledge Context:")
        for snippet in context_snippets[:3]:
            parts.append(f"  - {snippet[:200]}")

    return "\n".join(parts)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
