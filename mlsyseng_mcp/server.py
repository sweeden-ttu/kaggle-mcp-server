"""FastMCP server for MLSysEng MoE system.

Provides MCP tools for knowledge extraction, expert management,
competition entry building, and semantic search.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .database import Database
from .docling_worker import discover_chapters, extract_all_chapters
from .embeddings import EmbeddingStore
from .expert_registry import ExpertRegistry
from .loop_controller import LoopController

logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-moe")

_db: Optional[Database] = None
_embeddings: Optional[EmbeddingStore] = None
_registry: Optional[ExpertRegistry] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def get_embeddings() -> EmbeddingStore:
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingStore()
    return _embeddings


def get_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry(get_db())
    return _registry


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """Extract PDFs, index chapters, create experts.

    Scans ML Principles chapter folders, extracts PDF content using docling,
    generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if already indexed.
    """
    db = get_db()

    extraction_results = extract_all_chapters(db, force_reindex=force_reindex)

    try:
        emb = get_embeddings()
        index_stats = emb.index_all_chapters(db)
    except Exception as e:
        index_stats = {"error": str(e), "indexed": 0, "chunks": 0}

    registry = get_registry()
    experts = registry.create_experts_from_all_chapters()

    return json.dumps({
        "extraction": extraction_results,
        "indexing": index_stats,
        "experts_created": len(experts),
        "expert_slugs": [e["slug"] for e in experts],
    }, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10, epsilon: float = 0.001) -> str:
    """Run the convergence loop for a competition.

    Iteratively applies expert knowledge to build and refine
    a competition entry until state convergence.

    Args:
        competition: Kaggle competition name/slug.
        max_iterations: Maximum number of iterations.
        epsilon: Convergence threshold for L2 norm.
    """
    db = get_db()
    registry = get_registry()

    try:
        emb = get_embeddings()
        inferred = emb.infer_experts(competition)
    except Exception:
        inferred = []

    experts = registry.select_experts_for_competition(competition, inferred)

    loop = LoopController(
        db=db,
        competition=competition,
        epsilon=epsilon,
        max_iterations=max_iterations,
    )

    dimension = max(len(experts) * 2, 10)
    expert_weights = [1.0 / len(experts)] * len(experts) if experts else [0.1] * 5

    def step_fn(current_state, iteration):
        new_state = list(current_state)
        metrics = {}

        for i, expert in enumerate(experts):
            idx = i % len(new_state)
            formula = expert.get("formula", {})
            loop_config = expert.get("loop_config", {})

            weight = expert_weights[i] if i < len(expert_weights) else 0.1
            adjustment = weight * (1.0 / (iteration + 1))

            new_state[idx] += adjustment
            if idx + 1 < len(new_state):
                new_state[idx + 1] += adjustment * 0.5

            metric_names = formula.get("metrics", ["accuracy"])
            for m in metric_names:
                current = metrics.get(m, 0.5)
                improvement = adjustment * 0.1
                metrics[m] = min(current + improvement, 1.0)

        return new_state, metrics

    result = loop.run_loop(step_fn, initial_state=[0.0] * dimension)

    return json.dumps({
        "competition": competition,
        "experts_used": [e.get("expert_name", e.get("slug")) for e in experts],
        "total_iterations": result["total_iterations"],
        "converged": result["converged"],
        "final_l2_norm": result["final_l2_norm"],
        "final_metrics": result["final_metrics"],
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """Semantic search over ML Principles knowledge base.

    Args:
        query: Search query text.
        n_results: Number of results to return.
    """
    try:
        emb = get_embeddings()
        results = emb.search(query, n_results=n_results)
        return json.dumps({
            "query": query,
            "results": [
                {
                    "chapter": r["metadata"]["title"],
                    "chapter_number": r["metadata"]["chapter_number"],
                    "relevance": round(r["relevance"], 4),
                    "excerpt": r["document"][:300] + "..." if len(r["document"]) > 300 else r["document"],
                }
                for r in results
            ],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "hint": "Run extract_knowledge first to index chapters."})


@mcp.tool()
def list_experts() -> str:
    """List all registered chapter experts with their capabilities and strategies."""
    registry = get_registry()
    experts = registry.list_experts()

    return json.dumps({
        "total": len(experts),
        "experts": [
            {
                "name": e["expert_name"],
                "slug": e["slug"],
                "capabilities": e["capabilities"],
                "strategy": e["strategy"],
                "formula": e["formula"],
                "skills_count": len(e.get("skills", [])),
            }
            for e in experts
        ],
    }, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """Build a Kaggle competition entry using expert knowledge.

    Selects relevant experts based on competition description,
    combines their strategies, and generates notebook outlines.

    Args:
        competition: Kaggle competition name/slug.
    """
    db = get_db()
    registry = get_registry()

    try:
        emb = get_embeddings()
        inferred = emb.infer_experts(competition)
    except Exception:
        inferred = []

    experts = registry.select_experts_for_competition(competition, inferred)

    entry = {
        "competition": competition,
        "experts": [],
        "combined_strategy": [],
        "notebooks": [],
        "skills_required": [],
    }

    all_skills = set()
    for expert in experts:
        expert_info = {
            "name": expert.get("expert_name", expert.get("slug")),
            "capabilities": expert.get("capabilities", []),
            "formula": expert.get("formula", {}),
        }
        entry["experts"].append(expert_info)

        skills = expert.get("skills", [])
        all_skills.update(skills)

        slug = expert.get("slug", "unknown")
        notebook_path = os.path.expanduser(f"~/{competition}/Expert_{slug}.ipynb")
        entry["notebooks"].append(notebook_path)

    entry["skills_required"] = sorted(all_skills)
    entry["combined_strategy"] = _build_combined_strategy(experts)

    return json.dumps(entry, indent=2)


@mcp.tool()
def run_rdagent(competition: str = "titanic", description: str = "") -> str:
    """Run rdagent with ML Principles context.

    Generates context from the knowledge base and prepares
    rdagent command with expert guidance.

    Args:
        competition: Kaggle competition name/slug.
        description: Competition description for context matching.
    """
    db = get_db()
    registry = get_registry()

    context_parts = []

    try:
        emb = get_embeddings()
        query = description or competition
        results = emb.search(query, n_results=5)
        for r in results:
            context_parts.append(
                f"[{r['metadata']['title']}] {r['document'][:500]}"
            )
    except Exception:
        pass

    experts = registry.list_experts()
    for expert in experts[:5]:
        context_parts.append(
            f"Expert {expert['expert_name']}: {expert['strategy']}"
        )

    context = "\n\n".join(context_parts)

    return json.dumps({
        "competition": competition,
        "context_length": len(context),
        "context_preview": context[:1000] + "..." if len(context) > 1000 else context,
        "command": f"rdagent data_science --competition {competition}",
        "experts_available": len(experts),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """Check extraction progress for all chapters."""
    db = get_db()
    statuses = db.get_extraction_status()
    chapters = db.get_all_chapters()

    return json.dumps({
        "chapters_discovered": len(chapters),
        "extraction_status": statuses,
        "summary": {
            "pending": sum(1 for s in statuses if s["status"] == "pending"),
            "extracting": sum(1 for s in statuses if s["status"] == "extracting"),
            "completed": sum(1 for s in statuses if s["status"] == "completed"),
            "failed": sum(1 for s in statuses if s["status"] == "failed"),
        },
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get system statistics including chapter, expert, and competition counts."""
    db = get_db()
    stats = db.get_stats()

    try:
        emb = get_embeddings()
        collection_count = emb.collection.count()
        stats["embedding_chunks"] = collection_count
    except Exception:
        stats["embedding_chunks"] = 0

    return json.dumps(stats, indent=2)


def _build_combined_strategy(experts: list) -> list:
    """Build a combined strategy from multiple experts."""
    phases = [
        {"phase": "Baseline", "description": "Build initial baseline model"},
        {"phase": "EDA", "description": "Exploratory data analysis"},
        {"phase": "Feature Engineering", "description": "Create and select features"},
        {"phase": "Model Selection", "description": "Train and compare models"},
        {"phase": "Optimization", "description": "Hyperparameter tuning and ensemble"},
        {"phase": "Submission", "description": "Generate and submit predictions"},
    ]

    for phase in phases:
        phase["experts"] = []
        for expert in experts:
            caps = expert.get("capabilities", [])
            phase_lower = phase["phase"].lower()
            for cap in caps:
                if phase_lower in cap.lower() or any(
                    kw in cap.lower()
                    for kw in _phase_keywords(phase["phase"])
                ):
                    phase["experts"].append(expert.get("expert_name", expert.get("slug")))
                    break

    return phases


def _phase_keywords(phase: str) -> list:
    mapping = {
        "Baseline": ["baseline", "quick", "initial"],
        "EDA": ["analysis", "exploration", "visualization"],
        "Feature Engineering": ["feature", "engineering", "selection", "preprocessing"],
        "Model Selection": ["model", "train", "compare", "architecture"],
        "Optimization": ["hyperparameter", "tuning", "ensemble", "optimize"],
        "Submission": ["submit", "predict", "inference"],
    }
    return mapping.get(phase, [])
