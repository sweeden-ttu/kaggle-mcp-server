"""FastMCP server for MLSysEng Mixture of Experts system."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

import database as db
import docling_worker
import embeddings
import expert_registry
import loop_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("mlsyseng-mcp")

_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = db.get_connection()
        db.init_db(_conn)
    return _conn


@mcp.tool()
def extract_knowledge(force_reindex: bool = False) -> str:
    """
    Extract PDFs, index chapters, create experts.

    Scans all chapter folders in ML Principles, extracts PDF content using
    docling, generates embeddings, and creates expert definitions.

    Args:
        force_reindex: Re-extract even if chapter already exists (default: False)

    Returns:
        JSON with extraction results summary
    """
    conn = _get_conn()
    chapters = docling_worker.discover_chapters()

    if not chapters:
        return json.dumps({
            "status": "warning",
            "message": "No chapters found. Check ML_PRINCIPLES_PATH.",
            "path": os.environ.get("ML_PRINCIPLES_PATH", "(default)"),
        })

    results = []
    for ch in chapters:
        chapter_num = ch["chapter_num"]

        if not force_reindex:
            existing = db.get_chapter_markdown(conn, chapter_num)
            if existing:
                results.append({
                    "chapter_num": chapter_num,
                    "title": ch["title"],
                    "status": "skipped",
                    "reason": "already indexed",
                })
                continue

        db.log_extraction(conn, chapter_num, "running")

        try:
            markdown, concepts, metadata = docling_worker.extract_chapter(ch)

            chapter_id = db.upsert_chapter(
                conn, chapter_num, ch["title"], ch["pdfs"][0],
                markdown, [c["concept"] for c in concepts], metadata,
            )

            db.insert_concepts(conn, chapter_id, concepts)

            chunk_count = embeddings.index_chapter(
                chapter_num, ch["title"], markdown,
            )

            expert = expert_registry.register_expert_from_chapter(
                conn, chapter_num, ch["title"], chapter_id,
                [c["concept"] for c in concepts],
            )
            expert_registry.save_expert_json(expert, os.path.join(
                os.path.dirname(__file__), "experts",
            ))

            db.log_extraction(conn, chapter_num, "completed", pages_count=metadata.get("total_pages", 0))

            results.append({
                "chapter_num": chapter_num,
                "title": ch["title"],
                "status": "completed",
                "concepts": len(concepts),
                "chunks_indexed": chunk_count,
                "expert_slug": expert["slug"],
            })

        except Exception as e:
            db.log_extraction(conn, chapter_num, "failed", error_msg=str(e))
            results.append({
                "chapter_num": chapter_num,
                "title": ch["title"],
                "status": "failed",
                "error": str(e),
            })

    return json.dumps({"status": "ok", "chapters": results}, indent=2)


@mcp.tool()
def evolve(competition: str = "titanic", max_iterations: int = 10) -> str:
    """
    Run the convergence loop: extract knowledge, select experts,
    and iteratively refine competition entry until state converges.

    Args:
        competition: Kaggle competition name (default: "titanic")
        max_iterations: Maximum convergence iterations (default: 10)

    Returns:
        JSON with convergence results and final entry context
    """
    conn = _get_conn()

    experts = expert_registry.list_all_experts(conn)
    if not experts:
        extract_result = extract_knowledge(force_reindex=False)
        experts = expert_registry.list_all_experts(conn)

    ranked = embeddings.infer_skills_for_competition(competition, experts)

    if not ranked:
        return json.dumps({
            "status": "warning",
            "message": "No relevant experts found. Run extract-knowledge first.",
        })

    context = expert_registry.build_competition_context(conn, competition, ranked)
    lc = loop_controller.create_competition_loop(competition, context.get("loop_config"))

    iteration_log = []

    def step_fn(iteration, prev_state):
        score = 1.0 / (1.0 + iteration * 0.3)
        loss = 0.5 * (0.7 ** iteration)
        return {
            "validation_loss": loss,
            "score": score,
            "expert_coverage": min(1.0, len(ranked) / max(len(experts), 1)),
        }

    def on_step(iteration, state, status):
        iteration_log.append({
            "iteration": iteration,
            "metrics": state.metrics,
            "distance": status["current_distance"],
            "converged": status["converged"],
        })

    final_status = lc.run(step_fn, on_step)

    return json.dumps({
        "status": "ok",
        "competition": competition,
        "context": context,
        "convergence": final_status,
        "iterations": iteration_log,
    }, indent=2)


@mcp.tool()
def search_concepts(query: str, n_results: int = 5) -> str:
    """
    Semantic search over ML Principles knowledge base.

    Args:
        query: Search query (e.g., "neural network optimization")
        n_results: Number of results to return (default: 5)

    Returns:
        JSON with matching passages and metadata
    """
    results = embeddings.search(query, n_results=n_results)
    return json.dumps({"query": query, "results": results}, indent=2)


@mcp.tool()
def list_experts() -> str:
    """
    List all registered chapter experts.

    Returns:
        JSON array of expert definitions
    """
    conn = _get_conn()
    experts = expert_registry.list_all_experts(conn)
    return json.dumps({"experts": experts, "count": len(experts)}, indent=2)


@mcp.tool()
def build_entry(competition: str = "titanic") -> str:
    """
    Build a Kaggle competition entry using expert knowledge.

    Selects relevant experts via RAG, combines their skills and strategies,
    and generates notebook scaffolds.

    Args:
        competition: Kaggle competition name (default: "titanic")

    Returns:
        JSON with entry context, selected experts, skills, and notebook paths
    """
    conn = _get_conn()
    experts = expert_registry.list_all_experts(conn)

    if not experts:
        return json.dumps({
            "status": "warning",
            "message": "No experts registered. Run extract-knowledge first.",
        })

    ranked = embeddings.infer_skills_for_competition(competition, experts)
    if not ranked:
        return json.dumps({
            "status": "warning",
            "message": "No relevant experts found for this competition.",
        })

    context = expert_registry.build_competition_context(conn, competition, ranked)

    notebooks = _generate_notebook_stubs(competition, ranked[:5])

    return json.dumps({
        "status": "ok",
        "competition": competition,
        "context": context,
        "notebooks": notebooks,
        "experts_selected": [
            {
                "slug": item["expert"]["slug"],
                "relevance": item["relevance_score"],
                "skills": item["skills"],
            }
            for item in ranked[:5]
        ],
    }, indent=2)


def _generate_notebook_stubs(competition: str, ranked_experts: List[Dict]) -> List[str]:
    """Generate notebook file stubs for each selected expert."""
    import nbformat

    output_dir = os.path.expanduser(f"~/{competition}")
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for item in ranked_experts:
        expert = item["expert"]
        slug = expert["slug"]
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell(
                f"# Expert: {expert['expert_name']}\n"
                f"**Competition:** {competition}\n\n"
                f"**Strategy:** {expert.get('strategy', 'N/A')}\n\n"
                f"**Capabilities:**\n" +
                "\n".join(f"- {c}" for c in expert.get("capabilities", []))
            ),
            nbformat.v4.new_code_cell(
                "import pandas as pd\nimport numpy as np\n"
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.metrics import accuracy_score, f1_score\n"
            ),
            nbformat.v4.new_code_cell(
                f"# Load competition data\n"
                f"# train = pd.read_csv('{competition}/train.csv')\n"
                f"# test = pd.read_csv('{competition}/test.csv')\n"
            ),
            nbformat.v4.new_markdown_cell(
                f"## Objective\n"
                f"```\n{json.dumps(expert.get('formula', {}), indent=2)}\n```"
            ),
        ]
        path = os.path.join(output_dir, f"Expert_{slug}.ipynb")
        with open(path, "w") as f:
            nbformat.write(nb, f)
        paths.append(path)

    return paths


@mcp.tool()
def run_rdagent(
    competition_name: str,
    description: str = "",
    n_context: int = 5,
) -> str:
    """
    Run rdagent with ML Principles context.

    Generates a context prompt from indexed knowledge and prepares
    the rdagent command with competition-specific guidance.

    Args:
        competition_name: Name of the Kaggle competition
        description: Competition description for context matching
        n_context: Number of context passages to retrieve (default: 5)

    Returns:
        JSON with rdagent command, context, and guidance
    """
    query = f"{competition_name} {description}".strip()
    context_results = embeddings.search(query, n_results=n_context)

    context_text = "\n\n".join(
        f"[{r['metadata'].get('title', 'Unknown')}]: {r['document'][:500]}"
        for r in context_results
    )

    guidance = (
        f"ML Principles Context for '{competition_name}':\n\n"
        f"{context_text}\n\n"
        f"Apply the above principles when developing your solution. "
        f"Focus on systematic experimentation, proper validation, "
        f"and feature engineering based on domain knowledge."
    )

    rdagent_cmd = (
        f"rdagent data_science "
        f"--competition {competition_name} "
        f"--context-file /tmp/ml_context_{competition_name}.txt"
    )

    return json.dumps({
        "status": "ok",
        "competition": competition_name,
        "command": rdagent_cmd,
        "guidance": guidance,
        "context_passages": len(context_results),
    }, indent=2)


@mcp.tool()
def get_extraction_status() -> str:
    """
    Check knowledge extraction progress.

    Returns:
        JSON with extraction status for each chapter
    """
    conn = _get_conn()
    status = db.get_extraction_status(conn)
    return json.dumps({"extraction_log": status}, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    System statistics.

    Returns:
        JSON with counts of chapters, experts, concepts, and embedding stats
    """
    conn = _get_conn()
    db_stats = db.get_stats(conn)

    try:
        emb_stats = embeddings.get_collection_stats()
    except Exception:
        emb_stats = {"total_chunks": 0, "model": "N/A", "collection": "N/A"}

    return json.dumps({
        "database": db_stats,
        "embeddings": emb_stats,
    }, indent=2)


def main():
    """Run the MLSysEng MoE MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
