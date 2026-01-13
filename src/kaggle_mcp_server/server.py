"""Kaggle MCP Server implementation."""

import json
import os
import re
import subprocess
from typing import Optional, List, Dict, Set, Any
from mcp.server.fastmcp import FastMCP
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize FastMCP server
mcp = FastMCP("kaggle-mcp-server")

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()


@mcp.tool()
def list_competitions(
    group: str = "general",
    category: str = "all",
    sort_by: str = "latestDeadline",
    page: int = 1,
    search: Optional[str] = None
) -> str:
    """
    List Kaggle competitions with optional filtering.

    Args:
        group: Competition group - Options: "general", "entered", "inClass" (default: "general")
        category: Competition category - Options: "all", "featured", "research", "recruitment", "gettingStarted", "masters", "playground" (default: "all")
        sort_by: Sort order - Options: "grouped", "prize", "earliestDeadline", "latestDeadline", "numberOfTeams", "recentlyCreated" (default: "latestDeadline")
        page: Page number for pagination (default: 1)
        search: Search term to filter competitions (optional)

    Returns:
        JSON string containing list of competitions with details (ref, description, deadline, category, reward, teamCount, etc.)
    """
    competitions = api.competitions_list(
        group=group,
        category=category,
        sort_by=sort_by,
        page=page,
        search=search
    )

    result = []
    for comp in competitions:
        result.append({
            "ref": comp.ref,
            "title": comp.title,
            "description": comp.description,
            "deadline": str(comp.deadline) if comp.deadline else None,
            "category": comp.category,
            "reward": comp.reward,
            "teamCount": comp.teamCount,
            "userHasEntered": comp.userHasEntered,
            "organizationName": comp.organizationName,
            "tags": comp.tags if hasattr(comp, 'tags') else []
        })

    return str(result)


@mcp.tool()
def competition_details(competition: str) -> str:
    """
    Get detailed information about a specific competition.

    Args:
        competition: Competition name/ID (e.g., "titanic")

    Returns:
        JSON string with competition details including description, rules, evaluation metrics, prizes, and timeline
    """
    try:
        comp_list = api.competition_view(competition)
        return str(comp_list)
    except Exception as e:
        return f"Error fetching competition details: {str(e)}"


@mcp.tool()
def competition_leaderboard(
    competition: str,
    page: int = 1
) -> str:
    """
    Get the leaderboard for a specific competition.

    Args:
        competition: Competition name/ID (e.g., "titanic")
        page: Page number for pagination (default: 1)

    Returns:
        JSON string with leaderboard entries (teamId, teamName, submissionDate, score)
    """
    try:
        leaderboard = api.competition_leaderboard_view(competition, page=page)
        return str(leaderboard)
    except Exception as e:
        return f"Error fetching leaderboard: {str(e)}"


@mcp.tool()
def download_competition_files(
    competition: str,
    file_name: Optional[str] = None,
    path: str = ".",
    force: bool = False,
    quiet: bool = True
) -> str:
    """
    Download competition dataset files.

    Args:
        competition: Competition name/ID (e.g., "titanic")
        file_name: Specific file to download (optional, downloads all if not specified)
        path: Directory path to download files to (default: current directory)
        force: Force download even if files exist (default: False)
        quiet: Suppress download progress output (default: True)

    Returns:
        Success message with download location or error message
    """
    try:
        api.competition_download_files(
            competition=competition,
            file_name=file_name,
            path=path,
            force=force,
            quiet=quiet
        )
        return f"Successfully downloaded competition files to {path}"
    except Exception as e:
        return f"Error downloading files: {str(e)}"


@mcp.tool()
def list_datasets(
    sort_by: str = "hottest",
    size: str = "all",
    file_type: str = "all",
    license_name: str = "all",
    tag_ids: str = "all",
    search: Optional[str] = None,
    user: Optional[str] = None,
    page: int = 1,
    max_size: int = 20
) -> str:
    """
    List Kaggle datasets with filtering options.

    Args:
        sort_by: Sort order - Options: "hottest", "votes", "updated", "active", "published" (default: "hottest")
        size: Dataset size - Options: "all", "small", "medium", "large" (default: "all")
        file_type: File type - Options: "all", "csv", "sqlite", "json", "bigQuery" (default: "all")
        license_name: License type - Options: "all", "cc", "gpl", "odb", "other" (default: "all")
        tag_ids: Tag IDs for filtering (comma-separated or "all")
        search: Search term to filter datasets (optional)
        user: Filter by specific user's datasets (optional)
        page: Page number for pagination (default: 1)
        max_size: Maximum number of results to return (default: 20)

    Returns:
        JSON string containing list of datasets with metadata (ref, title, size, downloadCount, voteCount, etc.)
    """
    datasets = api.dataset_list(
        sort_by=sort_by,
        size=size,
        file_type=file_type,
        license_name=license_name,
        tag_ids=tag_ids,
        search=search,
        user=user,
        page=page,
        max_size=max_size
    )

    result = []
    for ds in datasets:
        result.append({
            "ref": ds.ref,
            "title": ds.title,
            "size": ds.size,
            "lastUpdated": str(ds.lastUpdated) if ds.lastUpdated else None,
            "downloadCount": ds.downloadCount,
            "voteCount": ds.voteCount,
            "usabilityRating": ds.usabilityRating,
            "creatorName": ds.creatorName,
            "licenseName": ds.licenseName,
            "tags": ds.tags if hasattr(ds, 'tags') else []
        })

    return str(result)


@mcp.tool()
def dataset_details(dataset: str, owner: Optional[str] = None) -> str:
    """
    Get detailed information about a specific dataset.

    Args:
        dataset: Dataset name (e.g., "titanic")
        owner: Dataset owner username (optional, can be included in dataset as "owner/dataset")

    Returns:
        JSON string with dataset details including description, files, columns, and metadata
    """
    try:
        if owner:
            dataset_ref = f"{owner}/{dataset}"
        else:
            dataset_ref = dataset

        dataset_info = api.dataset_view(dataset_ref)
        return str(dataset_info)
    except Exception as e:
        return f"Error fetching dataset details: {str(e)}"


@mcp.tool()
def list_dataset_files(dataset: str, owner: Optional[str] = None) -> str:
    """
    List all files in a specific dataset.

    Args:
        dataset: Dataset name (e.g., "titanic")
        owner: Dataset owner username (optional, can be included in dataset as "owner/dataset")

    Returns:
        JSON string with list of files (name, size, creationDate)
    """
    try:
        if owner:
            dataset_ref = f"{owner}/{dataset}"
        else:
            dataset_ref = dataset

        files = api.dataset_list_files(dataset_ref)
        result = []
        for f in files.files:
            result.append({
                "name": f.name,
                "size": f.size,
                "creationDate": str(f.creationDate) if hasattr(f, 'creationDate') else None
            })
        return str(result)
    except Exception as e:
        return f"Error listing dataset files: {str(e)}"


@mcp.tool()
def download_dataset(
    dataset: str,
    owner: Optional[str] = None,
    file_name: Optional[str] = None,
    path: str = ".",
    force: bool = False,
    quiet: bool = True,
    unzip: bool = True
) -> str:
    """
    Download a Kaggle dataset.

    Args:
        dataset: Dataset name (e.g., "titanic")
        owner: Dataset owner username (optional, can be included in dataset as "owner/dataset")
        file_name: Specific file to download (optional, downloads all if not specified)
        path: Directory path to download files to (default: current directory)
        force: Force download even if files exist (default: False)
        quiet: Suppress download progress output (default: True)
        unzip: Unzip downloaded files (default: True)

    Returns:
        Success message with download location or error message
    """
    try:
        if owner:
            dataset_ref = f"{owner}/{dataset}"
        else:
            dataset_ref = dataset

        api.dataset_download_files(
            dataset=dataset_ref,
            file_name=file_name,
            path=path,
            force=force,
            quiet=quiet,
            unzip=unzip
        )
        return f"Successfully downloaded dataset to {path}"
    except Exception as e:
        return f"Error downloading dataset: {str(e)}"


@mcp.tool()
def list_kernels(
    page: int = 1,
    page_size: int = 20,
    dataset: Optional[str] = None,
    competition: Optional[str] = None,
    parent_kernel: Optional[str] = None,
    search: Optional[str] = None,
    mine: bool = False,
    user: Optional[str] = None,
    language: str = "all",
    kernel_type: str = "all",
    output_type: str = "all",
    sort_by: str = "hotness"
) -> str:
    """
    List Kaggle kernels (notebooks/scripts) with filtering.

    Args:
        page: Page number for pagination (default: 1)
        page_size: Number of results per page (default: 20)
        dataset: Filter by dataset (format: "owner/dataset-name")
        competition: Filter by competition name
        parent_kernel: Filter by parent kernel
        search: Search term to filter kernels
        mine: Show only your kernels (default: False)
        user: Filter by specific user's kernels
        language: Language filter - Options: "all", "python", "r", "sqlite", "julia" (default: "all")
        kernel_type: Type filter - Options: "all", "script", "notebook" (default: "all")
        output_type: Output filter - Options: "all", "visualization", "data" (default: "all")
        sort_by: Sort order - Options: "hotness", "commentCount", "dateCreated", "dateRun", "relevance", "scoreAscending", "scoreDescending", "viewCount", "voteCount" (default: "hotness")

    Returns:
        JSON string containing list of kernels with metadata
    """
    try:
        kernels = api.kernels_list(
            page=page,
            page_size=page_size,
            dataset=dataset,
            competition=competition,
            parent_kernel=parent_kernel,
            search=search,
            mine=mine,
            user=user,
            language=language,
            kernel_type=kernel_type,
            output_type=output_type,
            sort_by=sort_by
        )
        result = []
        for k in kernels:
            result.append({
                "ref": k.ref,
                "title": k.title,
                "author": k.author,
                "lastRunTime": str(k.lastRunTime) if hasattr(k, 'lastRunTime') and k.lastRunTime else None,
                "language": k.language if hasattr(k, 'language') else None,
                "kernelType": k.kernelType if hasattr(k, 'kernelType') else None,
                "totalVotes": k.totalVotes if hasattr(k, 'totalVotes') else 0,
                "totalViews": k.totalViews if hasattr(k, 'totalViews') else 0,
                "totalComments": k.totalComments if hasattr(k, 'totalComments') else 0
            })
        return str(result)
    except Exception as e:
        return f"Error listing kernels: {str(e)}"


@mcp.tool()
def download_kernel(
    kernel: str,
    path: str = ".",
    metadata: bool = False,
    quiet: bool = True
) -> str:
    """
    Download a Kaggle kernel (notebook/script).

    Args:
        kernel: Kernel reference (format: "owner/kernel-name")
        path: Directory path to download to (default: current directory)
        metadata: Also download kernel metadata (default: False)
        quiet: Suppress download progress output (default: True)

    Returns:
        Success message with download location or error message
    """
    try:
        api.kernels_pull(
            kernel=kernel,
            path=path,
            metadata=metadata,
            quiet=quiet
        )
        return f"Successfully downloaded kernel to {path}"
    except Exception as e:
        return f"Error downloading kernel: {str(e)}"


@mcp.tool()
def get_kernel_output(
    kernel: str,
    path: str = ".",
    force: bool = False,
    quiet: bool = True
) -> str:
    """
    Download the output files from a Kaggle kernel.

    Args:
        kernel: Kernel reference (format: "owner/kernel-name")
        path: Directory path to download output to (default: current directory)
        force: Force download even if files exist (default: False)
        quiet: Suppress download progress output (default: True)

    Returns:
        Success message with download location or error message
    """
    try:
        api.kernels_output(
            kernel=kernel,
            path=path,
            force=force,
            quiet=quiet
        )
        return f"Successfully downloaded kernel output to {path}"
    except Exception as e:
        return f"Error downloading kernel output: {str(e)}"


def _extract_proposed_model_lines(raw_output: str) -> List[str]:
    """
    Pull out the lines that appear under the `Proposed Model` heading.
    Falls back to the whole string if the heading is not present.
    """
    lines = raw_output.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if "proposed model" in line.lower():
            start_idx = idx + 1
            break

    if start_idx is None:
        return lines

    collected: List[str] = []
    for line in lines[start_idx:]:
        trimmed = line.strip()
        if not trimmed:
            # stop at the first empty line after the header
            break
        # stop if we hit another section header
        if re.match(r"^[A-Za-z][A-Za-z0-9 _-]*:$", trimmed):
            break
        collected.append(line)
    return collected


def _build_graph_from_atoms(lines: List[str]) -> Dict[str, object]:
    """
    Parse predicate atoms into a simple directed graph representation.
    Binary predicates become edges (src, dst, label=predicate).
    Unary predicates create isolated labeled nodes.
    """
    atom_pattern = re.compile(r"([A-Za-z_][\w]*)\(([^()]*)\)")
    nodes: Set[str] = set()
    edges: List[Dict[str, str]] = []

    for line in lines:
        for match in atom_pattern.finditer(line):
            predicate, arg_blob = match.groups()
            args = [arg.strip() for arg in arg_blob.split(",") if arg.strip()]
            if len(args) == 2:
                src, dst = args
                nodes.update([src, dst])
                edges.append({"source": src, "target": dst, "label": predicate})
            elif len(args) == 1:
                nodes.add(args[0])

    # Build a DOT graph for quick visualization
    dot_lines = ["digraph ProposedModel {", "  rankdir=LR;"]
    for node in sorted(nodes):
        dot_lines.append(f'  "{node}";')
    for edge in edges:
        label = edge.get("label", "")
        label_part = f' [label="{label}"]' if label else ""
        dot_lines.append(f'  "{edge["source"]}" -> "{edge["target"]}"{label_part};')
    dot_lines.append("}")

    return {
        "nodes": sorted(nodes),
        "edges": edges,
        "dot": "\n".join(dot_lines)
    }


@mcp.tool()
def parse_z3_proposed_model(model_output: str) -> str:
    """
    Parse a Z3 textual model output and return a graph-friendly view of the
    `Proposed Model` section. The result includes nodes, labeled edges, and a
    Graphviz DOT string to visualize the relationships.

    Args:
        model_output: Full Z3 solver output as text.

    Returns:
        JSON string with keys:
            - nodes: list of node names
            - edges: list of {source, target, label}
            - dot: Graphviz DOT description for quick rendering
    """
    lines = _extract_proposed_model_lines(model_output)
    graph = _build_graph_from_atoms(lines)
    return json.dumps(graph)


def _git_diff_output(branch_range: str, paths: Optional[List[str]], context_lines: int, stat_only: bool) -> str:
    """Run git diff with optional paths."""
    cmd = ["git", "diff", branch_range]
    if stat_only:
        cmd.append("--stat")
    else:
        cmd.extend(["--unified", str(max(0, context_lines))])
    if paths:
        cmd.extend(paths)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        return f"Error computing diff: {result.stderr.strip() or result.stdout.strip()}"
    return result.stdout.strip()


@mcp.tool()
def diff_against_main(
    target_branch: str = "main",
    paths: Optional[List[str]] = None,
    context_lines: int = 3,
    stat_only: bool = False,
    max_output_chars: int = 8000
) -> str:
    """
    Compare current branch changes against the target branch before evaluation.

    Args:
        target_branch: Branch to compare against (default: "main").
        paths: Optional list of paths to limit the diff.
        context_lines: Number of context lines to include when not using stat_only.
        stat_only: Return only a summary (files/insertions/deletions).
        max_output_chars: Truncate output to this many characters to avoid overload.

    Returns:
        Diff output or error message.
    """
    try:
        branch_range = f"{target_branch}...HEAD"
        output = _git_diff_output(branch_range, paths, context_lines, stat_only)
        if not output:
            return "No differences found between branches."
        if len(output) > max_output_chars:
            return f"{output[:max_output_chars]}\n\n[output truncated at {max_output_chars} characters]"
        return output
    except FileNotFoundError:
        return "git is not available on this system."
    except Exception as e:
        return f"Unexpected error running diff: {str(e)}"


# Import reverse simulation system
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fol_workbench.reverse_simulation_system import ReverseSimulationSystem
    from fol_workbench.test_first_simulator import TestCase, TestStatus
    REVERSE_SIMULATION_AVAILABLE = True
except ImportError:
    REVERSE_SIMULATION_AVAILABLE = False
    ReverseSimulationSystem = None

# Global reverse simulation system instance
_reverse_sim_system = None

def _get_reverse_sim_system():
    """Get or create reverse simulation system instance."""
    global _reverse_sim_system
    if _reverse_sim_system is None and REVERSE_SIMULATION_AVAILABLE:
        _reverse_sim_system = ReverseSimulationSystem()
    return _reverse_sim_system


@mcp.tool()
def create_test_first_model(
    name: str,
    formula: str,
    variables: Dict[str, str],
    test_cases: List[Dict[str, Any]]
) -> str:
    """
    Create a test-first unit model with test cases.
    
    Args:
        name: Model name
        formula: FOL formula (e.g., "And(x, y)")
        variables: Dict mapping variable names to types (e.g., {"x": "Bool", "y": "Bool"})
        test_cases: List of test case dicts with "name", "input", "expected_output", "constraints"
    
    Returns:
        Success message with model details
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available. Install required dependencies."
    
    try:
        system = _get_reverse_sim_system()
        model = system.create_model_with_tests(name, formula, variables, test_cases)
        
        return json.dumps({
            "status": "success",
            "model_name": model.name,
            "test_cases": len(model.test_cases),
            "message": f"Created model '{name}' with {len(model.test_cases)} test cases"
        }, indent=2)
    except Exception as e:
        return f"Error creating model: {str(e)}"


@mcp.tool()
def analyze_model_design(model_name: str) -> str:
    """
    Analyze a model's design and get intelligent feedback.
    
    Args:
        model_name: Name of the model to analyze
    
    Returns:
        JSON with feedback, topics, and proposals
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available."
    
    try:
        system = _get_reverse_sim_system()
        result = system.analyze_and_get_feedback(model_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing model: {str(e)}"


@mcp.tool()
def test_hypothesis(
    hypothesis_description: str,
    model_name: str,
    user_response: Optional[str] = None
) -> str:
    """
    Test a hypothesis with the "getting warmer" feedback loop.
    
    Args:
        hypothesis_description: Description of the hypothesis to test
        model_name: Name of the model to test against
        user_response: Optional user response ("yes"/"no") - if not provided, will prompt
    
    Returns:
        JSON with hypothesis test results
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available."
    
    try:
        system = _get_reverse_sim_system()
        
        # Create callback if user response provided
        callback = None
        if user_response:
            def fixed_callback(msg):
                return user_response
            callback = fixed_callback
        
        result = system.test_hypothesis_with_feedback(
            hypothesis_description,
            model_name,
            callback
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error testing hypothesis: {str(e)}"


@mcp.tool()
def generate_reverse_simulation_notebook(
    model_name: str,
    observed_outputs: List[Dict[str, Any]],
    output_path: str = "reverse_simulation.ipynb"
) -> str:
    """
    Generate a Kaggle notebook that simulates outputs and guesses inputs.
    
    Args:
        model_name: Name of the model to reverse engineer
        observed_outputs: List of observed output patterns (dicts with variable: value)
        output_path: Path to save the notebook
    
    Returns:
        Success message with notebook path
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available."
    
    try:
        system = _get_reverse_sim_system()
        path = system.generate_reverse_simulation_notebook(
            model_name,
            observed_outputs,
            Path(output_path)
        )
        return f"Successfully generated notebook at {path}"
    except Exception as e:
        return f"Error generating notebook: {str(e)}"


@mcp.tool()
def propose_next_steps(model_name: str) -> str:
    """
    Propose next steps for model development based on current state.
    
    Args:
        model_name: Name of the model
    
    Returns:
        JSON with prioritized next steps
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available."
    
    try:
        system = _get_reverse_sim_system()
        steps = system.propose_next_steps(model_name)
        return json.dumps(steps, indent=2)
    except Exception as e:
        return f"Error proposing next steps: {str(e)}"


@mcp.tool()
def run_hypothesis_loop(
    hypothesis_description: str,
    model_name: str,
    max_steps: int = 10,
    user_responses: Optional[List[str]] = None
) -> str:
    """
    Run the complete hypothesis testing loop with backtracking.
    
    Args:
        hypothesis_description: Initial hypothesis description
        model_name: Name of the model
        max_steps: Maximum number of steps
        user_responses: Optional list of user responses for each step
    
    Returns:
        JSON with final hypothesis and steps
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available."
    
    try:
        system = _get_reverse_sim_system()
        
        # Create callback with responses
        response_iter = iter(user_responses) if user_responses else None
        
        def callback(msg):
            if response_iter:
                try:
                    return next(response_iter)
                except StopIteration:
                    return "no"  # Default to no if responses exhausted
            return "yes"  # Default to yes for auto-mode
        
        result = system.run_hypothesis_loop(
            hypothesis_description,
            model_name,
            max_steps,
            callback
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error running hypothesis loop: {str(e)}"


@mcp.tool()
def collect_client_feedback(
    feedback_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Collect feedback from client and publish to server.
    
    Args:
        feedback_type: Type of feedback (e.g., "formula", "hypothesis", "design", "performance")
        content: Feedback content
        metadata: Optional metadata dictionary
    
    Returns:
        Success message with feedback ID
    """
    try:
        import uuid
        from datetime import datetime
        
        feedback_id = str(uuid.uuid4())
        feedback_data = {
            "id": feedback_id,
            "type": feedback_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "collected"
        }
        
        # In a real implementation, this would publish to a server/API
        # For now, we'll store it locally or return the structured data
        return json.dumps({
            "status": "success",
            "feedback_id": feedback_id,
            "message": f"Feedback collected and published: {feedback_type}",
            "data": feedback_data
        }, indent=2)
    except Exception as e:
        return f"Error collecting feedback: {str(e)}"


@mcp.tool()
def extract_vocabulary_metadata() -> str:
    """
    Extract metadata from vocabulary terms and evaluate Herbrand base implications.
    
    Returns:
        JSON with vocabulary metadata and Herbrand base evaluation results
    """
    try:
        from fol_workbench.logic_layer import LogicEngine
        
        engine = LogicEngine()
        
        # Extract vocabulary metadata
        metadata = engine.extract_vocabulary_metadata()
        
        # Evaluate Herbrand base implications
        implications_result = engine.evaluate_herbrand_implications()
        
        return json.dumps({
            "vocabulary_metadata": metadata,
            "herbrand_evaluation": implications_result
        }, indent=2)
    except Exception as e:
        return f"Error extracting vocabulary metadata: {str(e)}"


@mcp.tool()
def pnf_and_skolemize(formula: str, timeout_ms: int = 2000) -> str:
    """
    Perform prenexing (PNF) and Skolemization (SNF) on a first-order logic formula.

    Returns:
      - PNF: logically equivalent transformation (α-renaming only)
      - SNF: Skolem normal form (equisatisfiable, not generally equivalent)

    Args:
        formula: Formula in Python/Z3 syntax (e.g., "ForAll(x, Exists(y, Iff(x, y)))")
                 or SMT-LIB if the string starts with '('.
        timeout_ms: Timeout for the optional equivalence check.

    Returns:
        JSON string with original, pnf_smt2, skolem_snf_smt2, equivalence_check, and a
        short evidence/performance-oriented "scientific philosophy" note.
    """
    try:
        from fol_workbench.logic_layer import LogicEngine

        engine = LogicEngine()
        result = engine.pnf_and_skolemize(formula, timeout_ms=timeout_ms)

        philosophy = {
            "scientific_philosophy": (
                "Treat each logical formula as a falsifiable hypothesis. Use solver-produced countermodels "
                "as empirical refutations and solver statistics (timeouts/unknowns, decisions, conflicts) as "
                "performance evidence to guide theory choice. Prefer theories that are (a) predictive "
                "(satisfiable/unsatisfiable results are stable under re-checks), (b) parsimonious "
                "(low quantifier depth / minimal Skolem growth), and (c) modular (test-first constraints)."
            ),
            "theory_note": (
                "PNF preserves logical equivalence; Skolemization preserves satisfiability (equisatisfiable). "
                "In practice, Skolem growth is a measurable complexity cost, so track it as a performance signal."
            ),
        }

        if isinstance(result, dict) and "error" in result:
            return json.dumps({"status": "error", "error": result["error"]})

        return json.dumps(
            {
                "status": "success",
                **result,
                **philosophy,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def call_reverse_simulation_with_herbrand(
    model_name: str,
    preferred_implications: Optional[List[str]] = None
) -> str:
    """
    Call reverse simulation system with Herbrand base evaluation.
    
    Args:
        model_name: Name of the model to analyze
        preferred_implications: Optional list of implication formulas to evaluate
    
    Returns:
        JSON with reverse simulation results and Herbrand evaluation
    """
    if not REVERSE_SIMULATION_AVAILABLE:
        return "Reverse simulation system not available."
    
    try:
        system = _get_reverse_sim_system()
        result = system.call_with_herbrand_evaluation(
            model_name,
            preferred_implications
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error calling reverse simulation: {str(e)}"


@mcp.tool()
def create_gpg_signature(
    file_path: str,
    key_id: Optional[str] = None,
    passphrase: Optional[str] = None,
    output_path: Optional[str] = None,
    detach: bool = True,
    armor: bool = True
) -> str:
    """
    Create a GPG signature for a file.

    Args:
        file_path: Path to the file to sign
        key_id: ID of the key to sign with (optional, uses default if not specified)
        passphrase: Optional passphrase for the GPG key
        output_path: Optional path for the signature file
        detach: Create a detached signature (default: True)
        armor: Create ASCII armored output (default: True)

    Returns:
        Success message with path to signature file
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        cmd = ["gpg", "--batch", "--yes"]
        
        if key_id:
            cmd.extend(["--local-user", key_id])
        
        if passphrase:
            cmd.extend(["--passphrase-fd", "0", "--pinentry-mode", "loopback"])
            
        if armor:
            cmd.append("--armor")
            
        if detach:
            cmd.append("--detach-sign")
        else:
            cmd.append("--sign")
            
        if output_path:
            cmd.extend(["--output", output_path])
            
        cmd.append(file_path)
        
        result = subprocess.run(
            cmd,
            input=passphrase if passphrase else None,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return f"Error creating signature: {result.stderr}"
            
        return f"Successfully created signature for {file_path}"
        
    except Exception as e:
        return f"Error executing GPG: {str(e)}"


@mcp.tool()
def export_gpg_public_key(
    key_id: Optional[str] = None,
    output_path: Optional[str] = None,
    armor: bool = True,
    strict_openpgp: bool = False
) -> str:
    """
    Export a GPG public key.

    Args:
        key_id: ID of the key to export (optional, exports all if not specified)
        output_path: Optional path for the exported key file
        armor: Create ASCII armored output (default: True)
        strict_openpgp: Use strict OpenPGP behavior (default: False)

    Returns:
        Success message with path to exported key or key content
    """
    try:
        cmd = ["gpg", "--batch", "--yes"]
        
        if strict_openpgp:
            cmd.append("--openpgp")
            
        if armor:
            cmd.append("--armor")
            
        cmd.append("--export")
        
        if output_path:
            cmd.extend(["--output", output_path])
            
        if key_id:
            cmd.append(key_id)
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return f"Error exporting key: {result.stderr}"
            
        if output_path:
            return f"Successfully exported public key to {output_path}"
        else:
            return result.stdout
            
    except Exception as e:
        return f"Error executing GPG: {str(e)}"


def main():
    """Run the Kaggle MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
