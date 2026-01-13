"""Kaggle MCP Server implementation."""

import json
import os
import re
import subprocess
from typing import Optional, List, Dict, Set
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


def main():
    """Run the Kaggle MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
