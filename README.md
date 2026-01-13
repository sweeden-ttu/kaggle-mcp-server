# Kaggle MCP Server

A Model Context Protocol (MCP) server that provides comprehensive access to the Kaggle API. This server enables AI assistants to search competitions, browse datasets, download files, and interact with Kaggle kernels.

## Features

This MCP server provides 14 tools for interacting with Kaggle:

### Competition Tools
- **list_competitions** - Browse and search Kaggle competitions
- **competition_details** - Get detailed information about a specific competition
- **competition_leaderboard** - View competition leaderboards
- **download_competition_files** - Download competition datasets

### Dataset Tools
- **list_datasets** - Search and browse Kaggle datasets
- **dataset_details** - Get detailed information about a dataset
- **list_dataset_files** - List all files in a dataset
- **download_dataset** - Download dataset files

### Kernel Tools
- **list_kernels** - Browse and search Kaggle kernels (notebooks/scripts)
- **download_kernel** - Download kernel source code
- **get_kernel_output** - Download kernel output files

### Utility
- **diff_against_main** - Compare the current branch to the main branch before running evaluations

## Installation

### Prerequisites

1. **Python 3.10+** is required
2. **Kaggle API credentials** - You need a Kaggle account and API token

#### Setting up Kaggle API Credentials

1. Go to your Kaggle account settings: https://www.kaggle.com/account
2. Scroll to "API" section and click "Create New Token"
3. This downloads `kaggle.json` with your credentials
4. Place the file at:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
5. Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

### Install the Server

#### Using uv (recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd kaggle-mcp-server

# Install dependencies
uv pip install -e .
```

#### Using pip

```bash
pip install -e .
```

## Usage

### Running with Claude Desktop

Add this to your Claude Desktop configuration file:

**MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "kaggle": {
      "command": "python",
      "args": ["-m", "kaggle_mcp_server"]
    }
  }
}
```

### Running Standalone

```bash
python -m kaggle_mcp_server
```

## Available Tools & Parameters

### list_competitions

Search and browse Kaggle competitions.

**Parameters:**
- `group` (string, default: "general") - Competition group
  - Options: "general", "entered", "inClass"
- `category` (string, default: "all") - Competition category
  - Options: "all", "featured", "research", "recruitment", "gettingStarted", "masters", "playground"
- `sort_by` (string, default: "latestDeadline") - Sort order
  - Options: "grouped", "prize", "earliestDeadline", "latestDeadline", "numberOfTeams", "recentlyCreated"
- `page` (integer, default: 1) - Page number for pagination
- `search` (string, optional) - Search term to filter competitions

**Returns:** List of competitions with ref, title, description, deadline, category, reward, teamCount, etc.

**Example:**
```python
# Search for machine learning competitions
list_competitions(category="featured", search="machine learning")
```

---

### competition_details

Get detailed information about a specific competition.

**Parameters:**
- `competition` (string, required) - Competition name/ID (e.g., "titanic")

**Returns:** Detailed competition information including rules, evaluation metrics, prizes, and timeline

**Example:**
```python
competition_details(competition="titanic")
```

---

### competition_leaderboard

View the leaderboard for a specific competition.

**Parameters:**
- `competition` (string, required) - Competition name/ID
- `page` (integer, default: 1) - Page number for pagination

**Returns:** Leaderboard entries with teamId, teamName, submissionDate, and score

---

### download_competition_files

Download competition dataset files.

**Parameters:**
- `competition` (string, required) - Competition name/ID
- `file_name` (string, optional) - Specific file to download (downloads all if not specified)
- `path` (string, default: ".") - Directory path to download files to
- `force` (boolean, default: false) - Force download even if files exist
- `quiet` (boolean, default: true) - Suppress download progress output

**Returns:** Success message with download location

**Example:**
```python
download_competition_files(
    competition="titanic",
    path="./data/titanic"
)
```

---

### list_datasets

Search and browse Kaggle datasets.

**Parameters:**
- `sort_by` (string, default: "hottest") - Sort order
  - Options: "hottest", "votes", "updated", "active", "published"
- `size` (string, default: "all") - Dataset size filter
  - Options: "all", "small", "medium", "large"
- `file_type` (string, default: "all") - File type filter
  - Options: "all", "csv", "sqlite", "json", "bigQuery"
- `license_name` (string, default: "all") - License filter
  - Options: "all", "cc", "gpl", "odb", "other"
- `tag_ids` (string, default: "all") - Tag IDs (comma-separated or "all")
- `search` (string, optional) - Search term to filter datasets
- `user` (string, optional) - Filter by specific user's datasets
- `page` (integer, default: 1) - Page number for pagination
- `max_size` (integer, default: 20) - Maximum number of results to return

**Returns:** List of datasets with ref, title, size, downloadCount, voteCount, usabilityRating, etc.

**Example:**
```python
list_datasets(
    search="covid-19",
    file_type="csv",
    sort_by="votes"
)
```

---

### dataset_details

Get detailed information about a specific dataset.

**Parameters:**
- `dataset` (string, required) - Dataset name (e.g., "titanic")
- `owner` (string, optional) - Dataset owner username (can be included in dataset as "owner/dataset")

**Returns:** Dataset details including description, files, columns, and metadata

**Example:**
```python
dataset_details(dataset="playground-series-s4e5")
# or
dataset_details(dataset="titanic", owner="niyamatalmass")
```

---

### list_dataset_files

List all files in a specific dataset.

**Parameters:**
- `dataset` (string, required) - Dataset name
- `owner` (string, optional) - Dataset owner username

**Returns:** List of files with name, size, and creationDate

---

### download_dataset

Download a Kaggle dataset.

**Parameters:**
- `dataset` (string, required) - Dataset name
- `owner` (string, optional) - Dataset owner username
- `file_name` (string, optional) - Specific file to download (downloads all if not specified)
- `path` (string, default: ".") - Directory path to download files to
- `force` (boolean, default: false) - Force download even if files exist
- `quiet` (boolean, default: true) - Suppress download progress output
- `unzip` (boolean, default: true) - Unzip downloaded files

**Returns:** Success message with download location

**Example:**
```python
download_dataset(
    dataset="playground-series-s4e5",
    path="./data",
    unzip=True
)
```

---

### list_kernels

Browse and search Kaggle kernels (notebooks/scripts).

**Parameters:**
- `page` (integer, default: 1) - Page number for pagination
- `page_size` (integer, default: 20) - Number of results per page
- `dataset` (string, optional) - Filter by dataset (format: "owner/dataset-name")
- `competition` (string, optional) - Filter by competition name
- `parent_kernel` (string, optional) - Filter by parent kernel
- `search` (string, optional) - Search term to filter kernels
- `mine` (boolean, default: false) - Show only your kernels
- `user` (string, optional) - Filter by specific user's kernels
- `language` (string, default: "all") - Language filter
  - Options: "all", "python", "r", "sqlite", "julia"
- `kernel_type` (string, default: "all") - Type filter
  - Options: "all", "script", "notebook"
- `output_type` (string, default: "all") - Output filter
  - Options: "all", "visualization", "data"
- `sort_by` (string, default: "hotness") - Sort order
  - Options: "hotness", "commentCount", "dateCreated", "dateRun", "relevance", "scoreAscending", "scoreDescending", "viewCount", "voteCount"

**Returns:** List of kernels with ref, title, author, language, votes, views, etc.

**Example:**
```python
list_kernels(
    competition="titanic",
    language="python",
    kernel_type="notebook",
    sort_by="voteCount"
)
```

---

### download_kernel

Download a Kaggle kernel (notebook/script) source code.

**Parameters:**
- `kernel` (string, required) - Kernel reference (format: "owner/kernel-name")
- `path` (string, default: ".") - Directory path to download to
- `metadata` (boolean, default: false) - Also download kernel metadata
- `quiet` (boolean, default: true) - Suppress download progress output

**Returns:** Success message with download location

**Example:**
```python
download_kernel(
    kernel="abhishek/approaching-almost-any-ml-problem",
    path="./kernels"
)
```

---

### get_kernel_output

Download the output files from a Kaggle kernel.

**Parameters:**
- `kernel` (string, required) - Kernel reference (format: "owner/kernel-name")
- `path` (string, default: ".") - Directory path to download output to
- `force` (boolean, default: false) - Force download even if files exist
- `quiet` (boolean, default: true) - Suppress download progress output

**Returns:** Success message with download location

---

### diff_against_main

Compare the current branch with a target branch (default: `main`) to see what changed before running evaluations.

**Parameters:**
- `target_branch` (string, default: "main") - Branch to compare against
- `paths` (list[string], optional) - Limit the diff to specific paths
- `context_lines` (integer, default: 3) - Number of context lines in the diff (ignored if `stat_only` is true)
- `stat_only` (boolean, default: false) - If true, return only a summary of file changes
- `max_output_chars` (integer, default: 8000) - Truncate output to this many characters to keep responses manageable

**Returns:** Diff output or an informative message if nothing changed

## Data Types Reference

### Competition Object
```python
{
    "ref": str,              # Competition reference/slug
    "title": str,            # Competition title
    "description": str,      # Short description
    "deadline": str,         # Deadline date (ISO format)
    "category": str,         # Competition category
    "reward": str,           # Prize amount/type
    "teamCount": int,        # Number of teams
    "userHasEntered": bool,  # Whether you've entered
    "organizationName": str, # Hosting organization
    "tags": list[str]        # Associated tags
}
```

### Dataset Object
```python
{
    "ref": str,              # Dataset reference (owner/name)
    "title": str,            # Dataset title
    "size": int,             # Size in bytes
    "lastUpdated": str,      # Last update date (ISO format)
    "downloadCount": int,    # Number of downloads
    "voteCount": int,        # Number of upvotes
    "usabilityRating": float,# Usability score (0-10)
    "creatorName": str,      # Dataset creator
    "licenseName": str,      # License type
    "tags": list[str]        # Associated tags
}
```

### Kernel Object
```python
{
    "ref": str,              # Kernel reference (owner/name)
    "title": str,            # Kernel title
    "author": str,           # Author username
    "lastRunTime": str,      # Last run date (ISO format)
    "language": str,         # Programming language
    "kernelType": str,       # "script" or "notebook"
    "totalVotes": int,       # Number of upvotes
    "totalViews": int,       # View count
    "totalComments": int     # Comment count
}
```

### File Object
```python
{
    "name": str,             # File name
    "size": int,             # Size in bytes
    "creationDate": str      # Creation date (ISO format)
}
```

## Example Workflows

### 1. Finding and Downloading a Dataset

```python
# Search for datasets
list_datasets(search="house prices", file_type="csv")

# Get details about a specific dataset
dataset_details(dataset="house-prices-advanced-regression-techniques")

# List files in the dataset
list_dataset_files(dataset="house-prices-advanced-regression-techniques")

# Download the dataset
download_dataset(
    dataset="house-prices-advanced-regression-techniques",
    path="./data/house-prices"
)
```

### 2. Exploring a Competition

```python
# Search competitions
list_competitions(category="featured", search="nlp")

# Get competition details
competition_details(competition="nlp-getting-started")

# View leaderboard
competition_leaderboard(competition="nlp-getting-started")

# Download competition data
download_competition_files(
    competition="nlp-getting-started",
    path="./competitions/nlp"
)
```

### 3. Finding Helpful Kernels

```python
# Search for kernels on a specific topic
list_kernels(
    search="gradient boosting tutorial",
    language="python",
    kernel_type="notebook",
    sort_by="voteCount"
)

# Download a kernel
download_kernel(
    kernel="alexisbcook/titanic-tutorial",
    path="./tutorials"
)
```

## Troubleshooting

### Authentication Error
If you see `OSError: Could not find kaggle.json`, ensure:
1. You've downloaded your API token from Kaggle
2. The file is in the correct location (`~/.kaggle/kaggle.json`)
3. File permissions are set correctly (Unix: `chmod 600`)

### Permission Denied
Some competitions/datasets require accepting terms:
1. Visit the competition/dataset page on Kaggle
2. Accept the terms and conditions
3. Try downloading again

### Rate Limiting
Kaggle API has rate limits. If you encounter errors, wait a few minutes before retrying.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
