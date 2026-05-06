"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml


def load_skills_yaml(yaml_path: Optional[str] = None) -> dict:
    """Load the skills.yaml source of truth."""
    path = yaml_path or os.path.join(os.path.dirname(__file__), "..", "skills.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def generate_openclaw_skill(skills_config: dict, output_dir: Optional[str] = None) -> str:
    """Generate OpenClaw SKILL.md."""
    output = Path(output_dir or os.path.expanduser("~/.openclaw/workspace/skills/mlsyseng"))
    output.mkdir(parents=True, exist_ok=True)

    skill_md = _render_skill_md(skills_config, platform="openclaw")
    filepath = output / "SKILL.md"
    filepath.write_text(skill_md)
    return str(filepath)


def generate_cursor_skill(skills_config: dict, output_dir: Optional[str] = None) -> str:
    """Generate Cursor SKILL.md."""
    output = Path(output_dir or os.path.expanduser("~/.cursor/skills/mlsyseng"))
    output.mkdir(parents=True, exist_ok=True)

    skill_md = _render_skill_md(skills_config, platform="cursor")
    filepath = output / "SKILL.md"
    filepath.write_text(skill_md)
    return str(filepath)


def generate_generic_skill(skills_config: dict, output_dir: Optional[str] = None) -> str:
    """Generate generic SKILL.md."""
    output = Path(output_dir or os.path.expanduser("~/.skills/mlsyseng"))
    output.mkdir(parents=True, exist_ok=True)

    skill_md = _render_skill_md(skills_config, platform="generic")
    filepath = output / "SKILL.md"
    filepath.write_text(skill_md)
    return str(filepath)


def generate_claude_desktop_config(skills_config: dict) -> str:
    """Update Claude Desktop MCP configuration."""
    config_path = Path(os.path.expanduser(
        "~/Library/Application Support/Claude/claude_desktop_config.json"
    ))

    existing_config = {}
    if config_path.exists():
        with open(config_path) as f:
            existing_config = json.load(f)

    mcp_servers = existing_config.get("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": "python",
        "args": ["-m", "mlsyseng_moe.server"],
        "env": skills_config.get("env", {}),
    }
    existing_config["mcpServers"] = mcp_servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing_config, f, indent=2)

    return str(config_path)


def generate_gemini_config(skills_config: dict) -> str:
    """Update Gemini MCP configuration."""
    config_path = Path(os.path.expanduser("~/.gemini/mcp_config.json"))

    existing_config = {}
    if config_path.exists():
        with open(config_path) as f:
            existing_config = json.load(f)

    mcp_servers = existing_config.get("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": "python",
        "args": ["-m", "mlsyseng_moe.server"],
        "env": skills_config.get("env", {}),
    }
    existing_config["mcpServers"] = mcp_servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing_config, f, indent=2)

    return str(config_path)


def generate_openclaw_config(skills_config: dict) -> str:
    """Update OpenClaw configuration with MCP server."""
    config_path = Path(os.path.expanduser("~/.openclaw/openclaw.json"))

    existing_config = {}
    if config_path.exists():
        with open(config_path) as f:
            existing_config = json.load(f)

    mcp_servers = existing_config.get("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": "python",
        "args": ["-m", "mlsyseng_moe.server"],
        "env": skills_config.get("env", {}),
    }
    existing_config["mcpServers"] = mcp_servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing_config, f, indent=2)

    return str(config_path)


def _render_skill_md(skills_config: dict, platform: str) -> str:
    """Render a SKILL.md file from the skills configuration."""
    name = skills_config.get("name", "MLSysEng MoE")
    description = skills_config.get("description", "ML Systems Expert Mixture of Experts")
    tools = skills_config.get("tools", [])
    experts = skills_config.get("experts", {})

    lines = [
        f"# {name}",
        "",
        f"{description}",
        "",
        "## Available Tools",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]

    for tool in tools:
        lines.append(f"| `{tool['name']}` | {tool['description']} |")

    lines.extend([
        "",
        "## Usage",
        "",
        "### Extract Knowledge",
        "```",
        "extract-knowledge(force_reindex=false)",
        "```",
        "",
        "### Search Concepts",
        "```",
        'search-concepts(query="neural network optimization")',
        "```",
        "",
        "### Build Competition Entry",
        "```",
        'build-entry(competition="titanic")',
        "```",
        "",
        "### Run Convergence Loop",
        "```",
        'evolve(competition="titanic")',
        "```",
        "",
        "## Expert System",
        "",
        "Each chapter expert has:",
        "- **Capabilities**: What the expert can do",
        "- **Skills**: Kaggle skills the expert recommends",
        "- **Strategy**: Step-by-step approach (Baseline → EDA → Feature Eng → Model → Submit)",
        "- **Formula**: Mathematical objective function",
        "- **Loop Config**: Convergence parameters (epsilon, patience, max_iterations)",
        "",
        "## Convergence Loop",
        "",
        "Exit condition: `||state[n] - state[n-1]||_2 < epsilon`",
        "",
        f"Platform: {platform}",
        "",
    ])

    return "\n".join(lines)


def generate_all(yaml_path: Optional[str] = None) -> dict:
    """Generate skill configurations for all platforms."""
    config = load_skills_yaml(yaml_path)
    results = {}

    try:
        results["openclaw_skill"] = generate_openclaw_skill(config)
    except Exception as e:
        results["openclaw_skill"] = f"error: {e}"

    try:
        results["cursor_skill"] = generate_cursor_skill(config)
    except Exception as e:
        results["cursor_skill"] = f"error: {e}"

    try:
        results["generic_skill"] = generate_generic_skill(config)
    except Exception as e:
        results["generic_skill"] = f"error: {e}"

    try:
        results["claude_desktop"] = generate_claude_desktop_config(config)
    except Exception as e:
        results["claude_desktop"] = f"error: {e}"

    try:
        results["gemini"] = generate_gemini_config(config)
    except Exception as e:
        results["gemini"] = f"error: {e}"

    try:
        results["openclaw_config"] = generate_openclaw_config(config)
    except Exception as e:
        results["openclaw_config"] = f"error: {e}"

    return results


if __name__ == "__main__":
    results = generate_all()
    print("Generated skill configurations:")
    for platform, path in results.items():
        print(f"  {platform}: {path}")
