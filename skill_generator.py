#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the canonical skills.yaml and installs SKILL.md and MCP config
for each supported platform.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


def load_skills_config(path: str = "skills.yaml") -> Dict[str, Any]:
    """Load skills.yaml configuration."""
    with open(path) as f:
        if yaml:
            return yaml.safe_load(f)
        content = f.read()
        return _parse_yaml_subset(content)


def _parse_yaml_subset(content: str) -> Dict[str, Any]:
    """Minimal YAML-like parser for when PyYAML is not available."""
    import re
    config = {
        "name": "mlsyseng-moe",
        "version": "0.1.0",
        "description": "MLSysEng MoE Expert System",
        "mcp_server": {
            "name": "mlsyseng-moe",
            "command": "python",
            "args": ["-m", "src.mlsyseng_mcp.server"],
        },
        "skill": {
            "name": "MLSysEng MoE Expert System",
            "instruction": "ML Systems Engineering expert powered by chapter knowledge.",
        },
    }
    return config


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate SKILL.md content from config."""
    skill = config.get("skill", {})
    tools = config.get("tools", [])

    tool_table = "| Tool | Description |\n|------|-------------|\n"
    for t in tools:
        tool_table += f"| `{t['name']}` | {t['description']} |\n"

    return f"""# {skill.get('name', 'MLSysEng MoE Expert System')}

{skill.get('instruction', '')}

## Available Tools

{tool_table}

## Quick Start

1. Extract knowledge: `extract-knowledge(force_reindex=false)`
2. List experts: `list-experts()`
3. Build entry: `build-entry(competition="titanic")`
4. Run convergence: `evolve(competition="titanic")`
5. Search concepts: `search-concepts(query="neural network optimization")`

## Expert Definition

Each expert has:
- **Capabilities**: What the expert can help with
- **Skills**: Kaggle skill paths for the competition pipeline
- **Strategy**: Step-by-step approach (e.g., Baseline → Feature Eng → Model → Submit)
- **Formula**: Objective function (e.g., minimize_validation_loss: L = f(X, θ, α))
- **Loop Config**: Convergence settings (epsilon, max_iterations, patience)

## Convergence Loop

The system uses L2 norm convergence:
- Exit condition: `||state[n] - state[n-1]||_2 < epsilon`
- Default epsilon: 0.001
- Default patience: 3 consecutive converging iterations
"""


def generate_mcp_config_entry(config: Dict[str, Any], workspace_path: str = "") -> Dict[str, Any]:
    """Generate MCP server config entry."""
    server = config.get("mcp_server", {})
    return {
        "command": server.get("command", "python"),
        "args": server.get("args", ["-m", "src.mlsyseng_mcp.server"]),
        "env": {
            "PYTHONPATH": workspace_path or ".",
        },
    }


def install_openclaw(config: Dict[str, Any], workspace_path: str):
    """Install skill for OpenClaw."""
    platforms = config.get("platforms", {})
    oc = platforms.get("openclaw", {})

    skill_dir = Path(os.path.expanduser(oc.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng")))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  Installed SKILL.md to {skill_dir}")

    config_path = Path(os.path.expanduser(oc.get("config_path", "~/.openclaw/openclaw.json")))
    if config_path.exists():
        try:
            oc_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            oc_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = generate_mcp_config_entry(config, workspace_path)
    config_path.write_text(json.dumps(oc_config, indent=2) + "\n")
    print(f"  Updated MCP config in {config_path}")


def install_claude_desktop(config: Dict[str, Any], workspace_path: str):
    """Install MCP config for Claude Desktop."""
    platforms = config.get("platforms", {})
    cd = platforms.get("claude_desktop", {})

    config_path = Path(os.path.expanduser(
        cd.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")
    ))

    if config_path.exists():
        try:
            cd_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            cd_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = generate_mcp_config_entry(config, workspace_path)
    config_path.write_text(json.dumps(cd_config, indent=2) + "\n")
    print(f"  Updated Claude Desktop config in {config_path}")


def install_cursor(config: Dict[str, Any], workspace_path: str):
    """Install skill for Cursor."""
    platforms = config.get("platforms", {})
    cur = platforms.get("cursor", {})

    skill_dir = Path(os.path.expanduser(cur.get("skill_path", "~/.cursor/skills/mlsyseng")))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  Installed SKILL.md to {skill_dir}")


def install_gemini(config: Dict[str, Any], workspace_path: str):
    """Install MCP config for Gemini."""
    platforms = config.get("platforms", {})
    gem = platforms.get("gemini", {})

    config_path = Path(os.path.expanduser(gem.get("config_path", "~/.gemini/mcp_config.json")))

    if config_path.exists():
        try:
            gem_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            gem_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gem_config = {}

    mcp_servers = gem_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = generate_mcp_config_entry(config, workspace_path)
    config_path.write_text(json.dumps(gem_config, indent=2) + "\n")
    print(f"  Updated Gemini config in {config_path}")


def install_generic(config: Dict[str, Any], workspace_path: str):
    """Install generic skill."""
    platforms = config.get("platforms", {})
    gen = platforms.get("generic", {})

    skill_dir = Path(os.path.expanduser(gen.get("skill_path", "~/.skills/mlsyseng")))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  Installed SKILL.md to {skill_dir}")


def main():
    """Generate and install skills for all platforms."""
    script_dir = Path(__file__).parent
    config_path = script_dir / "skills.yaml"

    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    config = load_skills_config(str(config_path))
    workspace_path = str(script_dir.resolve())

    print(f"MLSysEng MoE Skill Generator v{config.get('version', '0.1.0')}")
    print(f"Workspace: {workspace_path}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        print(f"Installing for {name}...")
        try:
            installer(config, workspace_path)
        except Exception as e:
            print(f"  Warning: Could not install for {name}: {e}")
        print()

    print("Done! Restart AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
