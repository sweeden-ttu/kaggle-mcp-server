"""Generate platform-specific skill configs from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(config_path: str = "skills.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _moe_server_dir() -> str:
    return str(Path(__file__).parent.resolve())


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate a SKILL.md file content."""
    tools = config.get("tools", [])
    tool_table = "| Tool | Description |\n|------|-------------|\n"
    for t in tools:
        name = t["name"]
        alias = t.get("alias", "")
        desc = t.get("description", "")
        display = f"`{name}`" + (f" / `{alias}`" if alias else "")
        tool_table += f"| {display} | {desc} |\n"

    return f"""# {config['name']}

{config['description']}

## MCP Server

Start the server:

```bash
cd {_moe_server_dir()}
python server.py
```

## Available Tools

{tool_table}

## Quick Start

1. **Extract knowledge**: `extract-knowledge(force_reindex=false)`
2. **List experts**: `list-experts()`
3. **Build entry**: `build-entry(competition="titanic")`
4. **Run convergence**: `evolve(competition="titanic")`
5. **Search concepts**: `search-concepts(query="neural network optimization")`

## Expert System

Each chapter becomes an expert with:
- **Capabilities**: Inferred from chapter content
- **Skills**: Mapped to Kaggle skill paths
- **Strategy**: `{config.get('experts', {}).get('strategy_template', 'Baseline → Submit')}`
- **Convergence**: `||state[n] - state[n-1]||_2 < epsilon` with patience-based exit
"""


def install_openclaw(config: Dict[str, Any]) -> None:
    """Install skill for OpenClaw."""
    platform = config.get("platforms", {}).get("openclaw", {})
    skill_path = os.path.expanduser(platform.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/"))
    os.makedirs(skill_path, exist_ok=True)

    skill_file = os.path.join(skill_path, platform.get("skill_file", "SKILL.md"))
    with open(skill_file, "w") as f:
        f.write(generate_skill_md(config))
    print(f"  OpenClaw skill: {skill_file}")

    config_path = os.path.expanduser(platform.get("config_path", "~/.openclaw/openclaw.json"))
    _update_json_config(config_path, config, "openclaw")


def install_claude_desktop(config: Dict[str, Any]) -> None:
    """Install MCP config for Claude Desktop."""
    platform = config.get("platforms", {}).get("claude_desktop", {})
    config_path = os.path.expanduser(platform.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json"))

    _update_json_config(config_path, config, "claude_desktop")


def install_cursor(config: Dict[str, Any]) -> None:
    """Install skill for Cursor."""
    platform = config.get("platforms", {}).get("cursor", {})
    skill_path = os.path.expanduser(platform.get("skill_path", "~/.cursor/skills/mlsyseng/"))
    os.makedirs(skill_path, exist_ok=True)

    skill_file = os.path.join(skill_path, platform.get("skill_file", "SKILL.md"))
    with open(skill_file, "w") as f:
        f.write(generate_skill_md(config))
    print(f"  Cursor skill:   {skill_file}")


def install_gemini(config: Dict[str, Any]) -> None:
    """Install MCP config for Gemini."""
    platform = config.get("platforms", {}).get("gemini", {})
    config_path = os.path.expanduser(platform.get("config_path", "~/.gemini/mcp_config.json"))

    _update_json_config(config_path, config, "gemini")


def install_generic(config: Dict[str, Any]) -> None:
    """Install generic skill."""
    platform = config.get("platforms", {}).get("generic", {})
    skill_path = os.path.expanduser(platform.get("skill_path", "~/.skills/mlsyseng/"))
    os.makedirs(skill_path, exist_ok=True)

    skill_file = os.path.join(skill_path, platform.get("skill_file", "SKILL.md"))
    with open(skill_file, "w") as f:
        f.write(generate_skill_md(config))
    print(f"  Generic skill:  {skill_file}")


def _update_json_config(config_path: str, config: Dict[str, Any], platform: str) -> None:
    """Update a JSON config file with MCP server entry."""
    mcp_config = config.get("mcp_server", {})
    server_entry = {
        "command": mcp_config.get("command", "python"),
        "args": [os.path.join(_moe_server_dir(), a) for a in mcp_config.get("args", ["server.py"])],
        "env": mcp_config.get("env", {}),
    }

    existing: Dict[str, Any] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][config["name"]] = server_entry

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  {platform} config: {config_path}")


def generate_all(config_path: str = "skills.yaml") -> None:
    """Generate skills for all platforms."""
    config = load_skills_config(config_path)
    print(f"Generating skills for: {config['name']} v{config['version']}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        try:
            print(f"Installing {name}...")
            installer(config)
        except Exception as e:
            print(f"  Warning: {name} install failed: {e}")

    print()
    print("Done! Restart your AI clients to pick up the changes.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"
    generate_all(config_path)
