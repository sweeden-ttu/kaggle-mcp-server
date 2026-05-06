#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import os
import json
import sys
from pathlib import Path

import yaml


def load_skills_config() -> dict:
    """Load skills.yaml configuration."""
    config_path = Path(__file__).parent / "skills.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_skill_md(config: dict) -> str:
    """Generate SKILL.md content from config."""
    skill = config.get("skill", {})
    tools = config.get("tools", [])

    tools_table = "| Tool | Description |\n|------|-------------|\n"
    for tool in tools:
        tools_table += f"| `{tool['name']}` | {tool['description']} |\n"

    return f"""# {skill.get('title', config['name'])}

{config.get('description', '').strip()}

## Instructions

{skill.get('instructions', '').strip()}

## Available Tools

{tools_table}

## MCP Server

Start the server with:
```bash
python -m mlsyseng_mcp.server
```

## Configuration

Environment variables:
- `ML_PRINCIPLES_PATH`: Path to ML Principles PDF chapters
- `SQLITE_DB_PATH`: SQLite database path
- `CHROMA_DB_PATH`: ChromaDB vector store path
- `KAGGLE_SKILLS_PATH`: Path to Kaggle skill definitions
"""


def install_openclaw(config: dict) -> None:
    """Install skill for OpenClaw."""
    platforms = config.get("platforms", {})
    openclaw = platforms.get("openclaw", {})

    skill_path = Path(os.path.expanduser(openclaw.get(
        "skill_path", "~/.openclaw/workspace/skills/mlsyseng/"
    )))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Installed: {skill_path / 'SKILL.md'}")

    config_path = Path(os.path.expanduser(openclaw.get(
        "config_path", "~/.openclaw/openclaw.json"
    )))

    mcp_config = config.get("mcp_server", {})
    server_entry = {
        "command": mcp_config.get("command", "python"),
        "args": mcp_config.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expanduser(v) for k, v in mcp_config.get("env", {}).items()},
    }

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = server_entry
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Updated: {config_path}")


def install_claude_desktop(config: dict) -> None:
    """Install MCP config for Claude Desktop."""
    platforms = config.get("platforms", {})
    claude = platforms.get("claude_desktop", {})

    config_path = Path(os.path.expanduser(claude.get(
        "config_path",
        "~/Library/Application Support/Claude/claude_desktop_config.json"
    )))

    mcp_config = config.get("mcp_server", {})
    server_entry = {
        "command": mcp_config.get("command", "python"),
        "args": mcp_config.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expanduser(v) for k, v in mcp_config.get("env", {}).items()},
    }

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = server_entry
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Updated: {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill for Cursor."""
    platforms = config.get("platforms", {})
    cursor = platforms.get("cursor", {})

    skill_path = Path(os.path.expanduser(cursor.get(
        "skill_path", "~/.cursor/skills/mlsyseng/"
    )))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Installed: {skill_path / 'SKILL.md'}")


def install_gemini(config: dict) -> None:
    """Install MCP config for Gemini."""
    platforms = config.get("platforms", {})
    gemini = platforms.get("gemini", {})

    config_path = Path(os.path.expanduser(gemini.get(
        "config_path", "~/.gemini/mcp_config.json"
    )))

    mcp_config = config.get("mcp_server", {})
    server_entry = {
        "command": mcp_config.get("command", "python"),
        "args": mcp_config.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expanduser(v) for k, v in mcp_config.get("env", {}).items()},
    }

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = server_entry
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Updated: {config_path}")


def install_generic(config: dict) -> None:
    """Install skill to generic location."""
    platforms = config.get("platforms", {})
    generic = platforms.get("generic", {})

    skill_path = Path(os.path.expanduser(generic.get(
        "skill_path", "~/.skills/mlsyseng/"
    )))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Installed: {skill_path / 'SKILL.md'}")


def main():
    """Generate and install skills for all platforms."""
    print("MLSysEng MoE Skill Generator")
    print("=" * 40)

    config = load_skills_config()
    print(f"Loaded: {config['name']} v{config['version']}")
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
            installer(config)
            print(f"  ✓ {name} complete")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
        print()

    print("Done! Restart AI clients to pick up new configuration.")


if __name__ == "__main__":
    main()
