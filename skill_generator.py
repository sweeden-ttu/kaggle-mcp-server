#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Installs skills to:
- OpenClaw: ~/.openclaw/workspace/skills/mlsyseng/
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
- Cursor: ~/.cursor/skills/mlsyseng/
- Gemini: ~/.gemini/mcp_config.json
- Generic: ~/.skills/mlsyseng/
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def _write_skill_md(directory: Path, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "SKILL.md"
    filepath.write_text(content)
    print(f"  Written: {filepath}")
    return filepath


def _update_json_config(config_path: Path, server_name: str, server_config: Dict):
    """Merge MCP server config into a JSON config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][server_name] = server_config
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Updated: {config_path}")


def generate_openclaw(config: Dict[str, Any]):
    print("\n[OpenClaw]")
    platforms = config.get("platforms", {}).get("openclaw", {})
    skill_dir = _expand(platforms.get("skill_dir", "~/.openclaw/workspace/skills/mlsyseng"))
    db_dir = _expand(platforms.get("db_dir", "~/.openclaw/workspace/mlsyseng"))
    config_file = _expand(platforms.get("config_file", "~/.openclaw/openclaw.json"))

    db_dir.mkdir(parents=True, exist_ok=True)
    _write_skill_md(skill_dir, config.get("skill_content", ""))

    mcp = config.get("mcp_server", {})
    server_config = {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", []),
        "env": {
            "SQLITE_DB_PATH": str(db_dir / "mlsyseng.db"),
            "CHROMA_DB_PATH": str(db_dir / "chroma_db"),
        },
    }
    _update_json_config(config_file, "mlsyseng-mcp", server_config)


def generate_claude_desktop(config: Dict[str, Any]):
    print("\n[Claude Desktop]")
    platforms = config.get("platforms", {}).get("claude_desktop", {})
    config_file = _expand(
        platforms.get("config_file", "~/Library/Application Support/Claude/claude_desktop_config.json")
    )

    mcp = config.get("mcp_server", {})
    server_config = {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", []),
        "env": mcp.get("env", {}),
    }
    _update_json_config(config_file, "mlsyseng-mcp", server_config)


def generate_cursor(config: Dict[str, Any]):
    print("\n[Cursor]")
    platforms = config.get("platforms", {}).get("cursor", {})
    skill_dir = _expand(platforms.get("skill_dir", "~/.cursor/skills/mlsyseng"))
    _write_skill_md(skill_dir, config.get("skill_content", ""))


def generate_gemini(config: Dict[str, Any]):
    print("\n[Gemini]")
    platforms = config.get("platforms", {}).get("gemini", {})
    config_file = _expand(platforms.get("config_file", "~/.gemini/mcp_config.json"))

    mcp = config.get("mcp_server", {})
    server_config = {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", []),
        "env": mcp.get("env", {}),
    }
    _update_json_config(config_file, "mlsyseng-mcp", server_config)


def generate_generic(config: Dict[str, Any]):
    print("\n[Generic]")
    platforms = config.get("platforms", {}).get("generic", {})
    skill_dir = _expand(platforms.get("skill_dir", "~/.skills/mlsyseng"))
    _write_skill_md(skill_dir, config.get("skill_content", ""))


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    print(f"Generating skills for: {config.get('name', 'unknown')}")

    generate_openclaw(config)
    generate_claude_desktop(config)
    generate_cursor(config)
    generate_gemini(config)
    generate_generic(config)

    print("\nDone! Restart AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
