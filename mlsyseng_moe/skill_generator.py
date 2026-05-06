"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def load_skills_config() -> dict:
    """Load the skills.yaml configuration."""
    config_path = Path(__file__).parent / "skills.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_skill_md(config: dict) -> str:
    """Generate SKILL.md content for skill-based platforms."""
    skill = config["skill"]
    tools = config["tools"]

    lines = [
        f"# {skill['name']}",
        "",
        config["description"],
        "",
        "## Tools",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]

    for tool in tools:
        lines.append(f"| `{tool['name']}` | {tool['description']} |")

    lines.extend([
        "",
        "## Instructions",
        "",
        skill["instructions"],
        "",
        "## Server Configuration",
        "",
        "```json",
        json.dumps({
            "command": config["server"]["command"],
            "args": config["server"]["args"],
        }, indent=2),
        "```",
    ])

    return "\n".join(lines)


def install_openclaw(config: dict) -> None:
    """Install skill for OpenClaw platform."""
    platforms = config["platforms"]
    skill_path = Path(os.path.expanduser(platforms["openclaw"]["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)

    config_path = Path(os.path.expanduser(platforms["openclaw"]["config_path"]))
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["server"]["command"],
        "args": config["server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["server"]["env"].items()},
    }
    config_path.write_text(json.dumps(existing, indent=2))

    print(f"  OpenClaw skill: {skill_path / 'SKILL.md'}")
    print(f"  OpenClaw config: {config_path}")


def install_claude_desktop(config: dict) -> None:
    """Install MCP server config for Claude Desktop."""
    platforms = config["platforms"]
    config_path = Path(os.path.expanduser(platforms["claude_desktop"]["config_path"]))

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["server"]["command"],
        "args": config["server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["server"]["env"].items()},
    }
    config_path.write_text(json.dumps(existing, indent=2))

    print(f"  Claude Desktop config: {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill for Cursor platform."""
    platforms = config["platforms"]
    skill_path = Path(os.path.expanduser(platforms["cursor"]["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)

    print(f"  Cursor skill: {skill_path / 'SKILL.md'}")


def install_gemini(config: dict) -> None:
    """Install MCP config for Gemini."""
    platforms = config["platforms"]
    config_path = Path(os.path.expanduser(platforms["gemini"]["config_path"]))

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["server"]["command"],
        "args": config["server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["server"]["env"].items()},
    }
    config_path.write_text(json.dumps(existing, indent=2))

    print(f"  Gemini config: {config_path}")


def install_generic(config: dict) -> None:
    """Install skill for generic platform."""
    platforms = config["platforms"]
    skill_path = Path(os.path.expanduser(platforms["generic"]["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)

    print(f"  Generic skill: {skill_path / 'SKILL.md'}")


def main():
    """Generate and install skills for all platforms."""
    print("MLSysEng MoE Skill Generator")
    print("=" * 40)

    config = load_skills_config()
    print(f"\nLoaded: {config['name']} v{config['version']}")
    print(f"Tools: {len(config['tools'])}")
    print()

    installers = {
        "OpenClaw": install_openclaw,
        "Claude Desktop": install_claude_desktop,
        "Cursor": install_cursor,
        "Gemini": install_gemini,
        "Generic": install_generic,
    }

    for platform_name, installer in installers.items():
        print(f"Installing for {platform_name}...")
        try:
            installer(config)
        except Exception as e:
            print(f"  Warning: {e}")
        print()

    print("Done! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
