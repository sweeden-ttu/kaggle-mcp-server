"""Skill generator for MLSysEng MoE.

Reads skills.yaml and generates platform-specific configurations for
Cursor, Claude Desktop, and generic skill directories.
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_config(config_path: str = "skills.yaml") -> dict:
    """Load the skills.yaml configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_skill_md(config: dict) -> str:
    """Generate the SKILL.md content from config."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_cursor_skill(config: dict, skill_md: str):
    """Install skill for Cursor IDE."""
    platforms = config.get("platforms", {})
    cursor_config = platforms.get("cursor", {})
    skill_dir = Path(os.path.expanduser(cursor_config.get("skill_dir", "~/.cursor/skills/mlsyseng")))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_file = skill_dir / cursor_config.get("file", "SKILL.md")
    skill_file.write_text(skill_md)
    print(f"  Cursor skill installed: {skill_file}")


def install_claude_desktop(config: dict):
    """Update Claude Desktop MCP configuration."""
    platforms = config.get("platforms", {})
    claude_config = platforms.get("claude_desktop", {})
    config_path = Path(os.path.expanduser(
        claude_config.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")
    ))

    mcp_server = config.get("mcp_server", {})
    server_config = {
        "command": mcp_server.get("command", "python"),
        "args": mcp_server.get("args", ["-m", "mlsyseng_moe.server"]),
    }

    env_vars = mcp_server.get("env", {})
    if env_vars:
        resolved_env = {}
        for k, v in env_vars.items():
            resolved_env[k] = os.path.expanduser(
                v.split(":-")[1].rstrip("}") if ":-" in v else v
            )
        server_config["env"] = resolved_env

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = server_config

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Claude Desktop config updated: {config_path}")


def install_generic_skill(config: dict, skill_md: str):
    """Install skill to generic skills directory."""
    platforms = config.get("platforms", {})
    generic_config = platforms.get("generic", {})
    skill_dir = Path(os.path.expanduser(generic_config.get("skill_dir", "~/.skills/mlsyseng")))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_file = skill_dir / generic_config.get("file", "SKILL.md")
    skill_file.write_text(skill_md)
    print(f"  Generic skill installed: {skill_file}")


def main():
    """Generate and install skills for all platforms."""
    config_path = "skills.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    print(f"Loading config from: {config_path}")
    config = load_skills_config(config_path)

    skill_md = generate_skill_md(config)
    print(f"Generated SKILL.md ({len(skill_md)} chars)")

    print("\nInstalling skills:")

    install_cursor_skill(config, skill_md)
    install_generic_skill(config, skill_md)

    try:
        install_claude_desktop(config)
    except Exception as e:
        print(f"  Claude Desktop: skipped ({e})")

    print("\nDone! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
