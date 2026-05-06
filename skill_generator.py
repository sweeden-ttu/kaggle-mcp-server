"""Generate platform-specific skill configurations from skills.yaml.

Reads the unified skills.yaml and generates configs for:
- Claude Code (skill/ directory)
- Claude Desktop (claude_desktop_config.json)
- Cursor (.skills/ directory)
- Gemini (mcp_config.json)
- Generic (~/.skills/ directory)
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_config(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(p)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_skill_md(config: dict) -> str:
    """Generate a SKILL.md file from the config."""
    skill = config.get("skill", {})
    tools = config.get("tools", [])

    md = f"# {skill.get('title', config['name'])}\n\n"
    md += f"{config.get('description', '')}\n\n"
    md += "## Instructions\n\n"
    md += skill.get("instructions", "") + "\n\n"
    md += "## Available Tools\n\n"
    md += "| Tool | Description |\n|------|-------------|\n"
    for tool in tools:
        md += f"| `{tool['name']}` | {tool['description']} |\n"
    md += "\n"
    return md


def generate_mcp_config(config: dict, workspace_root: str) -> dict:
    """Generate an MCP server configuration entry."""
    server = config.get("mcp_server", {})
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = expand_path(v)

    return {
        "command": server.get("command", "python"),
        "args": server.get("args", []),
        "cwd": workspace_root,
        "env": env,
    }


def install_claude_code(config: dict, workspace_root: str):
    """Install skill for Claude Code (skill/ directory in workspace)."""
    skill_dir = os.path.join(workspace_root, "skill")
    ensure_dir(skill_dir)
    skill_md = generate_skill_md(config)
    path = os.path.join(skill_dir, "SKILL.md")
    with open(path, "w") as f:
        f.write(skill_md)
    print(f"  Claude Code: {path}")


def install_claude_desktop(config: dict, workspace_root: str):
    """Install MCP server config for Claude Desktop."""
    platform = config.get("platforms", {}).get("claude_desktop", {})
    config_path = expand_path(platform.get(
        "config_path",
        "~/Library/Application Support/Claude/claude_desktop_config.json"))

    ensure_dir(os.path.dirname(config_path))

    existing = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    key = platform.get("config_key", "mcpServers")
    if key not in existing:
        existing[key] = {}

    existing[key][config["name"]] = generate_mcp_config(config, workspace_root)

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Claude Desktop: {config_path}")


def install_cursor(config: dict, workspace_root: str):
    """Install skill for Cursor (.skills/ directory)."""
    skill_dir = os.path.join(
        expand_path("~/.cursor"), "skills", "mlsyseng")
    ensure_dir(skill_dir)
    skill_md = generate_skill_md(config)
    path = os.path.join(skill_dir, "SKILL.md")
    with open(path, "w") as f:
        f.write(skill_md)
    print(f"  Cursor: {path}")


def install_gemini(config: dict, workspace_root: str):
    """Install MCP server config for Gemini."""
    platform = config.get("platforms", {}).get("gemini", {})
    config_path = expand_path(platform.get(
        "config_path", "~/.gemini/mcp_config.json"))

    ensure_dir(os.path.dirname(config_path))

    existing = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    key = platform.get("config_key", "mcpServers")
    if key not in existing:
        existing[key] = {}

    existing[key][config["name"]] = generate_mcp_config(config, workspace_root)

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Gemini: {config_path}")


def install_generic(config: dict, workspace_root: str):
    """Install skill to generic ~/.skills/ directory."""
    skill_dir = expand_path("~/.skills/mlsyseng")
    ensure_dir(skill_dir)
    skill_md = generate_skill_md(config)
    path = os.path.join(skill_dir, "SKILL.md")
    with open(path, "w") as f:
        f.write(skill_md)
    print(f"  Generic: {path}")


def main():
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    skills_yaml = os.path.join(workspace_root, "skills.yaml")

    if not os.path.exists(skills_yaml):
        print(f"Error: {skills_yaml} not found")
        sys.exit(1)

    config = load_skills_config(skills_yaml)
    print(f"Generating skills for: {config['name']} v{config['version']}")
    print()

    installers = [
        ("Claude Code", install_claude_code),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        try:
            installer(config, workspace_root)
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    print()
    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
