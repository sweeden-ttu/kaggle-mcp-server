#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate the SKILL.md content from config."""
    skill = config["skill"]
    lines = [
        f"# {skill['name']}",
        "",
        f"**Slug**: `{skill['slug']}`",
        "",
        skill["description"].strip(),
        "",
        "---",
        "",
        skill["instructions"].strip(),
        "",
        "---",
        "",
        "## MCP Server",
        "",
        f"- **Command**: `{config['mcp_server']['command']}`",
        f"- **Args**: `{' '.join(config['mcp_server']['args'])}`",
        "",
        "## Tools",
        "",
    ]
    for tool in config["tools"]:
        lines.append(f"- **{tool['name']}**: {tool['description']}")
    lines.append("")
    return "\n".join(lines)


def install_skill_file(skill_path: str, content: str) -> str:
    """Write SKILL.md to the given directory."""
    dest = Path(expand_path(skill_path))
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / "SKILL.md"
    filepath.write_text(content)
    return str(filepath)


def update_json_config(config_path: str, server_name: str, server_config: Dict) -> str:
    """Merge MCP server entry into a JSON config file."""
    path = Path(expand_path(config_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][server_name] = server_config

    path.write_text(json.dumps(existing, indent=2) + "\n")
    return str(path)


def build_mcp_server_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the MCP server JSON block."""
    srv = config["mcp_server"]
    env = {}
    for k, v in srv.get("env", {}).items():
        env[k] = expand_path(v.split(":-")[-1].rstrip("}")) if ":-" in v else v
    return {
        "command": srv["command"],
        "args": srv["args"],
        "env": env,
    }


def generate_all(config: Dict[str, Any]) -> None:
    """Generate and install skills for all platforms."""
    skill_md = generate_skill_md(config)
    mcp_entry = build_mcp_server_entry(config)
    platforms = config.get("platforms", {})
    installed = []

    for platform, pcfg in platforms.items():
        skill_path = pcfg.get("skill_path")
        config_path = pcfg.get("config_path")

        if skill_path:
            path = install_skill_file(skill_path, skill_md)
            installed.append(f"  {platform}: {path}")

        if config_path:
            path = update_json_config(config_path, "mlsyseng-moe", mcp_entry)
            installed.append(f"  {platform} config: {path}")

    if installed:
        print("Installed skills:")
        for line in installed:
            print(line)
    else:
        print("No platforms configured.")


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found", file=sys.stderr)
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    generate_all(config)
    print("\nDone. Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
