"""Skill generator for MLSysEng MoE.

Reads skills.yaml and generates platform-specific configurations
for OpenClaw, Claude Desktop, Cursor, Gemini, and generic installations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(path: str = "skills.yaml") -> Dict[str, Any]:
    """Load the skills.yaml configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> Path:
    """Expand ~ and environment variables in a path."""
    return Path(os.path.expandvars(os.path.expanduser(p)))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate the SKILL.md content from the config."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def build_mcp_server_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the MCP server configuration entry."""
    server = config.get("mcp_server", {})
    return {
        "command": server.get("command", "python"),
        "args": server.get("args", ["-m", "src.mlsyseng_moe.server"]),
        "env": {
            k: os.path.expandvars(v)
            for k, v in server.get("env", {}).items()
        },
    }


def write_skill_file(skill_dir: Path, filename: str, content: str) -> str:
    """Write a skill file to the target directory."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / filename
    path.write_text(content)
    return str(path)


def update_json_config(config_path: Path, config_key: str, server_entry: Dict[str, Any]) -> str:
    """Update a JSON configuration file with the MCP server entry."""
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    if config_key not in existing:
        existing[config_key] = {}

    existing[config_key]["mlsyseng-moe"] = server_entry
    config_path.write_text(json.dumps(existing, indent=2))
    return str(config_path)


def generate_all(config_path: str = "skills.yaml") -> Dict[str, str]:
    """Generate skill files and configs for all platforms."""
    config = load_skills_config(config_path)
    skill_content = generate_skill_md(config)
    server_entry = build_mcp_server_entry(config)
    platforms = config.get("platforms", {})
    results = {}

    for platform_name, platform_config in platforms.items():
        skill_dir = platform_config.get("skill_dir")
        skill_file = platform_config.get("skill_file")
        json_config = platform_config.get("config_path")
        json_key = platform_config.get("config_key")

        if skill_dir and skill_file:
            path = write_skill_file(
                expand_path(skill_dir),
                skill_file,
                skill_content,
            )
            results[f"{platform_name}_skill"] = path

        if json_config and json_key:
            path = update_json_config(
                expand_path(json_config),
                json_key,
                server_entry,
            )
            results[f"{platform_name}_config"] = path

    return results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"
    print(f"Generating skills from {config_path}...")
    results = generate_all(config_path)
    for key, path in sorted(results.items()):
        print(f"  {key}: {path}")
    print(f"\nGenerated {len(results)} outputs.")


if __name__ == "__main__":
    main()
