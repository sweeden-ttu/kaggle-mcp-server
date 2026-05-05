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


def write_skill_md(target_dir: str, template: str):
    """Write SKILL.md to a target directory."""
    target = Path(expand_path(target_dir))
    target.mkdir(parents=True, exist_ok=True)
    skill_file = target / "SKILL.md"
    skill_file.write_text(template)
    print(f"  Wrote {skill_file}")


def build_mcp_config(skills: Dict[str, Any]) -> Dict[str, Any]:
    """Build the MCP server configuration entry."""
    server = skills["mcp_server"]
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = expand_path(v.replace("${" + k + ":-", "").rstrip("}")) if ":-" in v else v

    return {
        "command": server["command"],
        "args": server["args"],
        "env": env,
    }


def update_json_config(config_path: str, config_key: str, server_name: str, server_config: Dict):
    """Update a JSON config file with the MCP server entry."""
    path = Path(expand_path(config_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if path.exists():
        try:
            config = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}

    if config_key not in config:
        config[config_key] = {}

    config[config_key][server_name] = server_config

    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Updated {path}")


def generate_all(skills_path: str = "skills.yaml"):
    """Generate skill files and configs for all platforms."""
    skills = load_skills_yaml(skills_path)
    mcp_config = build_mcp_config(skills)
    template = skills.get("skill_template", "# MLSysEng MoE\n")
    server_name = skills["name"]

    platforms = skills.get("platforms", {})

    print(f"Generating skills for {server_name} v{skills['version']}\n")

    for platform, pconf in platforms.items():
        print(f"Platform: {platform}")

        skill_path = pconf.get("skill_path")
        if skill_path:
            write_skill_md(skill_path, template)

        config_path = pconf.get("config_path")
        config_key = pconf.get("config_key")
        if config_path and config_key:
            update_json_config(config_path, config_key, server_name, mcp_config)

        print()

    print("Done. Restart your AI clients to pick up changes.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"
    generate_all(path)
