#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the canonical skills.yaml and produces:
  - OpenClaw: SKILL.md + MCP server config in openclaw.json
  - Claude Desktop: MCP server entry in claude_desktop_config.json
  - Cursor: SKILL.md in .cursor/skills/
  - Gemini: MCP server entry in mcp_config.json
  - Generic: SKILL.md in ~/.skills/
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand(p: str) -> str:
    return os.path.expanduser(p)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_skill_md(directory: str, content: str):
    ensure_dir(directory)
    target = os.path.join(directory, "SKILL.md")
    with open(target, "w") as f:
        f.write(content)
    print(f"  Wrote {target}")


def update_json_config(config_path: str, config_key: str, server_name: str, server_config: dict):
    """Update a JSON config file with MCP server entry."""
    path = expand(config_path)
    ensure_dir(os.path.dirname(path))

    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    if config_key not in existing:
        existing[config_key] = {}

    existing[config_key][server_name] = server_config

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Updated {path}")


def build_mcp_config(skills: dict) -> dict:
    """Build the MCP server configuration dict."""
    mcp = skills["mcp_server"]
    env = {}
    for k, v in mcp.get("env", {}).items():
        env[k] = expand(v)

    return {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": env,
    }


def install_openclaw(skills: dict, mcp_config: dict):
    print("\n[OpenClaw]")
    plat = skills["platforms"]["openclaw"]
    skill_dir = expand(plat["skill_path"])
    write_skill_md(skill_dir, skills["skill_content"])
    update_json_config(plat["config_path"], plat["config_key"], skills["name"], mcp_config)


def install_claude_desktop(skills: dict, mcp_config: dict):
    print("\n[Claude Desktop]")
    plat = skills["platforms"]["claude_desktop"]
    update_json_config(plat["config_path"], plat["config_key"], skills["name"], mcp_config)


def install_cursor(skills: dict):
    print("\n[Cursor]")
    plat = skills["platforms"]["cursor"]
    skill_dir = expand(plat["skill_path"])
    write_skill_md(skill_dir, skills["skill_content"])


def install_gemini(skills: dict, mcp_config: dict):
    print("\n[Gemini]")
    plat = skills["platforms"]["gemini"]
    update_json_config(plat["config_path"], plat["config_key"], skills["name"], mcp_config)


def install_generic(skills: dict):
    print("\n[Generic]")
    plat = skills["platforms"]["generic"]
    skill_dir = expand(plat["skill_path"])
    write_skill_md(skill_dir, skills["skill_content"])


def main():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    skills = load_skills_yaml(yaml_path)
    mcp_config = build_mcp_config(skills)

    print(f"MLSysEng MoE Skill Generator v{skills['version']}")
    print(f"Installing skills for: {skills['name']}")

    install_openclaw(skills, mcp_config)
    install_claude_desktop(skills, mcp_config)
    install_cursor(skills)
    install_gemini(skills, mcp_config)
    install_generic(skills)

    print(f"\nDone. Installed {len(skills['tools'])} tools across all platforms.")
    print("Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
