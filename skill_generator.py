#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the canonical skills.yaml and installs skill files and MCP server
configurations for each supported platform.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _expand(path_str: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path_str)))


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_skill_md(path: Path, content: str):
    _ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ Wrote skill: {path}")


def _mcp_server_block(config: Dict[str, Any]) -> Dict[str, Any]:
    env = {}
    for k, v in config.get("env", {}).items():
        env[k] = os.path.expandvars(v.split(":-")[0].replace("${", "").replace("}", ""))
    return {
        "command": config["command"],
        "args": config["args"],
        "env": env,
    }


def install_openclaw(skills: Dict[str, Any]):
    print("\n[OpenClaw]")
    platform = skills["platforms"]["openclaw"]
    skill_path = _expand(platform["skill_path"])
    _write_skill_md(skill_path, skills["skill_content"])

    config_path = _expand(platform["config_path"])
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        _ensure_parent(config_path)
        config = {}

    mcp_servers = config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-mcp"] = _mcp_server_block(skills["mcp_server"])

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Updated config: {config_path}")


def install_claude_desktop(skills: Dict[str, Any]):
    print("\n[Claude Desktop]")
    platform = skills["platforms"]["claude_desktop"]
    config_path = _expand(platform["config_path"])

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        _ensure_parent(config_path)
        config = {}

    mcp_servers = config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-mcp"] = _mcp_server_block(skills["mcp_server"])

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Updated config: {config_path}")


def install_cursor(skills: Dict[str, Any]):
    print("\n[Cursor]")
    platform = skills["platforms"]["cursor"]
    skill_path = _expand(platform["skill_path"])
    _write_skill_md(skill_path, skills["skill_content"])


def install_gemini(skills: Dict[str, Any]):
    print("\n[Gemini]")
    platform = skills["platforms"]["gemini"]
    config_path = _expand(platform["config_path"])

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        _ensure_parent(config_path)
        config = {}

    mcp_servers = config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-mcp"] = _mcp_server_block(skills["mcp_server"])

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Updated config: {config_path}")


def install_generic(skills: Dict[str, Any]):
    print("\n[Generic]")
    platform = skills["platforms"]["generic"]
    skill_path = _expand(platform["skill_path"])
    _write_skill_md(skill_path, skills["skill_content"])


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found", file=sys.stderr)
        sys.exit(1)

    skills = load_skills_yaml(yaml_path)
    print(f"MLSysEng MoE Skill Generator v{skills['version']}")
    print(f"Name: {skills['name']}")

    install_openclaw(skills)
    install_claude_desktop(skills)
    install_cursor(skills)
    install_gemini(skills)
    install_generic(skills)

    print("\n✓ All platforms configured successfully!")
    print("\nRemember to restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
