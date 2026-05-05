#!/usr/bin/env python3
"""Generate platform-specific skill configs from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def expand(p: str) -> str:
    return os.path.expanduser(p)


def write_skill_md(target_path: str, content: str) -> None:
    target = Path(expand(target_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"  Wrote skill: {target}")


def update_json_config(config_path: str, key: str, server_config: Dict[str, Any]) -> None:
    path = Path(expand(config_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    servers = existing.setdefault(key, {})
    servers["mlsyseng-moe"] = server_config
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"  Updated config: {path}")


def build_mcp_server_entry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mcp = cfg["mcp_server"]
    entry: Dict[str, Any] = {
        "command": mcp["command"],
        "args": mcp["args"],
    }
    env = {}
    for k, v in mcp.get("env", {}).items():
        env[k] = expand(str(v))
    if env:
        entry["env"] = env
    return entry


def generate_openclaw(cfg: Dict[str, Any]) -> None:
    print("\n[OpenClaw]")
    plat = cfg["platforms"]["openclaw"]
    skill_content = cfg["skill"]["content"]
    write_skill_md(plat["skill_path"], skill_content)

    server_entry = build_mcp_server_entry(cfg)
    update_json_config(plat["config_path"], plat["config_key"], server_entry)


def generate_claude_desktop(cfg: Dict[str, Any]) -> None:
    print("\n[Claude Desktop]")
    plat = cfg["platforms"]["claude_desktop"]
    server_entry = build_mcp_server_entry(cfg)
    update_json_config(plat["config_path"], plat["config_key"], server_entry)


def generate_cursor(cfg: Dict[str, Any]) -> None:
    print("\n[Cursor]")
    plat = cfg["platforms"]["cursor"]
    skill_content = cfg["skill"]["content"]
    write_skill_md(plat["skill_path"], skill_content)


def generate_gemini(cfg: Dict[str, Any]) -> None:
    print("\n[Gemini]")
    plat = cfg["platforms"]["gemini"]
    server_entry = build_mcp_server_entry(cfg)
    update_json_config(plat["config_path"], plat["config_key"], server_entry)


def generate_generic(cfg: Dict[str, Any]) -> None:
    print("\n[Generic]")
    plat = cfg["platforms"]["generic"]
    skill_content = cfg["skill"]["content"]
    write_skill_md(plat["skill_path"], skill_content)


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    print(f"Loading skills from {yaml_path}")
    cfg = load_skills_yaml(yaml_path)
    print(f"Generating configs for: {cfg['name']} v{cfg['version']}")

    generate_openclaw(cfg)
    generate_claude_desktop(cfg)
    generate_cursor(cfg)
    generate_gemini(cfg)
    generate_generic(cfg)

    print("\nDone. Restart AI clients to pick up the new MCP server configuration.")


if __name__ == "__main__":
    main()
