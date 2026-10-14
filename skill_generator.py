#!/usr/bin/env python3
"""Generate platform-specific skill configs from skills.yaml.

Installs skills and MCP server configuration to:
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

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def _write_skill_md(skill_path: str, content: str) -> None:
    p = _expand(skill_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    print(f"  Wrote skill file: {p}")


def _mcp_server_config(config: dict) -> dict:
    server = config["mcp_server"]
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = os.path.expandvars(v)
    return {
        "command": server["command"],
        "args": server["args"],
        "env": env,
    }


def _merge_json_config(config_path: str, server_name: str, server_config: dict) -> None:
    p = _expand(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][server_name] = server_config

    p.write_text(json.dumps(existing, indent=2))
    print(f"  Updated config: {p}")


def generate_openclaw(config: dict) -> None:
    print("\n[OpenClaw]")
    platform = config["platforms"]["openclaw"]
    skill_content = config["skill_content"]
    _write_skill_md(platform["skill_path"], skill_content)

    server_config = _mcp_server_config(config)
    db_base = _expand(platform["db_base"])
    db_base.mkdir(parents=True, exist_ok=True)
    server_config["env"]["SQLITE_DB_PATH"] = str(db_base / "mlsyseng.db")
    server_config["env"]["CHROMA_DB_PATH"] = str(db_base / "chroma_db")

    _merge_json_config(platform["config_path"], config["name"], server_config)


def generate_claude_desktop(config: dict) -> None:
    print("\n[Claude Desktop]")
    platform = config["platforms"]["claude_desktop"]
    server_config = _mcp_server_config(config)
    _merge_json_config(platform["config_path"], config["name"], server_config)


def generate_cursor(config: dict) -> None:
    print("\n[Cursor]")
    platform = config["platforms"]["cursor"]
    skill_content = config["skill_content"]
    _write_skill_md(platform["skill_path"], skill_content)


def generate_gemini(config: dict) -> None:
    print("\n[Gemini]")
    platform = config["platforms"]["gemini"]
    server_config = _mcp_server_config(config)
    _merge_json_config(platform["config_path"], config["name"], server_config)


def generate_generic(config: dict) -> None:
    print("\n[Generic]")
    platform = config["platforms"]["generic"]
    skill_content = config["skill_content"]
    _write_skill_md(platform["skill_path"], skill_content)


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
    config = load_skills_yaml(yaml_path)

    print(f"MLSysEng MoE Skill Generator v{config['version']}")
    print(f"Generating skills for: {config['name']}")

    generate_openclaw(config)
    generate_claude_desktop(config)
    generate_cursor(config)
    generate_gemini(config)
    generate_generic(config)

    print("\nDone. Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
