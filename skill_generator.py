#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the source-of-truth skills.yaml and outputs:
- OpenClaw: ~/.openclaw/workspace/skills/mlsyseng/SKILL.md + config
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
- Cursor: ~/.cursor/skills/mlsyseng/SKILL.md
- Gemini: ~/.gemini/mcp_config.json
- Generic: ~/.skills/mlsyseng/SKILL.md
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p)))


def write_skill_md(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content)
    print(f"  Written: {target_path}")


def update_json_config(config_path: Path, server_name: str, server_config: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][server_name] = server_config

    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Updated: {config_path}")


def build_mcp_server_config(cfg: dict) -> dict:
    mcp = cfg["mcp_server"]
    env = {}
    for k, v in mcp.get("env", {}).items():
        env[k] = os.path.expanduser(
            os.path.expandvars(v.split(":-")[1].rstrip("}") if ":-" in v else v)
        )

    return {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": env,
    }


def generate_openclaw(cfg: dict) -> None:
    print("\n[OpenClaw]")
    plat = cfg["platforms"]["openclaw"]

    skill_path = expand_path(plat["skill_path"])
    write_skill_md(skill_path, cfg["skill_content"])

    config_path = expand_path(plat["config_path"])
    server_config = build_mcp_server_config(cfg)
    update_json_config(config_path, "mlsyseng-moe", server_config)


def generate_claude_desktop(cfg: dict) -> None:
    print("\n[Claude Desktop]")
    plat = cfg["platforms"]["claude_desktop"]
    config_path = expand_path(plat["config_path"])
    server_config = build_mcp_server_config(cfg)
    update_json_config(config_path, "mlsyseng-moe", server_config)


def generate_cursor(cfg: dict) -> None:
    print("\n[Cursor]")
    plat = cfg["platforms"]["cursor"]
    skill_path = expand_path(plat["skill_path"])
    write_skill_md(skill_path, cfg["skill_content"])


def generate_gemini(cfg: dict) -> None:
    print("\n[Gemini]")
    plat = cfg["platforms"]["gemini"]
    config_path = expand_path(plat["config_path"])
    server_config = build_mcp_server_config(cfg)
    update_json_config(config_path, "mlsyseng-moe", server_config)


def generate_generic(cfg: dict) -> None:
    print("\n[Generic]")
    plat = cfg["platforms"]["generic"]
    skill_path = expand_path(plat["skill_path"])
    write_skill_md(skill_path, cfg["skill_content"])


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found", file=sys.stderr)
        sys.exit(1)

    cfg = load_skills_yaml(yaml_path)
    print(f"Loaded skills.yaml: {cfg['name']} v{cfg['version']}")

    generate_openclaw(cfg)
    generate_claude_desktop(cfg)
    generate_cursor(cfg)
    generate_gemini(cfg)
    generate_generic(cfg)

    print("\nDone! Restart your AI clients to pick up changes.")


if __name__ == "__main__":
    main()
