#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(p)


def write_skill_file(directory: str, filename: str, content: str):
    d = Path(expand_path(directory))
    d.mkdir(parents=True, exist_ok=True)
    filepath = d / filename
    filepath.write_text(content)
    print(f"  Wrote {filepath}")


def build_mcp_server_config(config: dict) -> dict:
    server = config["mcp_server"]
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = expand_path(v)
    return {
        "command": server["command"],
        "args": server["args"],
        "env": env,
    }


def update_json_config(config_path: str, config_key: str, server_config: dict):
    path = Path(expand_path(config_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    if config_key not in existing:
        existing[config_key] = {}

    existing[config_key]["mlsyseng-moe"] = server_config
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  Updated {path}")


def generate_openclaw(config: dict):
    print("\n[OpenClaw]")
    plat = config["platforms"]["openclaw"]
    skill_content = config["skill_content"]
    write_skill_file(plat["skill_path"], plat["skill_file"], skill_content)

    server_config = build_mcp_server_config(config)
    update_json_config(plat["config_path"], plat["config_key"], server_config)


def generate_claude_desktop(config: dict):
    print("\n[Claude Desktop]")
    plat = config["platforms"]["claude_desktop"]
    server_config = build_mcp_server_config(config)
    update_json_config(plat["config_path"], plat["config_key"], server_config)


def generate_cursor(config: dict):
    print("\n[Cursor]")
    plat = config["platforms"]["cursor"]
    skill_content = config["skill_content"]
    write_skill_file(plat["skill_path"], plat["skill_file"], skill_content)


def generate_gemini(config: dict):
    print("\n[Gemini]")
    plat = config["platforms"]["gemini"]
    server_config = build_mcp_server_config(config)
    update_json_config(plat["config_path"], plat["config_key"], server_config)


def generate_generic(config: dict):
    print("\n[Generic]")
    plat = config["platforms"]["generic"]
    skill_content = config["skill_content"]
    write_skill_file(plat["skill_path"], plat["skill_file"], skill_content)


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    if not Path(yaml_path).exists():
        script_dir = Path(__file__).parent
        yaml_path = str(script_dir / "skills.yaml")

    config = load_skills_yaml(yaml_path)
    print(f"MLSysEng MoE Skill Generator v{config['version']}")
    print(f"Loaded: {yaml_path}")

    generate_openclaw(config)
    generate_claude_desktop(config)
    generate_cursor(config)
    generate_gemini(config)
    generate_generic(config)

    print("\nDone! Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
