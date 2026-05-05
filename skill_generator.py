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


def get_skill_content(config: Dict[str, Any]) -> str:
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def get_mcp_server_config(config: Dict[str, Any], working_dir: str) -> Dict[str, Any]:
    server = config["mcp_server"]
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = expand_path(v)
    return {
        "command": server["command"],
        "args": server.get("args", []),
        "cwd": working_dir,
        "env": env,
    }


def install_skill_file(skill_dir: str, skill_file: str, content: str) -> str:
    d = Path(expand_path(skill_dir))
    d.mkdir(parents=True, exist_ok=True)
    out = d / skill_file
    out.write_text(content)
    return str(out)


def install_cursor(config: Dict[str, Any], working_dir: str) -> None:
    platform = config["platforms"].get("cursor", {})
    if not platform:
        return
    skill_dir = platform.get("skill_dir", "~/.cursor/skills/mlsyseng")
    skill_file = platform.get("skill_file", "SKILL.md")
    content = get_skill_content(config)
    path = install_skill_file(skill_dir, skill_file, content)
    print(f"  Cursor skill installed: {path}")


def install_openclaw(config: Dict[str, Any], working_dir: str) -> None:
    platform = config["platforms"].get("openclaw", {})
    if not platform:
        return

    skill_dir = platform.get("skill_dir", "~/.openclaw/workspace/skills/mlsyseng")
    skill_file = platform.get("skill_file", "SKILL.md")
    content = get_skill_content(config)
    path = install_skill_file(skill_dir, skill_file, content)
    print(f"  OpenClaw skill installed: {path}")

    config_path = expand_path(platform.get("config_path", "~/.openclaw/openclaw.json"))
    _update_json_config(config_path, config, working_dir, "mlsyseng-moe")


def install_claude_desktop(config: Dict[str, Any], working_dir: str) -> None:
    platform = config["platforms"].get("claude_desktop", {})
    if not platform:
        return
    config_path = expand_path(
        platform.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")
    )
    _update_json_config(config_path, config, working_dir, "mlsyseng-moe")


def install_gemini(config: Dict[str, Any], working_dir: str) -> None:
    platform = config["platforms"].get("gemini", {})
    if not platform:
        return
    config_path = expand_path(platform.get("config_path", "~/.gemini/mcp_config.json"))
    _update_json_config(config_path, config, working_dir, "mlsyseng-moe")


def install_generic(config: Dict[str, Any], working_dir: str) -> None:
    platform = config["platforms"].get("generic", {})
    if not platform:
        return
    skill_dir = platform.get("skill_dir", "~/.skills/mlsyseng")
    skill_file = platform.get("skill_file", "SKILL.md")
    content = get_skill_content(config)
    path = install_skill_file(skill_dir, skill_file, content)
    print(f"  Generic skill installed: {path}")


def _update_json_config(
    config_path: str,
    config: Dict[str, Any],
    working_dir: str,
    server_name: str,
) -> None:
    """Update a JSON config file with MCP server entry."""
    p = Path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    server_config = get_mcp_server_config(config, working_dir)
    existing["mcpServers"][server_name] = server_config

    p.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  Config updated: {config_path}")


def main() -> None:
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    if not Path(yaml_path).exists():
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    working_dir = os.path.dirname(os.path.abspath(yaml_path))

    print(f"MLSysEng MoE Skill Generator v{config.get('version', '0.1.0')}")
    print(f"Working directory: {working_dir}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        print(f"Installing {name}...")
        try:
            installer(config, working_dir)
        except Exception as exc:
            print(f"  Warning: {name} installation failed: {exc}")
        print()

    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
