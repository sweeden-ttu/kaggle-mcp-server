#!/usr/bin/env python3
"""Generate platform-specific skill configs from skills.yaml.

Reads skills.yaml (source of truth) and writes:
- OpenClaw:       ~/.openclaw/workspace/skills/mlsyseng/SKILL.md
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
- Cursor:         ~/.cursor/skills/mlsyseng/SKILL.md
- Gemini:         ~/.gemini/mcp_config.json
- Generic:        ~/.skills/mlsyseng/SKILL.md

Also updates ~/.openclaw/openclaw.json with the MCP server config.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand(path_str: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path_str)))


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _build_skill_md(config: Dict[str, Any]) -> str:
    """Build a SKILL.md file from the skills config."""
    skill = config.get("skill", {})
    name = skill.get("name", config.get("name", "MLSysEng MoE"))
    instructions = skill.get("instructions", "")
    tools = config.get("tools", [])

    lines = [
        f"# {name}",
        "",
        config.get("description", "").strip(),
        "",
        "## Tools",
        "",
    ]

    for tool in tools:
        lines.append(f"- **{tool['name']}**: {tool['description']}")

    lines.extend([
        "",
        "## Usage",
        "",
        instructions.strip(),
        "",
        "## MCP Server",
        "",
        "```json",
        json.dumps(
            {
                "command": config["mcp_server"]["command"],
                "args": config["mcp_server"]["args"],
            },
            indent=2,
        ),
        "```",
    ])

    return "\n".join(lines)


def install_openclaw(config: Dict[str, Any]):
    """Install skill to OpenClaw."""
    plat = config.get("platforms", {}).get("openclaw", {})
    skill_dir = _expand(plat.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/"))
    _ensure_dir(skill_dir)

    skill_md = _build_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    print(f"  OpenClaw skill: {skill_dir / 'SKILL.md'}")

    config_path = _expand(plat.get("config_file", "~/.openclaw/openclaw.json"))
    mcp_cfg = config.get("mcp_server", {})

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        _ensure_dir(config_path.parent)
        existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg.get("command", "python"),
        "args": mcp_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  OpenClaw config: {config_path}")


def install_claude_desktop(config: Dict[str, Any]):
    """Install MCP server config to Claude Desktop."""
    plat = config.get("platforms", {}).get("claude_desktop", {})
    config_path = _expand(
        plat.get(
            "config_file",
            "~/Library/Application Support/Claude/claude_desktop_config.json",
        )
    )
    mcp_cfg = config.get("mcp_server", {})

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        _ensure_dir(config_path.parent)
        existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg.get("command", "python"),
        "args": mcp_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Claude Desktop config: {config_path}")


def install_cursor(config: Dict[str, Any]):
    """Install skill to Cursor."""
    plat = config.get("platforms", {}).get("cursor", {})
    skill_dir = _expand(plat.get("skill_path", "~/.cursor/skills/mlsyseng/"))
    _ensure_dir(skill_dir)

    skill_md = _build_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    print(f"  Cursor skill: {skill_dir / 'SKILL.md'}")


def install_gemini(config: Dict[str, Any]):
    """Install MCP server config to Gemini."""
    plat = config.get("platforms", {}).get("gemini", {})
    config_path = _expand(plat.get("config_file", "~/.gemini/mcp_config.json"))
    mcp_cfg = config.get("mcp_server", {})

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        _ensure_dir(config_path.parent)
        existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg.get("command", "python"),
        "args": mcp_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Gemini config: {config_path}")


def install_generic(config: Dict[str, Any]):
    """Install skill to generic location."""
    plat = config.get("platforms", {}).get("generic", {})
    skill_dir = _expand(plat.get("skill_path", "~/.skills/mlsyseng/"))
    _ensure_dir(skill_dir)

    skill_md = _build_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    print(f"  Generic skill: {skill_dir / 'SKILL.md'}")


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    print(f"Loading skills from {yaml_path}...")
    config = load_skills_yaml(yaml_path)

    print(f"\nInstalling {config['name']} v{config['version']} skills...\n")

    install_openclaw(config)
    install_claude_desktop(config)
    install_cursor(config)
    install_gemini(config)
    install_generic(config)

    print("\nAll skills installed successfully!")
    print("\nRestart your AI clients to pick up the new MCP server configuration.")


if __name__ == "__main__":
    main()
