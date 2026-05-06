#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the canonical skills.yaml and produces:
- OpenClaw: SKILL.md + MCP server config in openclaw.json
- Claude Desktop: MCP server entry in claude_desktop_config.json
- Cursor: SKILL.md in ~/.cursor/skills/mlsyseng/
- Gemini: MCP config entry in ~/.gemini/mcp_config.json
- Generic: SKILL.md in ~/.skills/mlsyseng/
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_env(s: str) -> str:
    """Resolve ${VAR:-default} patterns in strings."""
    import re

    def _replace(match):
        var = match.group(1)
        default = match.group(2) if match.group(2) else ""
        return os.environ.get(var, default)

    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", _replace, s)


def _expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p))


def generate_skill_md(config: dict) -> str:
    """Generate SKILL.md content from config."""
    skill = config["skill"]
    tools = config["tools"]

    lines = [
        f"# {skill['title']}",
        "",
        skill["description"].strip(),
        "",
        "## MCP Tools",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]

    for tool in tools:
        lines.append(f"| `{tool['name']}` | {tool['description']} |")

    lines.extend(
        [
            "",
            "## Tool Details",
            "",
        ]
    )

    for tool in tools:
        lines.append(f"### `{tool['name']}`")
        lines.append(f"\n{tool['description']}\n")
        if "params" in tool:
            lines.append("**Parameters:**\n")
            for param in tool["params"]:
                required = " (required)" if param.get("required") else ""
                default = f" (default: {param.get('default')})" if "default" in param else ""
                lines.append(f"- `{param['name']}`: {param.get('type', 'string')}{required}{default}")
                if param.get("description"):
                    lines.append(f"  - {param['description']}")
            lines.append("")

    return "\n".join(lines)


def generate_mcp_server_config(config: dict) -> dict:
    """Generate MCP server configuration dict."""
    server = config["mcp_server"]
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = _resolve_env(v)

    return {
        "command": server["command"],
        "args": server["args"],
        "env": env,
    }


def install_openclaw(config: dict) -> None:
    """Install skill to OpenClaw."""
    platform = config["platforms"]["openclaw"]
    skill_dir = _expand_path(platform["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  OpenClaw SKILL.md -> {skill_dir / 'SKILL.md'}")

    workspace_dir = _expand_path(platform["workspace_dir"])
    workspace_dir.mkdir(parents=True, exist_ok=True)

    config_path = _expand_path(platform["config_path"])
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, Exception):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    mcp_config = generate_mcp_server_config(config)
    existing.setdefault("mcpServers", {})[config["name"]] = mcp_config
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  OpenClaw config -> {config_path}")


def install_claude_desktop(config: dict) -> None:
    """Install MCP server config to Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = _expand_path(platform["config_path"])

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, Exception):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    mcp_config = generate_mcp_server_config(config)
    existing.setdefault("mcpServers", {})[config["name"]] = mcp_config
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Claude Desktop config -> {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill to Cursor."""
    platform = config["platforms"]["cursor"]
    skill_dir = _expand_path(platform["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  Cursor SKILL.md -> {skill_dir / 'SKILL.md'}")


def install_gemini(config: dict) -> None:
    """Install MCP config to Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = _expand_path(platform["config_path"])

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, Exception):
            existing = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    mcp_config = generate_mcp_server_config(config)
    existing.setdefault("mcpServers", {})[config["name"]] = mcp_config
    config_path.write_text(json.dumps(existing, indent=2))
    print(f"  Gemini config -> {config_path}")


def install_generic(config: dict) -> None:
    """Install skill to generic location."""
    platform = config["platforms"]["generic"]
    skill_dir = _expand_path(platform["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  Generic SKILL.md -> {skill_dir / 'SKILL.md'}")


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
    config = load_skills_yaml(yaml_path)

    print(f"MLSysEng MoE Skill Generator v{config['version']}")
    print(f"Generating skills for: {config['name']}")
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
            installer(config)
        except Exception as e:
            print(f"  Warning: {name} installation failed: {e}")
        print()

    print("Done! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
