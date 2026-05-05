#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the single source-of-truth skills.yaml and writes:
- Claude Code skill (SKILL.md)
- Claude Desktop MCP config
- Cursor skill (SKILL.md)
- Gemini MCP config
- Generic skill (SKILL.md)
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


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate a SKILL.md from the skill_content section."""
    sc = config.get("skill_content", {})
    tools = config.get("tools", [])

    lines = [
        f"# {sc.get('title', config['name'])}",
        "",
        sc.get("description", ""),
        "",
    ]

    triggers = sc.get("triggers", [])
    if triggers:
        lines.append("## Triggers")
        lines.append("")
        for t in triggers:
            lines.append(f"- {t}")
        lines.append("")

    instructions = sc.get("instructions", "")
    if instructions:
        lines.append(instructions)
        lines.append("")

    lines.append("## Available Tools")
    lines.append("")
    lines.append("| Tool | Description |")
    lines.append("|------|-------------|")
    for tool in tools:
        name = tool.get("name", "")
        alias = tool.get("alias", "")
        desc = tool.get("description", "").replace("\n", " ").strip()
        display = f"`{name}`"
        if alias:
            display += f" / `{alias}`"
        lines.append(f"| {display} | {desc} |")
    lines.append("")

    lines.append("## MCP Server")
    lines.append("")
    mcp = config.get("mcp_server", {})
    cmd = mcp.get("command", "python")
    args = " ".join(mcp.get("args", []))
    lines.append(f"```bash")
    lines.append(f"{cmd} {args}")
    lines.append(f"```")
    lines.append("")

    env = mcp.get("env", {})
    if env:
        lines.append("### Environment Variables")
        lines.append("")
        lines.append("| Variable | Default |")
        lines.append("|----------|---------|")
        for k, v in env.items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    return "\n".join(lines)


def write_skill_file(dir_path: str, filename: str, content: str) -> str:
    d = Path(expand_path(dir_path))
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


def update_json_config(
    config_path: str, server_key: str, mcp_config: Dict[str, Any]
) -> str:
    """Update a JSON config file (Claude Desktop / Gemini) with MCP server entry."""
    p = Path(expand_path(config_path))
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"][server_key] = mcp_config

    p.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return str(p)


def build_mcp_server_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    mcp = config.get("mcp_server", {})
    entry = {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", []),
    }
    env = mcp.get("env", {})
    if env:
        resolved = {}
        for k, v in env.items():
            resolved[k] = expand_path(v)
        entry["env"] = resolved
    return entry


def generate_all(config: Dict[str, Any]) -> None:
    skill_md = generate_skill_md(config)
    mcp_entry = build_mcp_server_entry(config)
    platforms = config.get("platforms", {})

    results = []

    for platform_name, platform_cfg in platforms.items():
        if "skill_dir" in platform_cfg:
            path = write_skill_file(
                platform_cfg["skill_dir"],
                platform_cfg.get("skill_file", "SKILL.md"),
                skill_md,
            )
            results.append(f"  {platform_name}: {path}")

        if "config_path" in platform_cfg:
            path = update_json_config(
                platform_cfg["config_path"],
                platform_cfg.get("server_key", "mlsyseng-mcp"),
                mcp_entry,
            )
            results.append(f"  {platform_name}: {path}")

    print(f"Generated skills for {config['name']} v{config['version']}:")
    for r in results:
        print(r)


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found", file=sys.stderr)
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    generate_all(config)
    print("\nDone. Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
