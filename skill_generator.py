"""
Generates platform-specific skill configurations from skills.yaml.

Targets:
- Claude Code:    ~/.claude/skill/ (SKILL.md files)
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
- Cursor:         ~/.cursor/skills/mlsyseng/SKILL.md
- Gemini:         ~/.gemini/mcp_config.json
- Generic:        ~/.skills/mlsyseng/SKILL.md
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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_skill_md(config: Dict[str, Any]) -> str:
    """Build a SKILL.md file from the skills.yaml configuration."""
    meta = config.get("metadata", {})
    mcp = config.get("mcp_server", {})
    skills = config.get("skills", {})

    lines = [
        f"# {meta.get('name', 'mlsyseng-moe')}",
        "",
        f"{meta.get('description', '')}",
        "",
        "## MCP Server",
        "",
        f"Command: `{mcp.get('command', 'python')} {' '.join(mcp.get('args', []))}`",
        "",
        "## Available Tools",
        "",
    ]

    for skill_key, skill_def in skills.items():
        lines.append(f"### {skill_def.get('name', skill_key)}")
        lines.append("")
        lines.append(skill_def.get("description", ""))
        lines.append("")
        lines.append(f"**When to use:** {skill_def.get('when_to_use', '').strip()}")
        lines.append("")
        tools = skill_def.get("tools", [])
        if tools:
            lines.append("**Tools:**")
            for t in tools:
                lines.append(f"- `{t}`")
            lines.append("")
        examples = skill_def.get("examples", [])
        if examples:
            lines.append("**Examples:**")
            for ex in examples:
                lines.append(f"- {ex}")
            lines.append("")

    return "\n".join(lines)


def _build_mcp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build an MCP server configuration dict."""
    mcp = config.get("mcp_server", {})
    env = {}
    for key, val in mcp.get("env", {}).items():
        resolved = os.path.expandvars(val)
        if resolved != val:
            env[key] = resolved
        else:
            env[key] = val

    return {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", []),
        "env": env,
    }


def generate_claude_code(config: Dict[str, Any]) -> None:
    """Generate skills for Claude Code (~/.claude/skill/)."""
    target = Path.home() / ".claude" / "skill"
    _ensure_dir(target)
    skill_md = _build_skill_md(config)
    (target / "SKILL.md").write_text(skill_md, encoding="utf-8")
    print(f"  Claude Code: {target / 'SKILL.md'}")


def generate_claude_desktop(config: Dict[str, Any]) -> None:
    """Update Claude Desktop MCP config."""
    if sys.platform == "darwin":
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"

    _ensure_dir(config_path.parent)
    mcp_entry = _build_mcp_config(config)

    desktop_config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            desktop_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if "mcpServers" not in desktop_config:
        desktop_config["mcpServers"] = {}
    desktop_config["mcpServers"]["mlsyseng-moe"] = mcp_entry
    config_path.write_text(
        json.dumps(desktop_config, indent=2), encoding="utf-8"
    )
    print(f"  Claude Desktop: {config_path}")


def generate_cursor(config: Dict[str, Any]) -> None:
    """Generate skills for Cursor (~/.cursor/skills/mlsyseng/)."""
    target = Path.home() / ".cursor" / "skills" / "mlsyseng"
    _ensure_dir(target)
    skill_md = _build_skill_md(config)
    (target / "SKILL.md").write_text(skill_md, encoding="utf-8")
    print(f"  Cursor: {target / 'SKILL.md'}")


def generate_gemini(config: Dict[str, Any]) -> None:
    """Update Gemini MCP config."""
    config_path = Path.home() / ".gemini" / "mcp_config.json"
    _ensure_dir(config_path.parent)
    mcp_entry = _build_mcp_config(config)

    gemini_config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            gemini_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if "mcpServers" not in gemini_config:
        gemini_config["mcpServers"] = {}
    gemini_config["mcpServers"]["mlsyseng-moe"] = mcp_entry
    config_path.write_text(
        json.dumps(gemini_config, indent=2), encoding="utf-8"
    )
    print(f"  Gemini: {config_path}")


def generate_generic(config: Dict[str, Any]) -> None:
    """Generate generic skills (~/.skills/mlsyseng/)."""
    target = Path.home() / ".skills" / "mlsyseng"
    _ensure_dir(target)
    skill_md = _build_skill_md(config)
    (target / "SKILL.md").write_text(skill_md, encoding="utf-8")
    print(f"  Generic: {target / 'SKILL.md'}")


def main():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    config = load_skills_yaml(yaml_path)

    print("Generating MLSysEng MoE skills for all platforms...")
    print()

    generate_generic(config)
    generate_cursor(config)
    generate_claude_code(config)
    generate_claude_desktop(config)
    generate_gemini(config)

    print()
    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
