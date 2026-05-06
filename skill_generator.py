#!/usr/bin/env python3
"""
Generate platform-specific skill configurations from skills.yaml.

Installs skills to:
- OpenClaw:  ~/.openclaw/workspace/skills/mlsyseng/
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
- Cursor:    ~/.cursor/skills/mlsyseng/
- Gemini:    ~/.gemini/mcp_config.json
- Generic:   ~/.skills/mlsyseng/
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


def _expand_path(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _build_skill_md(config: Dict[str, Any]) -> str:
    """Generate a SKILL.md from the config."""
    lines = [
        f"# {config['name']}",
        "",
        config["description"].strip(),
        "",
        "## MCP Server",
        "",
        f"Command: `{config['mcp_server']['command']} {' '.join(config['mcp_server']['args'])}`",
        "",
        "## Available Tools",
        "",
    ]

    for tool_name, tool_info in config["tools"].items():
        lines.append(f"### `{tool_name}`")
        lines.append("")
        lines.append(tool_info["description"])
        lines.append("")
        lines.append(f"```python\n{tool_info['usage']}\n```")
        lines.append("")

    lines.extend(
        [
            "## Quick Start",
            "",
            "1. Extract knowledge: `extract_knowledge()`",
            "2. List experts: `list_experts()`",
            "3. Build entry: `build_entry(competition=\"titanic\")`",
            "4. Run convergence: `evolve(competition=\"titanic\")`",
            "",
        ]
    )

    return "\n".join(lines)


def _build_mcp_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the MCP server JSON config block."""
    return {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": config["mcp_server"].get("env", {}),
    }


def install_openclaw(config: Dict[str, Any]) -> None:
    plat = config["platforms"]["openclaw"]
    skill_dir = _expand_path(plat["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = _build_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  [openclaw] Wrote {skill_dir / 'SKILL.md'}")

    config_file = _expand_path(plat["config_file"])
    if config_file.exists():
        try:
            oc_config = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            oc_config = {}
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    if "mcpServers" not in oc_config:
        oc_config["mcpServers"] = {}

    oc_config["mcpServers"]["mlsyseng-moe"] = _build_mcp_server_config(config)
    config_file.write_text(json.dumps(oc_config, indent=2))
    print(f"  [openclaw] Updated {config_file}")


def install_claude_desktop(config: Dict[str, Any]) -> None:
    plat = config["platforms"]["claude_desktop"]
    config_file = _expand_path(plat["config_file"])

    if config_file.exists():
        try:
            cd_config = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            cd_config = {}
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    if "mcpServers" not in cd_config:
        cd_config["mcpServers"] = {}

    cd_config["mcpServers"]["mlsyseng-moe"] = _build_mcp_server_config(config)
    config_file.write_text(json.dumps(cd_config, indent=2))
    print(f"  [claude_desktop] Updated {config_file}")


def install_cursor(config: Dict[str, Any]) -> None:
    plat = config["platforms"]["cursor"]
    skill_dir = _expand_path(plat["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = _build_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  [cursor] Wrote {skill_dir / 'SKILL.md'}")


def install_gemini(config: Dict[str, Any]) -> None:
    plat = config["platforms"]["gemini"]
    config_file = _expand_path(plat["config_file"])

    if config_file.exists():
        try:
            gm_config = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            gm_config = {}
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        gm_config = {}

    if "mcpServers" not in gm_config:
        gm_config["mcpServers"] = {}

    gm_config["mcpServers"]["mlsyseng-moe"] = _build_mcp_server_config(config)
    config_file.write_text(json.dumps(gm_config, indent=2))
    print(f"  [gemini] Updated {config_file}")


def install_generic(config: Dict[str, Any]) -> None:
    plat = config["platforms"]["generic"]
    skill_dir = _expand_path(plat["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = _build_skill_md(config)
    (skill_dir / "SKILL.md").write_text(skill_md)
    print(f"  [generic] Wrote {skill_dir / 'SKILL.md'}")


def main():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    config = load_skills_yaml(yaml_path)

    print(f"Generating skills for {config['name']} v{config['version']}")
    print()

    installers = {
        "openclaw": install_openclaw,
        "claude_desktop": install_claude_desktop,
        "cursor": install_cursor,
        "gemini": install_gemini,
        "generic": install_generic,
    }

    for name, installer in installers.items():
        try:
            installer(config)
        except Exception as e:
            print(f"  [{name}] Skipped: {e}")

    print()
    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
