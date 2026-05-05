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


def _expand(path_str: str) -> str:
    return os.path.expanduser(os.path.expandvars(path_str))


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_skill_md(directory: str, config: Dict[str, Any]):
    """Write a SKILL.md file for a platform."""
    _ensure_dir(directory)
    skill = config.get("skill", {})
    instructions = skill.get("instructions", "")

    content = f"""# {skill.get('name', 'MLSysEng MoE')}

{config.get('description', '')}

{instructions}
"""
    path = os.path.join(directory, "SKILL.md")
    with open(path, "w") as f:
        f.write(content)
    print(f"  Wrote {path}")


def _mcp_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build MCP server JSON block."""
    mcp = config.get("mcp_server", {})
    return {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: _expand(v) for k, v in mcp.get("env", {}).items()},
    }


def generate_openclaw(config: Dict[str, Any]):
    print("\n[OpenClaw]")
    platforms = config.get("platforms", {})
    oc = platforms.get("openclaw", {})

    skill_dir = _expand(oc.get("skill_dir", "~/.openclaw/workspace/skills/mlsyseng"))
    _write_skill_md(skill_dir, config)

    config_file = _expand(oc.get("config_file", "~/.openclaw/openclaw.json"))
    if os.path.exists(config_file):
        with open(config_file) as f:
            oc_config = json.load(f)
    else:
        _ensure_dir(os.path.dirname(config_file))
        oc_config = {}

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = _mcp_server_config(config)

    with open(config_file, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  Updated {config_file}")


def generate_claude_desktop(config: Dict[str, Any]):
    print("\n[Claude Desktop]")
    platforms = config.get("platforms", {})
    cd = platforms.get("claude_desktop", {})

    config_file = _expand(
        cd.get("config_file", "~/Library/Application Support/Claude/claude_desktop_config.json")
    )
    if os.path.exists(config_file):
        with open(config_file) as f:
            cd_config = json.load(f)
    else:
        _ensure_dir(os.path.dirname(config_file))
        cd_config = {}

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = _mcp_server_config(config)

    with open(config_file, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Updated {config_file}")


def generate_cursor(config: Dict[str, Any]):
    print("\n[Cursor]")
    platforms = config.get("platforms", {})
    cu = platforms.get("cursor", {})

    skill_dir = _expand(cu.get("skill_dir", "~/.cursor/skills/mlsyseng"))
    _write_skill_md(skill_dir, config)


def generate_gemini(config: Dict[str, Any]):
    print("\n[Gemini]")
    platforms = config.get("platforms", {})
    ge = platforms.get("gemini", {})

    config_file = _expand(ge.get("config_file", "~/.gemini/mcp_config.json"))
    if os.path.exists(config_file):
        with open(config_file) as f:
            ge_config = json.load(f)
    else:
        _ensure_dir(os.path.dirname(config_file))
        ge_config = {}

    mcp_servers = ge_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = _mcp_server_config(config)

    with open(config_file, "w") as f:
        json.dump(ge_config, f, indent=2)
    print(f"  Updated {config_file}")


def generate_generic(config: Dict[str, Any]):
    print("\n[Generic]")
    platforms = config.get("platforms", {})
    gen = platforms.get("generic", {})

    skill_dir = _expand(gen.get("skill_dir", "~/.skills/mlsyseng"))
    _write_skill_md(skill_dir, config)


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    print(f"Loading {yaml_path}...")
    config = load_skills_yaml(yaml_path)
    print(f"Generating skills for: {config.get('name', 'unknown')}")

    generate_openclaw(config)
    generate_claude_desktop(config)
    generate_cursor(config)
    generate_gemini(config)
    generate_generic(config)

    print("\nDone! Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
