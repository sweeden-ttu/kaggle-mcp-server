#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


SKILLS_YAML = Path(__file__).parent / "skills.yaml"


def load_skills_config() -> Dict[str, Any]:
    with open(SKILLS_YAML) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def write_skill_md(directory: str, content: str):
    """Write SKILL.md to a directory."""
    d = Path(expand_path(directory))
    d.mkdir(parents=True, exist_ok=True)
    skill_file = d / "SKILL.md"
    skill_file.write_text(content)
    print(f"  ✓ Wrote {skill_file}")


def update_claude_desktop_config(config_path: str, mcp_config: Dict[str, Any]):
    """Update Claude Desktop config with MCP server entry."""
    path = Path(expand_path(config_path))
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = {
        "command": mcp_config["command"],
        "args": mcp_config["args"],
        "env": {k: expand_path(v) for k, v in mcp_config.get("env", {}).items()},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✓ Updated {path}")


def update_gemini_config(config_path: str, mcp_config: Dict[str, Any]):
    """Update Gemini MCP config."""
    path = Path(expand_path(config_path))
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = {
        "command": mcp_config["command"],
        "args": mcp_config["args"],
        "env": {k: expand_path(v) for k, v in mcp_config.get("env", {}).items()},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✓ Updated {path}")


def generate_all():
    config = load_skills_config()
    skill_content = config.get("skill_content", "# MLSysEng MoE\n")
    mcp_config = config.get("mcp_server", {})
    platforms = config.get("platforms", {})

    print(f"MLSysEng MoE Skill Generator v{config.get('version', '1.0.0')}")
    print("=" * 50)

    for platform, pconfig in platforms.items():
        print(f"\n[{platform}]")

        skill_dir = pconfig.get("skill_dir")
        if skill_dir:
            write_skill_md(skill_dir, skill_content)

        config_file = pconfig.get("config_file")
        if config_file:
            if "claude" in platform.lower() and "desktop" in platform.lower():
                update_claude_desktop_config(config_file, mcp_config)
            elif "gemini" in platform.lower():
                update_gemini_config(config_file, mcp_config)

    print("\n" + "=" * 50)
    print("Done! Restart your AI clients to pick up changes.")


if __name__ == "__main__":
    generate_all()
