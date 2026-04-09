#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the source-of-truth skills.yaml and generates:
- OpenClaw skill files and config
- Claude Desktop MCP config
- Cursor skill files
- Gemini MCP config
- Generic skill files
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p))


def write_skill_file(skill_path: str, skill_file: str, content: str):
    """Write a SKILL.md file to a platform directory."""
    dest = expand_path(skill_path)
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / skill_file
    filepath.write_text(content)
    print(f"  Wrote {filepath}")


def generate_openclaw(config: Dict[str, Any]):
    """Generate OpenClaw skill and update config."""
    print("OpenClaw:")
    platform = config["platforms"]["openclaw"]
    write_skill_file(
        platform["skill_path"],
        platform["skill_file"],
        config["skill_content"],
    )

    config_path = expand_path(platform["config_path"])
    if config_path.exists():
        try:
            with open(config_path) as f:
                oc_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            oc_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    mcp = config["mcp_server"]
    if "mcpServers" not in oc_config:
        oc_config["mcpServers"] = {}

    oc_config["mcpServers"][mcp["name"]] = {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": {k: os.path.expanduser(v) for k, v in mcp["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  Updated {config_path}")


def generate_claude_desktop(config: Dict[str, Any]):
    """Update Claude Desktop MCP config."""
    print("Claude Desktop:")
    platform = config["platforms"]["claude_desktop"]
    config_path = expand_path(platform["config_path"])

    if config_path.exists():
        try:
            with open(config_path) as f:
                cd_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cd_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    mcp = config["mcp_server"]
    if "mcpServers" not in cd_config:
        cd_config["mcpServers"] = {}

    cd_config["mcpServers"][mcp["name"]] = {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": {k: os.path.expanduser(v) for k, v in mcp["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Updated {config_path}")


def generate_cursor(config: Dict[str, Any]):
    """Generate Cursor skill file."""
    print("Cursor:")
    platform = config["platforms"]["cursor"]
    write_skill_file(
        platform["skill_path"],
        platform["skill_file"],
        config["skill_content"],
    )


def generate_gemini(config: Dict[str, Any]):
    """Update Gemini MCP config."""
    print("Gemini:")
    platform = config["platforms"]["gemini"]
    config_path = expand_path(platform["config_path"])

    if config_path.exists():
        try:
            with open(config_path) as f:
                g_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            g_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        g_config = {}

    mcp = config["mcp_server"]
    if "mcpServers" not in g_config:
        g_config["mcpServers"] = {}

    g_config["mcpServers"][mcp["name"]] = {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": {k: os.path.expanduser(v) for k, v in mcp["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(g_config, f, indent=2)
    print(f"  Updated {config_path}")


def generate_generic(config: Dict[str, Any]):
    """Generate generic skill file."""
    print("Generic:")
    platform = config["platforms"]["generic"]
    write_skill_file(
        platform["skill_path"],
        platform["skill_file"],
        config["skill_content"],
    )


def main():
    skills_path = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"
    config = load_skills_config(skills_path)

    print(f"Generating skills for: {config['name']} v{config['version']}")
    print("=" * 60)

    generate_openclaw(config)
    generate_claude_desktop(config)
    generate_cursor(config)
    generate_gemini(config)
    generate_generic(config)

    print("=" * 60)
    print("Done. Restart AI clients to pick up new skills.")


if __name__ == "__main__":
    main()
