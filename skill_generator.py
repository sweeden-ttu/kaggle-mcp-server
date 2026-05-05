#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_config(path: str = "skills.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_skill_md(config: dict) -> str:
    """Generate SKILL.md content from config."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_openclaw(config: dict):
    """Install skill to OpenClaw."""
    platform = config["platforms"]["openclaw"]
    skill_dir = Path(os.path.expanduser(platform["skill_path"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  OpenClaw skill installed: {skill_md}")

    config_path = Path(os.path.expanduser(platform["config_path"]))
    if config_path.exists():
        try:
            oc_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            oc_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["mcp_server"]["env"].items()},
    }
    config_path.write_text(json.dumps(oc_config, indent=2))
    print(f"  OpenClaw config updated: {config_path}")


def install_claude_desktop(config: dict):
    """Install MCP server config to Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = Path(os.path.expanduser(platform["config_path"]))

    if config_path.exists():
        try:
            cd_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            cd_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["mcp_server"]["env"].items()},
    }
    config_path.write_text(json.dumps(cd_config, indent=2))
    print(f"  Claude Desktop config updated: {config_path}")


def install_cursor(config: dict):
    """Install skill to Cursor."""
    platform = config["platforms"]["cursor"]
    skill_dir = Path(os.path.expanduser(platform["skill_path"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  Cursor skill installed: {skill_md}")


def install_gemini(config: dict):
    """Install MCP server config to Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = Path(os.path.expanduser(platform["config_path"]))

    if config_path.exists():
        try:
            gem_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            gem_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gem_config = {}

    mcp_servers = gem_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["mcp_server"]["env"].items()},
    }
    config_path.write_text(json.dumps(gem_config, indent=2))
    print(f"  Gemini config updated: {config_path}")


def install_generic(config: dict):
    """Install skill to generic path."""
    platform = config["platforms"]["generic"]
    skill_dir = Path(os.path.expanduser(platform["skill_path"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  Generic skill installed: {skill_md}")


INSTALLERS = {
    "openclaw": install_openclaw,
    "claude_desktop": install_claude_desktop,
    "cursor": install_cursor,
    "gemini": install_gemini,
    "generic": install_generic,
}


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"
    config = load_skills_config(config_path)

    print(f"MLSysEng MoE Skill Generator v{config.get('version', '0.1.0')}")
    print(f"Source: {config_path}")
    print()

    platforms = sys.argv[2:] if len(sys.argv) > 2 else list(INSTALLERS.keys())

    for platform in platforms:
        if platform not in INSTALLERS:
            print(f"  Unknown platform: {platform}, skipping")
            continue
        if platform not in config.get("platforms", {}):
            print(f"  Platform {platform} not configured, skipping")
            continue
        print(f"Installing for {platform}...")
        try:
            INSTALLERS[platform](config)
        except Exception as e:
            print(f"  Error installing for {platform}: {e}")

    print("\nDone. Restart your AI clients to pick up the changes.")


if __name__ == "__main__":
    main()
