"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config() -> Dict[str, Any]:
    """Load the skills.yaml configuration."""
    config_path = Path(__file__).parent / "skills.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_path(path_str: str) -> Path:
    """Expand ~ and environment variables in a path."""
    return Path(os.path.expandvars(os.path.expanduser(path_str)))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate the SKILL.md content."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_openclaw(config: Dict[str, Any]):
    """Install skill for OpenClaw/Claude Code."""
    platform = config["platforms"]["openclaw"]
    skill_path = expand_path(platform["skill_path"])
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(generate_skill_md(config))
    print(f"  [OpenClaw] Wrote {skill_file}")

    config_path = expand_path(platform["config_path"])
    if config_path.exists():
        try:
            with open(config_path) as f:
                openclaw_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            openclaw_config = {}

        mcp_servers = openclaw_config.setdefault("mcpServers", {})
        mcp_servers["mlsyseng-moe"] = {
            "command": config["mcp_server"]["command"],
            "args": config["mcp_server"]["args"],
            "env": {k: os.path.expandvars(v) for k, v in config["mcp_server"]["env"].items()},
        }

        with open(config_path, "w") as f:
            json.dump(openclaw_config, f, indent=2)
        print(f"  [OpenClaw] Updated {config_path}")
    else:
        print(f"  [OpenClaw] Config not found at {config_path}, skipping MCP registration")


def install_claude_desktop(config: Dict[str, Any]):
    """Install MCP server config for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = expand_path(platform["config_path"])

    if not config_path.parent.exists():
        print(f"  [Claude Desktop] Directory not found: {config_path.parent}, skipping")
        return

    try:
        if config_path.exists():
            with open(config_path) as f:
                desktop_config = json.load(f)
        else:
            desktop_config = {}
    except json.JSONDecodeError:
        desktop_config = {}

    mcp_servers = desktop_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["mcp_server"]["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(desktop_config, f, indent=2)
    print(f"  [Claude Desktop] Updated {config_path}")


def install_cursor(config: Dict[str, Any]):
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_path = expand_path(platform["skill_path"])
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(generate_skill_md(config))
    print(f"  [Cursor] Wrote {skill_file}")


def install_gemini(config: Dict[str, Any]):
    """Install MCP config for Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = expand_path(platform["config_path"])

    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if config_path.exists():
            with open(config_path) as f:
                gemini_config = json.load(f)
        else:
            gemini_config = {}
    except json.JSONDecodeError:
        gemini_config = {}

    mcp_servers = gemini_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: os.path.expandvars(v) for k, v in config["mcp_server"]["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(gemini_config, f, indent=2)
    print(f"  [Gemini] Updated {config_path}")


def install_generic(config: Dict[str, Any]):
    """Install skill to generic path."""
    platform = config["platforms"]["generic"]
    skill_path = expand_path(platform["skill_path"])
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(generate_skill_md(config))
    print(f"  [Generic] Wrote {skill_file}")


def main():
    """Generate and install skills for all platforms."""
    print("MLSysEng MoE - Skill Generator")
    print("=" * 40)

    config = load_skills_config()
    print(f"Loaded config: {config['name']} v{config['version']}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        try:
            installer(config)
        except Exception as e:
            print(f"  [{name}] Error: {e}")

    print()
    print("Done! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
