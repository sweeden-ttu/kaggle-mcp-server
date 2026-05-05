"""Platform-specific skill and config generator.

Reads skills.yaml and generates configurations for:
- OpenClaw (SKILL.md + openclaw.json update)
- Claude Desktop (claude_desktop_config.json update)
- Cursor (.cursor/skills/mlsyseng/SKILL.md)
- Gemini (mcp_config.json update)
- Generic (~/.skills/mlsyseng/SKILL.md)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(config_path: str = None) -> Dict[str, Any]:
    """Load the skills.yaml configuration."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "skills.yaml")

    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in a path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate the SKILL.md content from config."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_openclaw(config: Dict[str, Any]) -> str:
    """Install skill for OpenClaw platform."""
    platform = config["platforms"]["openclaw"]
    skill_dir = expand_path(platform["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_path = skill_dir / platform["skill_file"]
    skill_path.write_text(generate_skill_md(config))

    config_path = expand_path(platform["config_path"])
    if config_path.exists():
        try:
            with open(config_path) as f:
                openclaw_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            openclaw_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        openclaw_config = {}

    mcp_servers = openclaw_config.setdefault("mcpServers", {})
    server_config = config["mcp_server"]
    mcp_servers["mlsyseng-moe"] = {
        "command": server_config["command"],
        "args": server_config["args"],
        "env": {k: os.path.expandvars(v) for k, v in server_config["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(openclaw_config, f, indent=2)

    return f"OpenClaw: {skill_path}"


def install_claude_desktop(config: Dict[str, Any]) -> str:
    """Install MCP server config for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = expand_path(platform["config_path"])

    if config_path.exists():
        try:
            with open(config_path) as f:
                claude_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            claude_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        claude_config = {}

    mcp_servers = claude_config.setdefault("mcpServers", {})
    server_config = config["mcp_server"]
    mcp_servers["mlsyseng-moe"] = {
        "command": server_config["command"],
        "args": server_config["args"],
        "env": {k: os.path.expandvars(v) for k, v in server_config["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(claude_config, f, indent=2)

    return f"Claude Desktop: {config_path}"


def install_cursor(config: Dict[str, Any]) -> str:
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_dir = expand_path(platform["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_path = skill_dir / platform["skill_file"]
    skill_path.write_text(generate_skill_md(config))

    return f"Cursor: {skill_path}"


def install_gemini(config: Dict[str, Any]) -> str:
    """Install MCP config for Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = expand_path(platform["config_path"])

    if config_path.exists():
        try:
            with open(config_path) as f:
                gemini_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            gemini_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gemini_config = {}

    mcp_servers = gemini_config.setdefault("mcpServers", {})
    server_config = config["mcp_server"]
    mcp_servers["mlsyseng-moe"] = {
        "command": server_config["command"],
        "args": server_config["args"],
        "env": {k: os.path.expandvars(v) for k, v in server_config["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(gemini_config, f, indent=2)

    return f"Gemini: {config_path}"


def install_generic(config: Dict[str, Any]) -> str:
    """Install skill to generic location."""
    platform = config["platforms"]["generic"]
    skill_dir = expand_path(platform["skill_dir"])
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_path = skill_dir / platform["skill_file"]
    skill_path.write_text(generate_skill_md(config))

    return f"Generic: {skill_path}"


def install_all(config_path: str = None) -> Dict[str, str]:
    """Install skills to all platforms."""
    config = load_skills_config(config_path)
    results = {}

    installers = [
        ("openclaw", install_openclaw),
        ("claude_desktop", install_claude_desktop),
        ("cursor", install_cursor),
        ("gemini", install_gemini),
        ("generic", install_generic),
    ]

    for name, installer in installers:
        try:
            result = installer(config)
            results[name] = result
        except Exception as e:
            results[name] = f"Error: {e}"

    return results


def main():
    """CLI entry point for skill generation."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = install_all(config_path)

    print("MLSysEng MoE - Skill Generator")
    print("=" * 40)
    for platform, result in results.items():
        print(f"  {platform}: {result}")
    print("\nDone. Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
