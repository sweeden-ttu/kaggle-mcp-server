"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    """Load the skills.yaml source of truth."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(path_str: str) -> str:
    """Expand ~ and environment variables in paths."""
    return os.path.expandvars(os.path.expanduser(path_str))


def generate_skill_md(config: dict) -> str:
    """Generate the SKILL.md content."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_openclaw(config: dict) -> None:
    """Install skill for OpenClaw."""
    platform = config["platforms"].get("openclaw", {})
    skill_path = expand_path(platform.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/"))
    config_path = expand_path(platform.get("config_path", "~/.openclaw/openclaw.json"))

    os.makedirs(skill_path, exist_ok=True)
    skill_md = generate_skill_md(config)
    with open(os.path.join(skill_path, "SKILL.md"), "w") as f:
        f.write(skill_md)

    _update_json_config(config_path, config, "openclaw")
    print(f"  OpenClaw: {skill_path}SKILL.md")


def install_claude_desktop(config: dict) -> None:
    """Install skill for Claude Desktop."""
    platform = config["platforms"].get("claude_desktop", {})
    config_path = expand_path(platform.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json"))

    _update_json_config(config_path, config, "claude_desktop")
    print(f"  Claude Desktop: {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill for Cursor."""
    platform = config["platforms"].get("cursor", {})
    skill_path = expand_path(platform.get("skill_path", "~/.cursor/skills/mlsyseng/"))

    os.makedirs(skill_path, exist_ok=True)
    skill_md = generate_skill_md(config)
    with open(os.path.join(skill_path, "SKILL.md"), "w") as f:
        f.write(skill_md)
    print(f"  Cursor: {skill_path}SKILL.md")


def install_gemini(config: dict) -> None:
    """Install skill for Gemini."""
    platform = config["platforms"].get("gemini", {})
    config_path = expand_path(platform.get("config_path", "~/.gemini/mcp_config.json"))

    _update_json_config(config_path, config, "gemini")
    print(f"  Gemini: {config_path}")


def install_generic(config: dict) -> None:
    """Install skill for generic platform."""
    platform = config["platforms"].get("generic", {})
    skill_path = expand_path(platform.get("skill_path", "~/.skills/mlsyseng/"))

    os.makedirs(skill_path, exist_ok=True)
    skill_md = generate_skill_md(config)
    with open(os.path.join(skill_path, "SKILL.md"), "w") as f:
        f.write(skill_md)
    print(f"  Generic: {skill_path}SKILL.md")


def _update_json_config(config_path: str, config: dict, platform_name: str) -> None:
    """Update a JSON config file with MCP server configuration."""
    mcp_server = config.get("mcp_server", {})
    server_entry = {
        "command": mcp_server.get("command", "python"),
        "args": mcp_server.get("args", ["-m", "mlsyseng_mcp.server"]),
    }

    env = mcp_server.get("env", {})
    if env:
        resolved_env = {}
        for k, v in env.items():
            resolved_env[k] = expand_path(v.split(":-")[1].rstrip("}")) if ":-" in v else v
        server_entry["env"] = resolved_env

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    existing = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["mlsyseng-moe"] = server_entry

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    """Generate and install skills for all platforms."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    print(f"MLSysEng MoE Skill Generator v{config.get('version', '0.1.0')}")
    print("=" * 50)
    print(f"Installing skills for: {config.get('name', 'mlsyseng-moe')}")
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
            print(f"  {name}: SKIPPED ({e})")

    print()
    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
