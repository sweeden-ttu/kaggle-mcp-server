"""Generate platform-specific skill configurations from skills.yaml.

Reads the source-of-truth skills.yaml and generates configuration files
for OpenClaw, Claude Desktop, Cursor, Gemini, and generic platforms.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(yaml_path: str = "skills.yaml") -> Dict[str, Any]:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def generate_skill_markdown(config: Dict[str, Any]) -> str:
    """Generate SKILL.md content from config."""
    skill = config.get("skill", {})
    tools = config.get("tools", [])

    lines = [
        f"# {skill.get('name', 'MLSysEng MoE')}",
        "",
        skill.get("description", ""),
        "",
    ]

    if skill.get("instructions"):
        lines.append(skill["instructions"])
        lines.append("")

    lines.append("## MCP Tools")
    lines.append("")
    lines.append("| Tool | Description |")
    lines.append("|------|-------------|")
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        alias = f" / `{tool['alias']}`" if tool.get("alias") else ""
        lines.append(f"| `{name}`{alias} | {desc} |")
    lines.append("")

    return "\n".join(lines)


def generate_mcp_server_config(config: Dict[str, Any], working_dir: str) -> Dict[str, Any]:
    """Generate MCP server configuration block."""
    server = config.get("mcp_server", {})
    env = {}
    for key, val in server.get("env", {}).items():
        env[key] = expand_path(val.split(":-")[1].rstrip("}")) if ":-" in val else expand_path(val)

    return {
        "command": server.get("command", "python"),
        "args": server.get("args", ["-m", "src.mlsyseng_mcp.server"]),
        "cwd": working_dir,
        "env": env,
    }


def install_openclaw(config: Dict[str, Any], working_dir: str):
    """Install skill and MCP config for OpenClaw."""
    platform = config.get("platforms", {}).get("openclaw", {})
    skill_path = expand_path(platform.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng"))
    skill_file = platform.get("skill_file", "SKILL.md")

    os.makedirs(skill_path, exist_ok=True)
    skill_content = generate_skill_markdown(config)
    filepath = os.path.join(skill_path, skill_file)
    with open(filepath, "w") as f:
        f.write(skill_content)
    print(f"  OpenClaw skill: {filepath}")

    config_path = expand_path(platform.get("config_path", "~/.openclaw/openclaw.json"))
    mcp_config = generate_mcp_server_config(config, working_dir)

    if os.path.exists(config_path):
        with open(config_path) as f:
            try:
                openclaw_config = json.load(f)
            except json.JSONDecodeError:
                openclaw_config = {}
    else:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        openclaw_config = {}

    if "mcpServers" not in openclaw_config:
        openclaw_config["mcpServers"] = {}

    openclaw_config["mcpServers"]["mlsyseng-moe"] = mcp_config

    with open(config_path, "w") as f:
        json.dump(openclaw_config, f, indent=2)
    print(f"  OpenClaw config: {config_path}")


def install_claude_desktop(config: Dict[str, Any], working_dir: str):
    """Install MCP config for Claude Desktop."""
    platform = config.get("platforms", {}).get("claude_desktop", {})
    config_path = expand_path(platform.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json"))

    mcp_config = generate_mcp_server_config(config, working_dir)

    if os.path.exists(config_path):
        with open(config_path) as f:
            try:
                desktop_config = json.load(f)
            except json.JSONDecodeError:
                desktop_config = {}
    else:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        desktop_config = {}

    if "mcpServers" not in desktop_config:
        desktop_config["mcpServers"] = {}

    desktop_config["mcpServers"]["mlsyseng-moe"] = mcp_config

    with open(config_path, "w") as f:
        json.dump(desktop_config, f, indent=2)
    print(f"  Claude Desktop config: {config_path}")


def install_cursor(config: Dict[str, Any], working_dir: str):
    """Install skill for Cursor."""
    platform = config.get("platforms", {}).get("cursor", {})
    skill_path = expand_path(platform.get("skill_path", "~/.cursor/skills/mlsyseng"))
    skill_file = platform.get("skill_file", "SKILL.md")

    os.makedirs(skill_path, exist_ok=True)
    skill_content = generate_skill_markdown(config)
    filepath = os.path.join(skill_path, skill_file)
    with open(filepath, "w") as f:
        f.write(skill_content)
    print(f"  Cursor skill: {filepath}")


def install_gemini(config: Dict[str, Any], working_dir: str):
    """Install MCP config for Gemini."""
    platform = config.get("platforms", {}).get("gemini", {})
    config_path = expand_path(platform.get("config_path", "~/.gemini/mcp_config.json"))

    mcp_config = generate_mcp_server_config(config, working_dir)

    if os.path.exists(config_path):
        with open(config_path) as f:
            try:
                gemini_config = json.load(f)
            except json.JSONDecodeError:
                gemini_config = {}
    else:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        gemini_config = {}

    if "mcpServers" not in gemini_config:
        gemini_config["mcpServers"] = {}

    gemini_config["mcpServers"]["mlsyseng-moe"] = mcp_config

    with open(config_path, "w") as f:
        json.dump(gemini_config, f, indent=2)
    print(f"  Gemini config: {config_path}")


def install_generic(config: Dict[str, Any], working_dir: str):
    """Install skill for generic platform."""
    platform = config.get("platforms", {}).get("generic", {})
    skill_path = expand_path(platform.get("skill_path", "~/.skills/mlsyseng"))
    skill_file = platform.get("skill_file", "SKILL.md")

    os.makedirs(skill_path, exist_ok=True)
    skill_content = generate_skill_markdown(config)
    filepath = os.path.join(skill_path, skill_file)
    with open(filepath, "w") as f:
        f.write(skill_content)
    print(f"  Generic skill: {filepath}")


def main():
    yaml_path = "skills.yaml"
    if not os.path.exists(yaml_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, "skills.yaml")

    if not os.path.exists(yaml_path):
        print(f"Error: skills.yaml not found")
        sys.exit(1)

    config = load_skills_config(yaml_path)
    working_dir = os.path.dirname(os.path.abspath(yaml_path))

    print(f"MLSysEng MoE Skill Generator v{config.get('version', '1.0.0')}")
    print(f"Working directory: {working_dir}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        print(f"Installing {name}...")
        try:
            installer(config, working_dir)
        except Exception as e:
            print(f"  Warning: {name} installation failed: {e}")
        print()

    print("Done! Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
