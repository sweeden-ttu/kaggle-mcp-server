"""Generate platform-specific skill configurations from skills.yaml.

Reads the source-of-truth skills.yaml and installs skill definitions
for OpenClaw, Claude Desktop, Cursor, Gemini, and generic platforms.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    """Load the skills.yaml source of truth."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_env(value: str) -> str:
    """Expand environment variable references like ${VAR:-default}."""
    if not isinstance(value, str):
        return value
    import re
    def replace(match):
        var = match.group(1)
        default = match.group(3) if match.group(3) else ""
        return os.environ.get(var, default)
    return re.sub(r"\$\{([^:}]+)(:-([^}]*))?\}", replace, value)


def get_mcp_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build MCP server configuration from skills.yaml."""
    mcp = config.get("mcp_server", {})
    env = {}
    for k, v in mcp.get("env", {}).items():
        env[k] = expand_env(v)

    return {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": env,
    }


def write_skill_md(skill_path: str, config: Dict[str, Any]):
    """Write a SKILL.md file for a platform."""
    path = Path(os.path.expanduser(skill_path))
    path.mkdir(parents=True, exist_ok=True)

    skill_file = path / "SKILL.md"
    content = config.get("skill_content", "# MLSysEng MoE Skill\n\nNo content defined in skills.yaml.")
    skill_file.write_text(content)
    print(f"  Written: {skill_file}")


def update_json_config(config_path: str, config_key: str, server_config: Dict[str, Any], server_name: str = "mlsyseng-moe"):
    """Update a JSON configuration file with MCP server config."""
    path = Path(os.path.expanduser(config_path))

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    if config_key not in existing:
        existing[config_key] = {}

    existing[config_key][server_name] = server_config

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  Updated: {path}")


def install_openclaw(config: Dict[str, Any], server_config: Dict[str, Any]):
    """Install skills for OpenClaw."""
    print("\n[OpenClaw]")
    platform = config.get("platforms", {}).get("openclaw", {})

    skill_path = platform.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/")
    write_skill_md(skill_path, config)

    config_path = platform.get("config_path", "~/.openclaw/openclaw.json")
    config_key = platform.get("config_key", "mcpServers")
    update_json_config(config_path, config_key, server_config)


def install_claude_desktop(config: Dict[str, Any], server_config: Dict[str, Any]):
    """Install skills for Claude Desktop."""
    print("\n[Claude Desktop]")
    platform = config.get("platforms", {}).get("claude_desktop", {})

    config_path = platform.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")
    config_key = platform.get("config_key", "mcpServers")
    update_json_config(config_path, config_key, server_config)


def install_cursor(config: Dict[str, Any], server_config: Dict[str, Any]):
    """Install skills for Cursor."""
    print("\n[Cursor]")
    platform = config.get("platforms", {}).get("cursor", {})

    skill_path = platform.get("skill_path", "~/.cursor/skills/mlsyseng/")
    write_skill_md(skill_path, config)


def install_gemini(config: Dict[str, Any], server_config: Dict[str, Any]):
    """Install skills for Gemini."""
    print("\n[Gemini]")
    platform = config.get("platforms", {}).get("gemini", {})

    config_path = platform.get("config_path", "~/.gemini/mcp_config.json")
    config_key = platform.get("config_key", "mcpServers")
    update_json_config(config_path, config_key, server_config)


def install_generic(config: Dict[str, Any], server_config: Dict[str, Any]):
    """Install skills for generic platform."""
    print("\n[Generic]")
    platform = config.get("platforms", {}).get("generic", {})

    skill_path = platform.get("skill_path", "~/.skills/mlsyseng/")
    write_skill_md(skill_path, config)


def main():
    """Generate and install skills for all platforms."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: skills.yaml not found at {yaml_path}")
        sys.exit(1)

    print(f"Loading skills from: {yaml_path}")
    config = load_skills_yaml(yaml_path)
    server_config = get_mcp_server_config(config)

    print(f"\nMCP Server: {config.get('name', 'mlsyseng-moe')} v{config.get('version', '0.1.0')}")
    print(f"Command: {server_config['command']} {' '.join(server_config['args'])}")

    install_openclaw(config, server_config)
    install_claude_desktop(config, server_config)
    install_cursor(config, server_config)
    install_gemini(config, server_config)
    install_generic(config, server_config)

    print("\nDone! Restart your AI clients to pick up changes.")
    print("\nQuick test:")
    print(f"  {server_config['command']} {' '.join(server_config['args'])}")


if __name__ == "__main__":
    main()
