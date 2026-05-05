"""Generate platform-specific skill configurations from skills.yaml.

Reads the source-of-truth skills.yaml and generates:
- OpenClaw SKILL.md and MCP config
- Claude Desktop MCP config
- Cursor SKILL.md
- Gemini MCP config
- Generic SKILL.md
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    """Load the skills.yaml source of truth."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in a path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def generate_skill_md(config: dict) -> str:
    """Generate a SKILL.md file from the skills config."""
    skill = config.get("skill", {})
    tools = config.get("tools", [])

    lines = [
        f"# {skill.get('name', config['name'])}",
        "",
        config.get("description", "").strip(),
        "",
        "## Tools",
        "",
    ]

    for tool in tools:
        name = tool["name"]
        desc = tool.get("description", "").strip().split("\n")[0]
        lines.append(f"- **{name}**: {desc}")

    lines.extend([
        "",
        "## Instructions",
        "",
        skill.get("instructions", "Use the MCP tools to interact with the system.").strip(),
        "",
        "## MCP Server",
        "",
        "```json",
        json.dumps(
            {
                "command": config["server"]["command"],
                "args": config["server"]["args"],
                "env": config["server"].get("env", {}),
            },
            indent=2,
        ),
        "```",
    ])

    return "\n".join(lines)


def generate_mcp_config_entry(config: dict) -> dict:
    """Generate an MCP server configuration entry."""
    server = config["server"]
    return {
        "command": server["command"],
        "args": server["args"],
        "env": {k: os.path.expandvars(v) for k, v in server.get("env", {}).items()},
    }


def install_openclaw(config: dict) -> None:
    """Install skill for OpenClaw platform."""
    platforms = config.get("platforms", {})
    openclaw_cfg = platforms.get("openclaw", {})

    skill_path = expand_path(openclaw_cfg.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/"))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  [OpenClaw] Written: {skill_path / 'SKILL.md'}")

    config_path = expand_path(openclaw_cfg.get("config_path", "~/.openclaw/openclaw.json"))
    _update_json_config(config_path, config)
    print(f"  [OpenClaw] Updated: {config_path}")


def install_claude_desktop(config: dict) -> None:
    """Install MCP config for Claude Desktop."""
    platforms = config.get("platforms", {})
    claude_cfg = platforms.get("claude_desktop", {})

    config_path = expand_path(claude_cfg.get(
        "config_path",
        "~/Library/Application Support/Claude/claude_desktop_config.json",
    ))

    if not config_path.parent.exists():
        print(f"  [Claude Desktop] Skipped (path not found): {config_path.parent}")
        return

    _update_json_config(config_path, config)
    print(f"  [Claude Desktop] Updated: {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill for Cursor."""
    platforms = config.get("platforms", {})
    cursor_cfg = platforms.get("cursor", {})

    skill_path = expand_path(cursor_cfg.get("skill_path", "~/.cursor/skills/mlsyseng/"))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  [Cursor] Written: {skill_path / 'SKILL.md'}")


def install_gemini(config: dict) -> None:
    """Install MCP config for Gemini."""
    platforms = config.get("platforms", {})
    gemini_cfg = platforms.get("gemini", {})

    config_path = expand_path(gemini_cfg.get("config_path", "~/.gemini/mcp_config.json"))

    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    _update_json_config(config_path, config)
    print(f"  [Gemini] Updated: {config_path}")


def install_generic(config: dict) -> None:
    """Install skill to generic path."""
    platforms = config.get("platforms", {})
    generic_cfg = platforms.get("generic", {})

    skill_path = expand_path(generic_cfg.get("skill_path", "~/.skills/mlsyseng/"))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  [Generic] Written: {skill_path / 'SKILL.md'}")


def _update_json_config(config_path: Path, skills_config: dict) -> None:
    """Update a JSON config file with MCP server entry."""
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    server_name = skills_config["name"]
    existing["mcpServers"][server_name] = generate_mcp_config_entry(skills_config)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2))


def main():
    """Generate and install skills for all platforms."""
    yaml_path = Path(__file__).parent / "skills.yaml"
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_yaml(str(yaml_path))
    print(f"Loaded skills config: {config['name']} v{config['version']}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for platform_name, installer in installers:
        try:
            installer(config)
        except Exception as e:
            print(f"  [{platform_name}] Error: {e}")

    print()
    print("Done! Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
