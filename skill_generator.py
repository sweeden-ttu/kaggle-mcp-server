"""Generate platform-specific skill configurations from skills.yaml.

Reads the single source of truth (skills.yaml) and installs skills to:
- OpenClaw: ~/.openclaw/workspace/skills/mlsyseng/
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
- Cursor: ~/.cursor/skills/mlsyseng/
- Gemini: ~/.gemini/mcp_config.json
- Generic: ~/.skills/mlsyseng/
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_config(path: str = "skills.yaml") -> dict:
    """Load the skills.yaml configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _generate_skill_md(config: dict) -> str:
    """Generate a SKILL.md file from the config."""
    skill = config.get("skill", {})
    tools = config.get("tools", [])

    lines = [
        f"# {skill.get('title', config['name'])}",
        "",
        config.get("description", "").strip(),
        "",
        "## Tools",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]

    for tool in tools:
        name = tool["name"]
        if tool.get("alias"):
            name += f" / {tool['alias']}"
        lines.append(f"| `{name}` | {tool['description']} |")

    lines.extend([
        "",
        "## Usage",
        "",
    ])

    for tool in tools:
        lines.append(f"### {tool['name']}")
        lines.append(f"```python\n{tool['usage']}\n```")
        lines.append("")

    instructions = skill.get("instructions", "")
    if instructions:
        lines.extend([
            "## Instructions",
            "",
            instructions.strip(),
            "",
        ])

    when = skill.get("when_to_use", [])
    if when:
        lines.append("## When to Use")
        lines.append("")
        for item in when:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def _mcp_server_config(config: dict, project_dir: str) -> dict:
    """Generate MCP server JSON config."""
    server = config.get("mcp_server", {})
    return {
        "command": server.get("command", "python"),
        "args": server.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {
            k: os.path.expanduser(v.split(":-")[1].rstrip("}"))
            if ":-" in v else v
            for k, v in server.get("env", {}).items()
        },
        "cwd": project_dir,
    }


def install_openclaw(config: dict, project_dir: str):
    """Install skill to OpenClaw."""
    plat = config.get("platforms", {}).get("openclaw", {})
    skill_path = Path(os.path.expanduser(
        plat.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/")
    ))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = _generate_skill_md(config)
    skill_file = skill_path / plat.get("skill_file", "SKILL.md")
    skill_file.write_text(skill_md)
    print(f"  OpenClaw skill: {skill_file}")

    config_path = Path(os.path.expanduser(
        plat.get("config_path", "~/.openclaw/openclaw.json")
    ))
    if config_path.exists():
        try:
            oc_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            oc_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = _mcp_server_config(config, project_dir)
    config_path.write_text(json.dumps(oc_config, indent=2) + "\n")
    print(f"  OpenClaw config: {config_path}")


def install_claude_desktop(config: dict, project_dir: str):
    """Install MCP server config for Claude Desktop."""
    plat = config.get("platforms", {}).get("claude_desktop", {})
    config_path = Path(os.path.expanduser(
        plat.get("config_path",
                  "~/Library/Application Support/Claude/claude_desktop_config.json")
    ))

    if config_path.exists():
        try:
            cd_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            cd_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = _mcp_server_config(config, project_dir)
    config_path.write_text(json.dumps(cd_config, indent=2) + "\n")
    print(f"  Claude Desktop config: {config_path}")


def install_cursor(config: dict, project_dir: str):
    """Install skill to Cursor."""
    plat = config.get("platforms", {}).get("cursor", {})
    skill_path = Path(os.path.expanduser(
        plat.get("skill_path", "~/.cursor/skills/mlsyseng/")
    ))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = _generate_skill_md(config)
    skill_file = skill_path / plat.get("skill_file", "SKILL.md")
    skill_file.write_text(skill_md)
    print(f"  Cursor skill: {skill_file}")


def install_gemini(config: dict, project_dir: str):
    """Install MCP server config for Gemini."""
    plat = config.get("platforms", {}).get("gemini", {})
    config_path = Path(os.path.expanduser(
        plat.get("config_path", "~/.gemini/mcp_config.json")
    ))

    if config_path.exists():
        try:
            gem_config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            gem_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gem_config = {}

    mcp_servers = gem_config.setdefault("mcpServers", {})
    mcp_servers["mlsyseng-moe"] = _mcp_server_config(config, project_dir)
    config_path.write_text(json.dumps(gem_config, indent=2) + "\n")
    print(f"  Gemini config: {config_path}")


def install_generic(config: dict, project_dir: str):
    """Install skill to generic path."""
    plat = config.get("platforms", {}).get("generic", {})
    skill_path = Path(os.path.expanduser(
        plat.get("skill_path", "~/.skills/mlsyseng/")
    ))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = _generate_skill_md(config)
    skill_file = skill_path / plat.get("skill_file", "SKILL.md")
    skill_file.write_text(skill_md)
    print(f"  Generic skill: {skill_file}")


def main():
    """Generate and install skills for all platforms."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "skills.yaml")

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_config(yaml_path)
    project_dir = script_dir

    print(f"MLSysEng MoE Skill Generator v{config.get('version', '0.1.0')}")
    print(f"Project directory: {project_dir}")
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
            installer(config, project_dir)
        except Exception as e:
            print(f"  Warning: {name} installation failed: {e}")
        print()

    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
