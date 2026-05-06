"""Generate platform-specific skill configs from skills.yaml.

Reads the single source-of-truth skills.yaml and produces:
  - OpenClaw SKILL.md + openclaw.json update
  - Claude Desktop claude_desktop_config.json update
  - Cursor .cursor/skills/ SKILL.md
  - Gemini mcp_config.json update
  - Generic ~/.skills/ SKILL.md
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def load_skills_yaml(path: str = None) -> dict:
    if path is None:
        path = str(Path(__file__).parent / "skills.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def generate_skill_md(config: dict) -> str:
    """Generate a SKILL.md file from the skills config."""
    tools_section = ""
    for tool in config.get("tools", []):
        params = tool.get("parameters", {})
        param_lines = ""
        for pname, pdef in params.items():
            ptype = pdef.get("type", "string")
            default = pdef.get("default", "")
            desc = pdef.get("description", "")
            param_lines += f"  - `{pname}` ({ptype}): {desc}"
            if default:
                param_lines += f" (default: {default})"
            param_lines += "\n"

        tools_section += f"### {tool['name']}\n\n{tool['description']}\n\n"
        if param_lines:
            tools_section += f"**Parameters:**\n{param_lines}\n"

    return f"""# {config['name']}

{config['description']}

## MCP Server

```json
{{
  "command": "{config['mcp_server']['command']}",
  "args": {json.dumps(config['mcp_server']['args'])},
  "working_directory": "{config['mcp_server']['working_directory']}"
}}
```

## Tools

{tools_section}

## Quick Start

1. Extract knowledge: `extract-knowledge(force_reindex=false)`
2. List experts: `list-experts()`
3. Build entry: `build-entry(competition="titanic")`
4. Run convergence: `evolve(competition="titanic")`

## Expert System

Each chapter becomes an expert with:
- **Capabilities**: Domain-specific ML knowledge
- **Skills**: Mapped Kaggle skill paths
- **Strategy**: Step-by-step competition approach
- **Formula**: Objective function and metrics
- **Loop Config**: Convergence parameters (epsilon, patience, max_iterations)

## Convergence Loop

Exit condition: `||state[n] - state[n-1]||_2 < epsilon`
"""


def install_openclaw(config: dict) -> None:
    """Install skill to OpenClaw."""
    platform = config["platforms"].get("openclaw", {})
    skill_path = Path(os.path.expanduser(platform.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/")))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  OpenClaw: {skill_path / 'SKILL.md'}")

    config_path = Path(os.path.expanduser(platform.get("config_path", "~/.openclaw/openclaw.json")))
    _update_json_config(config_path, config, "openclaw")


def install_claude_desktop(config: dict) -> None:
    """Install skill to Claude Desktop."""
    platform = config["platforms"].get("claude_desktop", {})
    config_path = Path(os.path.expanduser(
        platform.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")
    ))
    _update_json_config(config_path, config, "claude_desktop")


def install_cursor(config: dict) -> None:
    """Install skill to Cursor."""
    platform = config["platforms"].get("cursor", {})
    skill_path = Path(os.path.expanduser(platform.get("skill_path", "~/.cursor/skills/mlsyseng/")))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Cursor: {skill_path / 'SKILL.md'}")


def install_gemini(config: dict) -> None:
    """Install skill to Gemini."""
    platform = config["platforms"].get("gemini", {})
    config_path = Path(os.path.expanduser(
        platform.get("config_path", "~/.gemini/mcp_config.json")
    ))
    _update_json_config(config_path, config, "gemini")


def install_generic(config: dict) -> None:
    """Install skill to generic path."""
    platform = config["platforms"].get("generic", {})
    skill_path = Path(os.path.expanduser(platform.get("skill_path", "~/.skills/mlsyseng/")))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Generic: {skill_path / 'SKILL.md'}")


def _update_json_config(config_path: Path, config: dict, platform_name: str) -> None:
    """Update a JSON config file with MCP server definition."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    server_def = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
    }

    env = config["mcp_server"].get("env", {})
    if env:
        server_def["env"] = {k: os.path.expanduser(str(v)) for k, v in env.items()}

    if platform_name in ("claude_desktop", "gemini"):
        servers = existing.setdefault("mcpServers", {})
        servers[config["name"]] = server_def
    else:
        servers = existing.setdefault("mcp_servers", {})
        servers[config["name"]] = server_def

    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  {platform_name}: {config_path}")


def main():
    config = load_skills_yaml()
    print(f"Generating skills for {config['name']} v{config['version']}")
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
            installer(config)
        except Exception as e:
            print(f"  Warning: {name} install failed: {e}")

    print()
    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
