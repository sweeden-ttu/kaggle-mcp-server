#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate SKILL.md content from the skills config."""
    skill = config["skill"]
    tools = skill.get("tools", [])
    tool_list = "\n".join(f"- `{t}`" for t in tools)

    return f"""# {skill['name']}

{skill['description'].strip()}

## Instructions

{skill['instructions'].strip()}

## Available Tools

{tool_list}

## MCP Server

Start the server with:

```bash
python -m mlsyseng_mcp.server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_PRINCIPLES_PATH` | `~/Desktop/Machine Learning Principles - Chapters` | PDF chapters path |
| `SQLITE_DB_PATH` | `~/.openclaw/workspace/mlsyseng/mlsyseng.db` | Database path |
| `CHROMA_DB_PATH` | `~/.openclaw/workspace/mlsyseng/chroma_db` | Vector store path |
| `KAGGLE_SKILLS_PATH` | `~/skills` | Kaggle skills path |
"""


def generate_mcp_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the MCP server configuration block."""
    mcp = config["mcp_server"]
    env = {}
    for key, val in mcp.get("env", {}).items():
        cleaned = val.replace("${", "").split(":-")
        default = cleaned[1].rstrip("}") if len(cleaned) > 1 else ""
        env[key] = expand_path(default)

    return {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": env,
    }


def install_openclaw(config: Dict[str, Any]) -> None:
    """Install skill and MCP config for OpenClaw."""
    platform = config["platforms"]["openclaw"]
    skill_path = Path(expand_path(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Wrote {skill_path / 'SKILL.md'}")

    config_path = Path(expand_path(platform["config_path"]))
    if config_path.exists():
        with open(config_path) as f:
            oc_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    key = platform["config_key"]
    if key not in oc_config:
        oc_config[key] = {}

    oc_config[key]["mlsyseng-moe"] = generate_mcp_server_config(config)

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  Updated {config_path}")


def install_claude_desktop(config: Dict[str, Any]) -> None:
    """Install MCP config for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = Path(expand_path(platform["config_path"]))

    if config_path.exists():
        with open(config_path) as f:
            cd_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    key = platform["config_key"]
    if key not in cd_config:
        cd_config[key] = {}

    cd_config[key]["mlsyseng-moe"] = generate_mcp_server_config(config)

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Updated {config_path}")


def install_cursor(config: Dict[str, Any]) -> None:
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_path = Path(expand_path(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Wrote {skill_path / 'SKILL.md'}")


def install_gemini(config: Dict[str, Any]) -> None:
    """Install MCP config for Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = Path(expand_path(platform["config_path"]))

    if config_path.exists():
        with open(config_path) as f:
            gem_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gem_config = {}

    key = platform["config_key"]
    if key not in gem_config:
        gem_config[key] = {}

    gem_config[key]["mlsyseng-moe"] = generate_mcp_server_config(config)

    with open(config_path, "w") as f:
        json.dump(gem_config, f, indent=2)
    print(f"  Updated {config_path}")


def install_generic(config: Dict[str, Any]) -> None:
    """Install skill to generic path."""
    platform = config["platforms"]["generic"]
    skill_path = Path(expand_path(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    (skill_path / "SKILL.md").write_text(skill_md)
    print(f"  Wrote {skill_path / 'SKILL.md'}")


INSTALLERS = {
    "openclaw": install_openclaw,
    "claude_desktop": install_claude_desktop,
    "cursor": install_cursor,
    "gemini": install_gemini,
    "generic": install_generic,
}


def main():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    config = load_skills_yaml(yaml_path)

    platforms = sys.argv[1:] if len(sys.argv) > 1 else list(INSTALLERS.keys())

    print(f"MLSysEng MoE Skill Generator v{config['version']}")
    print(f"Installing skills for: {', '.join(platforms)}\n")

    for platform in platforms:
        installer = INSTALLERS.get(platform)
        if installer is None:
            print(f"Unknown platform: {platform}")
            continue
        print(f"[{platform}]")
        try:
            installer(config)
        except Exception as exc:
            print(f"  Error: {exc}")
        print()

    print("Done. Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
