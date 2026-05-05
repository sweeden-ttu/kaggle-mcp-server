#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


SKILL_MD_TEMPLATE = """\
# {name}

{description}

## MCP Server

Start the server:

```bash
cd {server_dir}
python -m mlsyseng_mcp.server
```

## Available Tools

{tools_section}

## Expert System

Each ML Principles chapter is registered as an expert with:
- **Capabilities**: What the expert can do
- **Skills**: Kaggle skill paths the expert recommends
- **Strategy**: Step-by-step strategy (Baseline → EDA → Feature Eng → Model → Submit)
- **Formula**: Objective function and metrics

## Convergence Loop

The system uses a state convergence loop:
- **Exit condition**: `||state[n] - state[n-1]||_2 < epsilon`
- **Epsilon**: 0.001 (configurable)
- **Patience**: 3 consecutive converging iterations
- **Max iterations**: 10 (configurable)

## Quick Start

1. Extract knowledge: `extract-knowledge(force_reindex=false)`
2. List experts: `list-experts()`
3. Build entry: `build-entry(competition="titanic")`
4. Run convergence: `evolve(competition="titanic")`
"""


def _build_tools_section(config: Dict[str, Any]) -> str:
    lines = []
    for tool in config.get("tools", []):
        name = tool["name"]
        desc = tool.get("description", "").strip().replace("\n", " ")
        lines.append(f"- **`{name}`**: {desc}")
    return "\n".join(lines)


def generate_skill_md(config: Dict[str, Any], server_dir: str) -> str:
    return SKILL_MD_TEMPLATE.format(
        name=config["name"],
        description=config["description"].strip(),
        server_dir=server_dir,
        tools_section=_build_tools_section(config),
    )


def install_openclaw(config: Dict[str, Any], server_dir: str) -> None:
    platform = config["platforms"]["openclaw"]
    skill_path = Path(_expand_path(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(generate_skill_md(config, server_dir))
    print(f"  OpenClaw skill: {skill_file}")

    config_path = Path(_expand_path(platform["config_path"]))
    if config_path.exists():
        with open(config_path, "r") as f:
            oc_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_cfg = config["mcp_server"]
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg["command"],
        "args": mcp_cfg["args"],
        "env": {k: _expand_path(v) for k, v in mcp_cfg["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  OpenClaw config: {config_path}")


def install_claude_desktop(config: Dict[str, Any], server_dir: str) -> None:
    platform = config["platforms"]["claude_desktop"]
    config_path = Path(_expand_path(platform["config_path"]))

    if config_path.exists():
        with open(config_path, "r") as f:
            cd_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_cfg = config["mcp_server"]
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg["command"],
        "args": mcp_cfg["args"],
        "env": {k: _expand_path(v) for k, v in mcp_cfg["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Claude Desktop config: {config_path}")


def install_cursor(config: Dict[str, Any], server_dir: str) -> None:
    platform = config["platforms"]["cursor"]
    skill_path = Path(_expand_path(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(generate_skill_md(config, server_dir))
    print(f"  Cursor skill: {skill_file}")


def install_gemini(config: Dict[str, Any], server_dir: str) -> None:
    platform = config["platforms"]["gemini"]
    config_path = Path(_expand_path(platform["config_path"]))

    if config_path.exists():
        with open(config_path, "r") as f:
            gm_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gm_config = {}

    mcp_servers = gm_config.setdefault("mcpServers", {})
    mcp_cfg = config["mcp_server"]
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg["command"],
        "args": mcp_cfg["args"],
        "env": {k: _expand_path(v) for k, v in mcp_cfg["env"].items()},
    }

    with open(config_path, "w") as f:
        json.dump(gm_config, f, indent=2)
    print(f"  Gemini config: {config_path}")


def install_generic(config: Dict[str, Any], server_dir: str) -> None:
    platform = config["platforms"]["generic"]
    skill_path = Path(_expand_path(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(generate_skill_md(config, server_dir))
    print(f"  Generic skill: {skill_file}")


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"
    config = load_skills_config(config_file)
    server_dir = os.path.dirname(os.path.abspath(config_file))

    print(f"MLSysEng MoE Skill Generator v{config['version']}")
    print(f"Server directory: {server_dir}")
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
            installer(config, server_dir)
        except Exception as e:
            print(f"  Warning: {name} install failed: {e}")
        print()

    print("Done! Restart your AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
