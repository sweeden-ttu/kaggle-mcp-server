#!/usr/bin/env python3
"""Skill generator for MLSysEng MoE.

Reads skills.yaml and generates platform-specific skill configurations
for OpenClaw, Claude Desktop, Cursor, Gemini, and generic installs.
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def generate_skill_md(config: dict) -> str:
    """Generate a SKILL.md file from the skills config."""
    tools_md = ""
    for tool in config.get("tools", []):
        params = tool.get("parameters", {})
        param_lines = ""
        for pname, pinfo in params.items():
            param_lines += (
                f"  - `{pname}` ({pinfo.get('type', 'string')}): "
                f"{pinfo.get('description', '')} "
                f"(default: {pinfo.get('default', 'required')})\n"
            )
        tools_md += f"### {tool['name']}\n\n{tool['description']}\n\n"
        if param_lines:
            tools_md += f"**Parameters:**\n{param_lines}\n"

    return f"""# {config['name']} - ML Systems Expert MoE

{config['description']}

## MCP Server

This skill provides access to the MLSysEng MoE MCP server tools.

### Quick Start

1. Extract knowledge from ML Principles PDFs:
   ```
   extract-knowledge(force_reindex=false)
   ```

2. List available experts:
   ```
   list-experts()
   ```

3. Build a competition entry:
   ```
   build-entry(competition="titanic")
   ```

4. Run the convergence loop:
   ```
   evolve(competition="titanic")
   ```

## Tools

{tools_md}

## Architecture

The system uses a Mixture of Experts architecture where each ML Principles
chapter becomes an expert with capabilities, skills, strategy, and formulas.

RAG-based retrieval matches competition requirements to expert knowledge,
and a state convergence loop (L2-norm exit condition) iteratively refines
model performance.
"""


def generate_mcp_config(config: dict) -> dict:
    """Generate MCP server configuration block."""
    server = config.get("mcp_server", {})
    return {
        "command": server.get("command", "python"),
        "args": server.get("args", ["-m", "src.mlsyseng_mcp.server"]),
        "env": {
            k: expand_path(v) for k, v in server.get("env", {}).items()
        },
    }


def install_openclaw(config: dict):
    """Install skill to OpenClaw."""
    platform = config.get("platforms", {}).get("openclaw", {})
    skill_path = expand_path(platform.get(
        "skill_path", "~/.openclaw/workspace/skills/mlsyseng"
    ))
    config_path = expand_path(platform.get(
        "config_path", "~/.openclaw/openclaw.json"
    ))

    Path(skill_path).mkdir(parents=True, exist_ok=True)
    skill_md = generate_skill_md(config)
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(skill_md)
    print(f"  OpenClaw skill: {skill_file}")

    mcp_block = generate_mcp_config(config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            oc_config = json.load(f)
    else:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    key = platform.get("mcp_config_key", "mcpServers")
    if key not in oc_config:
        oc_config[key] = {}
    oc_config[key]["mlsyseng-moe"] = mcp_block

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  OpenClaw config: {config_path}")


def install_claude_desktop(config: dict):
    """Install skill to Claude Desktop."""
    platform = config.get("platforms", {}).get("claude_desktop", {})
    config_path = expand_path(platform.get(
        "config_path",
        "~/Library/Application Support/Claude/claude_desktop_config.json",
    ))

    mcp_block = generate_mcp_config(config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            cd_config = json.load(f)
    else:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    key = platform.get("mcp_config_key", "mcpServers")
    if key not in cd_config:
        cd_config[key] = {}
    cd_config[key]["mlsyseng-moe"] = mcp_block

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Claude Desktop config: {config_path}")


def install_cursor(config: dict):
    """Install skill to Cursor."""
    platform = config.get("platforms", {}).get("cursor", {})
    skill_path = expand_path(platform.get(
        "skill_path", "~/.cursor/skills/mlsyseng"
    ))

    Path(skill_path).mkdir(parents=True, exist_ok=True)
    skill_md = generate_skill_md(config)
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(skill_md)
    print(f"  Cursor skill: {skill_file}")


def install_gemini(config: dict):
    """Install skill to Gemini."""
    platform = config.get("platforms", {}).get("gemini", {})
    config_path = expand_path(platform.get(
        "config_path", "~/.gemini/mcp_config.json"
    ))

    mcp_block = generate_mcp_config(config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            gm_config = json.load(f)
    else:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        gm_config = {}

    key = platform.get("mcp_config_key", "mcpServers")
    if key not in gm_config:
        gm_config[key] = {}
    gm_config[key]["mlsyseng-moe"] = mcp_block

    with open(config_path, "w") as f:
        json.dump(gm_config, f, indent=2)
    print(f"  Gemini config: {config_path}")


def install_generic(config: dict):
    """Install skill to generic path."""
    platform = config.get("platforms", {}).get("generic", {})
    skill_path = expand_path(platform.get(
        "skill_path", "~/.skills/mlsyseng"
    ))

    Path(skill_path).mkdir(parents=True, exist_ok=True)
    skill_md = generate_skill_md(config)
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(skill_md)
    print(f"  Generic skill: {skill_file}")


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found", file=sys.stderr)
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    print(f"Generating skills for: {config['name']} v{config['version']}")
    print()

    installers = {
        "OpenClaw": install_openclaw,
        "Claude Desktop": install_claude_desktop,
        "Cursor": install_cursor,
        "Gemini": install_gemini,
        "Generic": install_generic,
    }

    for name, installer in installers.items():
        print(f"Installing {name}...")
        try:
            installer(config)
        except Exception as e:
            print(f"  Warning: {name} install failed: {e}")
        print()

    print("Done! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
