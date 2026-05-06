"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def load_skills_yaml(path: str = None) -> dict:
    """Load the skills.yaml source of truth."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "skills.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def expand_env(value: str) -> str:
    """Expand environment variable references in a string."""
    if not isinstance(value, str):
        return value
    result = value
    while "${" in result:
        start = result.index("${")
        end = result.index("}", start)
        var_expr = result[start + 2:end]
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
        else:
            var_name, default = var_expr, ""
        result = result[:start] + os.environ.get(var_name, default) + result[end + 1:]
    return os.path.expanduser(result)


def generate_skill_md(config: dict) -> str:
    """Generate SKILL.md content."""
    skill = config["skill_content"]
    lines = [
        f"# {skill['title']}",
        "",
        f"## Description",
        "",
        config["description"].strip(),
        "",
        "## Triggers",
        "",
    ]
    for trigger in skill["triggers"]:
        lines.append(f"- {trigger}")

    lines.extend(["", "## Instructions", "", skill["instructions"].strip()])

    lines.extend(["", "## Available Tools", ""])
    for tool in config["tools"]:
        params = ""
        if tool.get("parameters"):
            param_strs = []
            for p in tool["parameters"]:
                req = " (required)" if p.get("required") else ""
                param_strs.append(f"`{p['name']}`{req}")
            params = f" - Params: {', '.join(param_strs)}"
        lines.append(f"- **{tool['name']}**: {tool['description']}{params}")

    lines.append("")
    return "\n".join(lines)


def generate_mcp_config(config: dict) -> dict:
    """Generate MCP server configuration block."""
    mcp = config["mcp_server"]
    env = {}
    for key, value in mcp.get("env", {}).items():
        env[key] = expand_env(value)

    return {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": env,
    }


def install_openclaw(config: dict) -> None:
    """Install skill for OpenClaw."""
    platform = config["platforms"]["openclaw"]
    skill_path = Path(expand_env(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_content = generate_skill_md(config)
    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(skill_content)
    print(f"  Written: {skill_file}")

    config_path = Path(expand_env(platform["config_path"]))
    if config_path.exists():
        with open(config_path) as f:
            openclaw_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        openclaw_config = {}

    if "mcpServers" not in openclaw_config:
        openclaw_config["mcpServers"] = {}

    openclaw_config["mcpServers"]["mlsyseng-moe"] = generate_mcp_config(config)

    with open(config_path, "w") as f:
        json.dump(openclaw_config, f, indent=2)
    print(f"  Updated: {config_path}")


def install_claude_desktop(config: dict) -> None:
    """Install skill for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = Path(expand_env(platform["config_path"]))

    if config_path.exists():
        with open(config_path) as f:
            claude_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        claude_config = {}

    if "mcpServers" not in claude_config:
        claude_config["mcpServers"] = {}

    claude_config["mcpServers"]["mlsyseng-moe"] = generate_mcp_config(config)

    with open(config_path, "w") as f:
        json.dump(claude_config, f, indent=2)
    print(f"  Updated: {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_path = Path(expand_env(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_content = generate_skill_md(config)
    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(skill_content)
    print(f"  Written: {skill_file}")


def install_gemini(config: dict) -> None:
    """Install skill for Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = Path(expand_env(platform["config_path"]))

    if config_path.exists():
        with open(config_path) as f:
            gemini_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gemini_config = {}

    if "mcpServers" not in gemini_config:
        gemini_config["mcpServers"] = {}

    gemini_config["mcpServers"]["mlsyseng-moe"] = generate_mcp_config(config)

    with open(config_path, "w") as f:
        json.dump(gemini_config, f, indent=2)
    print(f"  Updated: {config_path}")


def install_generic(config: dict) -> None:
    """Install skill to generic location."""
    platform = config["platforms"]["generic"]
    skill_path = Path(expand_env(platform["skill_path"]))
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_content = generate_skill_md(config)
    skill_file = skill_path / platform["skill_file"]
    skill_file.write_text(skill_content)
    print(f"  Written: {skill_file}")


def generate_all(yaml_path: str = None) -> None:
    """Generate skill configurations for all platforms."""
    config = load_skills_yaml(yaml_path)

    print("MLSysEng MoE Skill Generator")
    print("=" * 40)
    print(f"Version: {config['version']}")
    print()

    installers = {
        "OpenClaw": install_openclaw,
        "Claude Desktop": install_claude_desktop,
        "Cursor": install_cursor,
        "Gemini": install_gemini,
        "Generic": install_generic,
    }

    for platform_name, installer in installers.items():
        print(f"\n[{platform_name}]")
        try:
            installer(config)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 40)
    print("Done! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_all(yaml_path)
