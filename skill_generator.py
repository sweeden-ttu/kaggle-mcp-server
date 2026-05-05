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


def write_skill_md(output_dir: str, content: str) -> str:
    """Write SKILL.md to the given directory."""
    output_dir = expand_path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "SKILL.md")
    with open(path, "w") as f:
        f.write(content)
    print(f"  Wrote {path}")
    return path


def generate_openclaw(config: Dict[str, Any], mcp_config: Dict[str, Any]) -> None:
    """Generate OpenClaw skill and update config."""
    print("\n[OpenClaw]")
    platform = config["platforms"]["openclaw"]

    write_skill_md(platform["skill_dir"], config["skill_content"])

    workspace_dir = expand_path(platform["workspace_dir"])
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    config_path = expand_path(platform["config_path"])
    oc_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            oc_config = json.load(f)

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_servers[mcp_config["name"]] = {
        "command": mcp_config["command"],
        "args": mcp_config["args"],
        "env": mcp_config.get("env", {}),
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  Updated {config_path}")


def generate_claude_desktop(config: Dict[str, Any], mcp_config: Dict[str, Any]) -> None:
    """Update Claude Desktop config with MCP server."""
    print("\n[Claude Desktop]")
    platform = config["platforms"]["claude_desktop"]
    config_path = expand_path(platform["config_path"])

    cd_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cd_config = json.load(f)

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_servers[mcp_config["name"]] = {
        "command": mcp_config["command"],
        "args": mcp_config["args"],
        "env": mcp_config.get("env", {}),
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Updated {config_path}")


def generate_cursor(config: Dict[str, Any]) -> None:
    """Generate Cursor skill."""
    print("\n[Cursor]")
    platform = config["platforms"]["cursor"]
    write_skill_md(platform["skill_dir"], config["skill_content"])


def generate_gemini(config: Dict[str, Any], mcp_config: Dict[str, Any]) -> None:
    """Update Gemini MCP config."""
    print("\n[Gemini]")
    platform = config["platforms"]["gemini"]
    config_path = expand_path(platform["config_path"])

    gem_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            gem_config = json.load(f)

    mcp_servers = gem_config.setdefault("mcpServers", {})
    mcp_servers[mcp_config["name"]] = {
        "command": mcp_config["command"],
        "args": mcp_config["args"],
        "env": mcp_config.get("env", {}),
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(gem_config, f, indent=2)
    print(f"  Updated {config_path}")


def generate_generic(config: Dict[str, Any]) -> None:
    """Generate generic skill."""
    print("\n[Generic]")
    platform = config["platforms"]["generic"]
    write_skill_md(platform["skill_dir"], config["skill_content"])


def main() -> None:
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    mcp_config = config["mcp_server"]

    print(f"Generating skills for: {config['name']} v{config['version']}")

    generate_openclaw(config, mcp_config)
    generate_claude_desktop(config, mcp_config)
    generate_cursor(config)
    generate_gemini(config, mcp_config)
    generate_generic(config)

    print("\nDone! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
