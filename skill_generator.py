"""Generate platform-specific skill configurations from skills.yaml.

Reads the central skills.yaml and writes platform-specific configs for:
- OpenClaw
- Claude Desktop
- Cursor
- Gemini
- Generic
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(config_path: str = "skills.yaml") -> Dict[str, Any]:
    """Load the skills.yaml configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    """Expand ~ and environment variables in a path."""
    return os.path.expanduser(os.path.expandvars(p))


def write_skill_md(skill_path: str, content: str):
    """Write a SKILL.md file to the given directory."""
    path = Path(expand_path(skill_path))
    path.mkdir(parents=True, exist_ok=True)
    skill_file = path / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    print(f"  Written: {skill_file}")


def generate_openclaw(config: Dict[str, Any]):
    """Generate OpenClaw skill and update config."""
    platforms = config.get("platforms", {})
    openclaw = platforms.get("openclaw", {})
    skill_path = openclaw.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng")
    config_path = openclaw.get("config_path", "~/.openclaw/openclaw.json")

    template = config.get("skill_template", "# MLSysEng MoE Skill")
    write_skill_md(skill_path, template)

    oc_config_path = Path(expand_path(config_path))
    mcp_entry = {
        "mlsyseng-moe": {
            "command": config["mcp_server"]["command"],
            "args": config["mcp_server"]["args"],
            "env": config["mcp_server"].get("env", {}),
        }
    }

    if oc_config_path.exists():
        try:
            with open(oc_config_path, "r", encoding="utf-8") as f:
                oc_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            oc_data = {}
    else:
        oc_config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_data = {}

    if "mcpServers" not in oc_data:
        oc_data["mcpServers"] = {}
    oc_data["mcpServers"].update(mcp_entry)

    with open(oc_config_path, "w", encoding="utf-8") as f:
        json.dump(oc_data, f, indent=2)
    print(f"  Updated: {oc_config_path}")


def generate_claude_desktop(config: Dict[str, Any]):
    """Generate Claude Desktop MCP config."""
    platforms = config.get("platforms", {})
    claude = platforms.get("claude_desktop", {})
    config_path = claude.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")

    cd_config_path = Path(expand_path(config_path))
    mcp_entry = {
        "mlsyseng-moe": {
            "command": config["mcp_server"]["command"],
            "args": config["mcp_server"]["args"],
            "env": config["mcp_server"].get("env", {}),
        }
    }

    if cd_config_path.exists():
        try:
            with open(cd_config_path, "r", encoding="utf-8") as f:
                cd_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            cd_data = {}
    else:
        cd_config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_data = {}

    if "mcpServers" not in cd_data:
        cd_data["mcpServers"] = {}
    cd_data["mcpServers"].update(mcp_entry)

    with open(cd_config_path, "w", encoding="utf-8") as f:
        json.dump(cd_data, f, indent=2)
    print(f"  Updated: {cd_config_path}")


def generate_cursor(config: Dict[str, Any]):
    """Generate Cursor skill."""
    platforms = config.get("platforms", {})
    cursor = platforms.get("cursor", {})
    skill_path = cursor.get("skill_path", "~/.cursor/skills/mlsyseng")

    template = config.get("skill_template", "# MLSysEng MoE Skill")
    write_skill_md(skill_path, template)


def generate_gemini(config: Dict[str, Any]):
    """Generate Gemini MCP config."""
    platforms = config.get("platforms", {})
    gemini = platforms.get("gemini", {})
    config_path = gemini.get("config_path", "~/.gemini/mcp_config.json")

    gm_config_path = Path(expand_path(config_path))
    mcp_entry = {
        "mlsyseng-moe": {
            "command": config["mcp_server"]["command"],
            "args": config["mcp_server"]["args"],
            "env": config["mcp_server"].get("env", {}),
        }
    }

    if gm_config_path.exists():
        try:
            with open(gm_config_path, "r", encoding="utf-8") as f:
                gm_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            gm_data = {}
    else:
        gm_config_path.parent.mkdir(parents=True, exist_ok=True)
        gm_data = {}

    if "mcpServers" not in gm_data:
        gm_data["mcpServers"] = {}
    gm_data["mcpServers"].update(mcp_entry)

    with open(gm_config_path, "w", encoding="utf-8") as f:
        json.dump(gm_data, f, indent=2)
    print(f"  Updated: {gm_config_path}")


def generate_generic(config: Dict[str, Any]):
    """Generate generic skill."""
    platforms = config.get("platforms", {})
    generic = platforms.get("generic", {})
    skill_path = generic.get("skill_path", "~/.skills/mlsyseng")

    template = config.get("skill_template", "# MLSysEng MoE Skill")
    write_skill_md(skill_path, template)


def main():
    """Generate skills for all platforms."""
    config_path = "skills.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path) if not os.path.isabs(config_path) else config_path

    print(f"Loading config from: {full_path}")
    config = load_skills_config(full_path)

    print(f"\nGenerating skills for: {config.get('name', 'unknown')}")
    print(f"Version: {config.get('version', '?')}\n")

    generators = {
        "OpenClaw": generate_openclaw,
        "Claude Desktop": generate_claude_desktop,
        "Cursor": generate_cursor,
        "Gemini": generate_gemini,
        "Generic": generate_generic,
    }

    for platform, gen_fn in generators.items():
        print(f"[{platform}]")
        try:
            gen_fn(config)
        except Exception as e:
            print(f"  Warning: {e}")
        print()

    print("Done! Restart your AI clients to pick up the new skills.")


if __name__ == "__main__":
    main()
