"""Generate platform-specific skill configs from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def write_skill_md(target_dir: str, content: str) -> str:
    """Write SKILL.md to a target directory."""
    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    skill_path = target / "SKILL.md"
    skill_path.write_text(content)
    return str(skill_path)


def generate_openclaw(config: Dict[str, Any]) -> None:
    """Generate OpenClaw skill and update config."""
    platforms = config.get("platforms", {})
    openclaw = platforms.get("openclaw", {})

    skill_path = openclaw.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/")
    path = write_skill_md(skill_path, config.get("skill_content", ""))
    print(f"  OpenClaw skill: {path}")

    config_path = Path(openclaw.get("config_path", "~/.openclaw/openclaw.json")).expanduser()
    if config_path.exists():
        try:
            with open(config_path) as f:
                oc_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            oc_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    mcp_servers = oc_config.setdefault("mcpServers", {})
    mcp_cfg = config.get("mcp_server", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg.get("command", "python"),
        "args": mcp_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  OpenClaw config: {config_path}")


def generate_claude_desktop(config: Dict[str, Any]) -> None:
    """Update Claude Desktop MCP config."""
    platforms = config.get("platforms", {})
    claude = platforms.get("claude_desktop", {})

    config_path = Path(
        claude.get(
            "config_path",
            "~/Library/Application Support/Claude/claude_desktop_config.json",
        )
    ).expanduser()

    if config_path.exists():
        try:
            with open(config_path) as f:
                cd_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            cd_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    mcp_servers = cd_config.setdefault("mcpServers", {})
    mcp_cfg = config.get("mcp_server", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg.get("command", "python"),
        "args": mcp_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  Claude Desktop config: {config_path}")


def generate_cursor(config: Dict[str, Any]) -> None:
    """Generate Cursor skill."""
    platforms = config.get("platforms", {})
    cursor = platforms.get("cursor", {})

    skill_path = cursor.get("skill_path", "~/.cursor/skills/mlsyseng/")
    path = write_skill_md(skill_path, config.get("skill_content", ""))
    print(f"  Cursor skill: {path}")


def generate_gemini(config: Dict[str, Any]) -> None:
    """Update Gemini MCP config."""
    platforms = config.get("platforms", {})
    gemini = platforms.get("gemini", {})

    config_path = Path(
        gemini.get("config_path", "~/.gemini/mcp_config.json")
    ).expanduser()

    if config_path.exists():
        try:
            with open(config_path) as f:
                gm_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            gm_config = {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gm_config = {}

    mcp_servers = gm_config.setdefault("mcpServers", {})
    mcp_cfg = config.get("mcp_server", {})
    mcp_servers["mlsyseng-moe"] = {
        "command": mcp_cfg.get("command", "python"),
        "args": mcp_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    with open(config_path, "w") as f:
        json.dump(gm_config, f, indent=2)
    print(f"  Gemini config: {config_path}")


def generate_generic(config: Dict[str, Any]) -> None:
    """Generate generic skill."""
    platforms = config.get("platforms", {})
    generic = platforms.get("generic", {})

    skill_path = generic.get("skill_path", "~/.skills/mlsyseng/")
    path = write_skill_md(skill_path, config.get("skill_content", ""))
    print(f"  Generic skill: {path}")


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    print(f"Loading skills from {yaml_path}...")
    config = load_skills_yaml(yaml_path)

    print(f"\nGenerating skills for: {config.get('name', 'mlsyseng-moe')}")
    print(f"Version: {config.get('version', '0.1.0')}\n")

    generators = [
        ("OpenClaw", generate_openclaw),
        ("Claude Desktop", generate_claude_desktop),
        ("Cursor", generate_cursor),
        ("Gemini", generate_gemini),
        ("Generic", generate_generic),
    ]

    for name, gen_fn in generators:
        print(f"[{name}]")
        try:
            gen_fn(config)
        except Exception as e:
            print(f"  Warning: {e}")
        print()

    print("Done! Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
