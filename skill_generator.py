"""Generate platform-specific skill configurations from skills.yaml."""

import os
import json
import sys
from pathlib import Path

import yaml


SKILLS_YAML_PATH = Path(__file__).parent / "skills.yaml"


def load_skills_config() -> dict:
    """Load the skills.yaml configuration."""
    with open(SKILLS_YAML_PATH) as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in a path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def generate_skill_md(config: dict) -> str:
    """Generate the SKILL.md content from config."""
    return config.get("skill_content", "# MLSysEng MoE\n\nNo skill content defined.")


def install_openclaw(config: dict) -> None:
    """Install skill for OpenClaw (Claude Code)."""
    platform = config["platforms"]["openclaw"]
    skill_path = expand_path(platform["skill_path"])
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = skill_path / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  [openclaw] Installed SKILL.md to {skill_path}")

    config_path = expand_path(platform["config_path"])
    if config_path.exists():
        try:
            with open(config_path) as f:
                openclaw_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            openclaw_config = {}

        if "mcpServers" not in openclaw_config:
            openclaw_config["mcpServers"] = {}

        mcp_cfg = config["mcp_server"]
        openclaw_config["mcpServers"]["mlsyseng-moe"] = {
            "command": mcp_cfg["command"],
            "args": mcp_cfg["args"],
            "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
        }

        with open(config_path, "w") as f:
            json.dump(openclaw_config, f, indent=2)
        print(f"  [openclaw] Updated MCP config in {config_path}")


def install_claude_desktop(config: dict) -> None:
    """Install skill for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = expand_path(platform["config_path"])

    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if config_path.exists():
            with open(config_path) as f:
                desktop_config = json.load(f)
        else:
            desktop_config = {}
    except (json.JSONDecodeError, FileNotFoundError):
        desktop_config = {}

    if "mcpServers" not in desktop_config:
        desktop_config["mcpServers"] = {}

    mcp_cfg = config["mcp_server"]
    desktop_config["mcpServers"]["mlsyseng-moe"] = {
        "command": mcp_cfg["command"],
        "args": mcp_cfg["args"],
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    with open(config_path, "w") as f:
        json.dump(desktop_config, f, indent=2)
    print(f"  [claude_desktop] Updated config at {config_path}")


def install_cursor(config: dict) -> None:
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_path = expand_path(platform["skill_path"])
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = skill_path / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  [cursor] Installed SKILL.md to {skill_path}")


def install_gemini(config: dict) -> None:
    """Install skill for Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = expand_path(platform["config_path"])

    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if config_path.exists():
            with open(config_path) as f:
                gemini_config = json.load(f)
        else:
            gemini_config = {}
    except (json.JSONDecodeError, FileNotFoundError):
        gemini_config = {}

    if "mcpServers" not in gemini_config:
        gemini_config["mcpServers"] = {}

    mcp_cfg = config["mcp_server"]
    gemini_config["mcpServers"]["mlsyseng-moe"] = {
        "command": mcp_cfg["command"],
        "args": mcp_cfg["args"],
        "env": {k: os.path.expandvars(v) for k, v in mcp_cfg.get("env", {}).items()},
    }

    with open(config_path, "w") as f:
        json.dump(gemini_config, f, indent=2)
    print(f"  [gemini] Updated config at {config_path}")


def install_generic(config: dict) -> None:
    """Install skill to generic location."""
    platform = config["platforms"]["generic"]
    skill_path = expand_path(platform["skill_path"])
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_md = skill_path / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  [generic] Installed SKILL.md to {skill_path}")


def main():
    """Generate and install skills for all platforms."""
    print(f"Loading skills config from {SKILLS_YAML_PATH}")
    config = load_skills_config()

    print(f"\nInstalling {config['name']} v{config['version']}")
    print(f"Description: {config['description'].strip()[:80]}...")
    print()

    installers = {
        "openclaw": install_openclaw,
        "claude_desktop": install_claude_desktop,
        "cursor": install_cursor,
        "gemini": install_gemini,
        "generic": install_generic,
    }

    for platform_name, installer in installers.items():
        try:
            installer(config)
        except Exception as e:
            print(f"  [{platform_name}] Failed: {e}")

    print("\nDone! Restart AI clients to pick up new configuration.")


if __name__ == "__main__":
    main()
