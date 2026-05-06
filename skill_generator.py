"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    """Load the skills.yaml source of truth."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    """Expand ~ and env vars in a path."""
    return os.path.expanduser(os.path.expandvars(p))


def get_mcp_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the MCP server config block."""
    server = config["mcp_server"]
    env = {}
    for k, v in server.get("env", {}).items():
        env[k] = expand_path(v)

    return {
        "command": server["command"],
        "args": server["args"],
        "env": env,
    }


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate the SKILL.md content."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_openclaw(config: Dict[str, Any]):
    """Install skill for OpenClaw platform."""
    platform = config["platforms"]["openclaw"]
    skill_dir = Path(expand_path(platform["skill_dir"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  [openclaw] Wrote {skill_md}")

    config_file = Path(expand_path(platform["config_file"]))
    if config_file.exists():
        with open(config_file) as f:
            openclaw_config = json.load(f)
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        openclaw_config = {}

    key = platform["config_key"]
    if key not in openclaw_config:
        openclaw_config[key] = {}

    openclaw_config[key]["mlsyseng-moe"] = get_mcp_server_config(config)

    with open(config_file, "w") as f:
        json.dump(openclaw_config, f, indent=2)
    print(f"  [openclaw] Updated {config_file}")


def install_claude_desktop(config: Dict[str, Any]):
    """Install skill for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_file = Path(expand_path(platform["config_file"]))

    if config_file.exists():
        with open(config_file) as f:
            desktop_config = json.load(f)
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        desktop_config = {}

    key = platform["config_key"]
    if key not in desktop_config:
        desktop_config[key] = {}

    desktop_config[key]["mlsyseng-moe"] = get_mcp_server_config(config)

    with open(config_file, "w") as f:
        json.dump(desktop_config, f, indent=2)
    print(f"  [claude_desktop] Updated {config_file}")


def install_cursor(config: Dict[str, Any]):
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_dir = Path(expand_path(platform["skill_dir"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  [cursor] Wrote {skill_md}")


def install_gemini(config: Dict[str, Any]):
    """Install skill for Gemini."""
    platform = config["platforms"]["gemini"]
    config_file = Path(expand_path(platform["config_file"]))

    if config_file.exists():
        with open(config_file) as f:
            gemini_config = json.load(f)
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        gemini_config = {}

    key = platform["config_key"]
    if key not in gemini_config:
        gemini_config[key] = {}

    gemini_config[key]["mlsyseng-moe"] = get_mcp_server_config(config)

    with open(config_file, "w") as f:
        json.dump(gemini_config, f, indent=2)
    print(f"  [gemini] Updated {config_file}")


def install_generic(config: Dict[str, Any]):
    """Install skill to generic location."""
    platform = config["platforms"]["generic"]
    skill_dir = Path(expand_path(platform["skill_dir"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(generate_skill_md(config))
    print(f"  [generic] Wrote {skill_md}")


def main():
    """Generate and install skills for all platforms."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.yaml")
    config = load_skills_yaml(yaml_path)

    print(f"MLSysEng MoE Skill Generator v{config['version']}")
    print(f"Description: {config['description'].strip()}")
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
            print(f"  [WARN] {platform_name}: {e}")

    print()
    print("Done! Restart AI clients to pick up new skills.")


if __name__ == "__main__":
    main()
