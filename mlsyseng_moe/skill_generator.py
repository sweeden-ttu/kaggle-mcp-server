"""Generate platform-specific skill configurations from skills.yaml."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_yaml() -> Dict[str, Any]:
    """Load the skills.yaml source of truth."""
    yaml_path = Path(__file__).parent / "skills.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> str:
    """Expand ~ and environment variables in a path."""
    return os.path.expanduser(os.path.expandvars(path))


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate the SKILL.md content."""
    return config.get("skill_content", "# MLSysEng MoE Skill\n")


def install_openclaw(config: Dict[str, Any]):
    """Install skill for OpenClaw platform."""
    platform = config["platforms"]["openclaw"]
    skill_path = expand_path(platform["skill_path"])
    config_path = expand_path(platform["config_path"])

    Path(skill_path).mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(skill_md)
    print(f"  [OpenClaw] Wrote {skill_file}")

    mcp_config = {
        "mcpServers": {
            "mlsyseng-moe": {
                "command": config["mcp_server"]["command"],
                "args": config["mcp_server"]["args"],
                "env": {
                    k: expand_path(v) for k, v in config["mcp_server"]["env"].items()
                },
            }
        }
    }

    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = json.load(f)
        if "mcpServers" not in existing:
            existing["mcpServers"] = {}
        existing["mcpServers"]["mlsyseng-moe"] = mcp_config["mcpServers"]["mlsyseng-moe"]
        mcp_config = existing

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(mcp_config, f, indent=2)
    print(f"  [OpenClaw] Updated {config_path}")


def install_claude_desktop(config: Dict[str, Any]):
    """Install skill for Claude Desktop."""
    platform = config["platforms"]["claude_desktop"]
    config_path = expand_path(platform["config_path"])

    mcp_entry = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: expand_path(v) for k, v in config["mcp_server"]["env"].items()},
    }

    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    existing["mcpServers"]["mlsyseng-moe"] = mcp_entry

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  [Claude Desktop] Updated {config_path}")


def install_cursor(config: Dict[str, Any]):
    """Install skill for Cursor."""
    platform = config["platforms"]["cursor"]
    skill_path = expand_path(platform["skill_path"])

    Path(skill_path).mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(skill_md)
    print(f"  [Cursor] Wrote {skill_file}")


def install_gemini(config: Dict[str, Any]):
    """Install skill for Gemini."""
    platform = config["platforms"]["gemini"]
    config_path = expand_path(platform["config_path"])

    mcp_entry = {
        "command": config["mcp_server"]["command"],
        "args": config["mcp_server"]["args"],
        "env": {k: expand_path(v) for k, v in config["mcp_server"]["env"].items()},
    }

    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    existing["mcpServers"]["mlsyseng-moe"] = mcp_entry

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  [Gemini] Updated {config_path}")


def install_generic(config: Dict[str, Any]):
    """Install skill to generic location."""
    platform = config["platforms"]["generic"]
    skill_path = expand_path(platform["skill_path"])

    Path(skill_path).mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(skill_md)
    print(f"  [Generic] Wrote {skill_file}")


def main():
    """Generate and install skills for all platforms."""
    print("MLSysEng MoE - Skill Generator")
    print("=" * 40)

    config = load_skills_yaml()
    print(f"Loaded skills.yaml (v{config['version']})")
    print()

    installers = {
        "openclaw": install_openclaw,
        "claude_desktop": install_claude_desktop,
        "cursor": install_cursor,
        "gemini": install_gemini,
        "generic": install_generic,
    }

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        for name, installer in installers.items():
            print(f"Installing for {name}...")
            try:
                installer(config)
            except Exception as e:
                print(f"  [WARNING] Failed: {e}")
            print()
    elif target in installers:
        print(f"Installing for {target}...")
        installers[target](config)
    else:
        print(f"Unknown platform: {target}")
        print(f"Available: {', '.join(installers.keys())}, all")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
