"""Generate platform-specific skill configs from skills.yaml.

Reads the source-of-truth skills.yaml and installs skill definitions
and MCP server configs for OpenClaw, Claude Desktop, Cursor, Gemini,
and a generic fallback.
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


def write_skill_md(target_dir: str, content: str):
    """Write SKILL.md to the target directory."""
    target = Path(expand_path(target_dir))
    target.mkdir(parents=True, exist_ok=True)
    skill_file = target / "SKILL.md"
    skill_file.write_text(content)
    print(f"  Wrote {skill_file}")


def update_json_config(config_path: str, server_name: str, server_config: dict):
    """Update a JSON config file with MCP server entry."""
    path = Path(expand_path(config_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if path.exists():
        try:
            config = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"][server_name] = server_config

    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Updated {path}")


def build_mcp_server_config(skills: dict) -> dict:
    """Build the MCP server configuration dict."""
    mcp = skills["mcp_server"]
    env = {}
    for k, v in mcp.get("env", {}).items():
        env[k] = expand_path(v.split(":-")[1].rstrip("}")) if ":-" in v else v

    return {
        "command": mcp["command"],
        "args": mcp["args"],
        "env": env,
    }


def install_openclaw(skills: dict, server_config: dict):
    """Install skill and config for OpenClaw."""
    print("\nOpenClaw:")
    platform = skills["platforms"]["openclaw"]

    write_skill_md(platform["skill_path"], skills["skill_template"])

    update_json_config(
        platform["config_path"],
        skills["name"],
        server_config,
    )


def install_claude_desktop(skills: dict, server_config: dict):
    """Install config for Claude Desktop."""
    print("\nClaude Desktop:")
    platform = skills["platforms"]["claude_desktop"]

    update_json_config(
        platform["config_path"],
        skills["name"],
        server_config,
    )


def install_cursor(skills: dict):
    """Install skill for Cursor."""
    print("\nCursor:")
    platform = skills["platforms"]["cursor"]
    write_skill_md(platform["skill_path"], skills["skill_template"])


def install_gemini(skills: dict, server_config: dict):
    """Install config for Gemini."""
    print("\nGemini:")
    platform = skills["platforms"]["gemini"]

    update_json_config(
        platform["config_path"],
        skills["name"],
        server_config,
    )


def install_generic(skills: dict):
    """Install skill for generic consumers."""
    print("\nGeneric:")
    platform = skills["platforms"]["generic"]
    write_skill_md(platform["skill_path"], skills["skill_template"])


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print(f"Loading {yaml_path}...")
    skills = load_skills_yaml(yaml_path)
    server_config = build_mcp_server_config(skills)

    print(f"\nInstalling skills for: {skills['name']} v{skills['version']}")

    install_openclaw(skills, server_config)
    install_claude_desktop(skills, server_config)
    install_cursor(skills)
    install_gemini(skills, server_config)
    install_generic(skills)

    print(f"\nDone! Installed {skills['name']} to all platforms.")
    print("Restart your AI clients to pick up the changes.")


if __name__ == "__main__":
    main()
