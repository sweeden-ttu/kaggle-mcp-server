"""Generate platform-specific skill configurations from skills.yaml.

Reads the canonical skills.yaml and writes out:
  - OpenClaw:       ~/.openclaw/workspace/skills/mlsyseng/SKILL.md
  - Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
  - Cursor:         ~/.cursor/skills/mlsyseng/SKILL.md
  - Gemini:         ~/.gemini/mcp_config.json
  - Generic:        ~/.skills/mlsyseng/SKILL.md
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_skills_yaml(path: str = "skills.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> Path:
    p = Path(os.path.expanduser(path))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _expand(val: str) -> str:
    return os.path.expanduser(os.path.expandvars(val))


def _write_skill_md(target_dir: str, skill_content: str, name: str):
    """Write SKILL.md to the given directory."""
    d = _ensure_dir(target_dir)
    skill_path = d / "SKILL.md"
    skill_path.write_text(skill_content)
    print(f"  ✓ {name}: {skill_path}")


def _build_mcp_server_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single MCP server config entry from skills.yaml mcp_server block."""
    mcp = config.get("mcp_server", {})
    entry = {
        "command": mcp.get("command", "python"),
        "args": mcp.get("args", ["-m", "mlsyseng_mcp.server"]),
    }
    env = mcp.get("env", {})
    if env:
        resolved_env = {}
        for k, v in env.items():
            resolved_env[k] = _expand(str(v))
        entry["env"] = resolved_env
    return entry


def _update_json_config(
    config_path: str,
    config_key: str,
    server_name: str,
    server_entry: Dict[str, Any],
    platform_name: str,
):
    """Update a JSON config file's mcpServers section."""
    path = Path(_expand(config_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    if config_key not in existing:
        existing[config_key] = {}
    existing[config_key][server_name] = server_entry

    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"  ✓ {platform_name}: {path}")


def generate_openclaw(config: Dict[str, Any]):
    """Generate OpenClaw skill and update config."""
    platforms = config.get("platforms", {})
    oc = platforms.get("openclaw", {})
    skill_path = oc.get("skill_path", "~/.openclaw/workspace/skills/mlsyseng/")
    skill_content = config.get("skill", {}).get("content", "")
    _write_skill_md(skill_path, skill_content, "OpenClaw")

    config_path = oc.get("config_path", "~/.openclaw/openclaw.json")
    config_key = oc.get("config_key", "mcpServers")
    server_entry = _build_mcp_server_entry(config)
    _update_json_config(config_path, config_key, "mlsyseng-mcp", server_entry, "OpenClaw config")


def generate_claude_desktop(config: Dict[str, Any]):
    """Update Claude Desktop MCP config."""
    platforms = config.get("platforms", {})
    cd = platforms.get("claude_desktop", {})
    config_path = cd.get(
        "config_path",
        "~/Library/Application Support/Claude/claude_desktop_config.json",
    )
    config_key = cd.get("config_key", "mcpServers")
    server_entry = _build_mcp_server_entry(config)
    _update_json_config(config_path, config_key, "mlsyseng-mcp", server_entry, "Claude Desktop")


def generate_cursor(config: Dict[str, Any]):
    """Generate Cursor skill."""
    platforms = config.get("platforms", {})
    cs = platforms.get("cursor", {})
    skill_path = cs.get("skill_path", "~/.cursor/skills/mlsyseng/")
    skill_content = config.get("skill", {}).get("content", "")
    _write_skill_md(skill_path, skill_content, "Cursor")


def generate_gemini(config: Dict[str, Any]):
    """Update Gemini MCP config."""
    platforms = config.get("platforms", {})
    gm = platforms.get("gemini", {})
    config_path = gm.get("config_path", "~/.gemini/mcp_config.json")
    config_key = gm.get("config_key", "mcpServers")
    server_entry = _build_mcp_server_entry(config)
    _update_json_config(config_path, config_key, "mlsyseng-mcp", server_entry, "Gemini")


def generate_generic(config: Dict[str, Any]):
    """Generate generic skill."""
    platforms = config.get("platforms", {})
    gn = platforms.get("generic", {})
    skill_path = gn.get("skill_path", "~/.skills/mlsyseng/")
    skill_content = config.get("skill", {}).get("content", "")
    _write_skill_md(skill_path, skill_content, "Generic")


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    script_dir = Path(__file__).parent
    full_path = script_dir / yaml_path if not Path(yaml_path).is_absolute() else Path(yaml_path)

    if not full_path.exists():
        print(f"Error: {full_path} not found")
        sys.exit(1)

    config = _load_skills_yaml(str(full_path))
    print(f"Generating skills from: {full_path}")
    print(f"  Name: {config.get('name', 'unknown')}")
    print(f"  Version: {config.get('version', 'unknown')}")
    print()

    generators = [
        ("OpenClaw", generate_openclaw),
        ("Claude Desktop", generate_claude_desktop),
        ("Cursor", generate_cursor),
        ("Gemini", generate_gemini),
        ("Generic", generate_generic),
    ]

    for name, gen_fn in generators:
        try:
            gen_fn(config)
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\nDone! Restart AI clients to pick up the new configuration.")


if __name__ == "__main__":
    main()
