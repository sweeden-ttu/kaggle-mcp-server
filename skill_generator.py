"""Generate platform-specific skill configurations from skills.yaml.

Reads the unified skills.yaml and generates skill files for:
- OpenClaw
- Claude Desktop
- Cursor
- Gemini
- Generic (.skills/)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def load_skills_config(path: str = "skills.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def generate_skill_markdown(config: dict) -> str:
    """Generate a SKILL.md file from the skills config."""
    lines = [
        f"# {config['name']}",
        "",
        config.get("description", "").strip(),
        "",
        "## MCP Server",
        "",
        f"- **Server**: `{config['mcp_server']['name']}`",
        f"- **Command**: `{config['mcp_server']['command']} {' '.join(config['mcp_server']['args'])}`",
        "",
        "## Available Tools",
        "",
    ]

    for skill in config.get("skills", []):
        lines.append(f"### {skill['name']}")
        lines.append("")
        lines.append(skill.get("description", "").strip())
        lines.append("")
        if skill.get("usage"):
            lines.append("**Usage:**")
            lines.append("")
            lines.append(skill["usage"].strip())
            lines.append("")
        if skill.get("examples"):
            lines.append("**Examples:**")
            lines.append("")
            for ex in skill["examples"]:
                lines.append(f"```python\n{ex}\n```")
                lines.append("")

    lines.append("## Environment Variables")
    lines.append("")
    lines.append("| Variable | Default | Description |")
    lines.append("|----------|---------|-------------|")
    for key, val in config.get("mcp_server", {}).get("env", {}).items():
        lines.append(f"| `{key}` | `{val}` | |")
    lines.append("")

    return "\n".join(lines)


def generate_mcp_config(config: dict) -> dict:
    """Generate MCP server configuration for JSON-based platforms."""
    server = config["mcp_server"]
    return {
        server["name"]: {
            "command": server["command"],
            "args": server["args"],
            "env": server.get("env", {}),
        }
    }


def write_skill_file(platform: str, config: dict) -> str | None:
    """Write SKILL.md to the platform's skill directory."""
    platforms = config.get("platforms", {})
    if platform not in platforms:
        return None

    pconfig = platforms[platform]
    skill_path = pconfig.get("skill_path")
    if not skill_path:
        return None

    target_dir = Path(expand_path(skill_path))
    target_dir.mkdir(parents=True, exist_ok=True)

    skill_file = pconfig.get("skill_file", "SKILL.md")
    target = target_dir / skill_file

    content = generate_skill_markdown(config)
    target.write_text(content)
    return str(target)


def update_json_config(platform: str, config: dict) -> str | None:
    """Update JSON config files for platforms that use them."""
    platforms = config.get("platforms", {})
    if platform not in platforms:
        return None

    pconfig = platforms[platform]
    config_path = pconfig.get("config_path")
    if not config_path:
        return None

    target = Path(expand_path(config_path))
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if target.exists():
        try:
            with open(target) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    config_key = pconfig.get("config_key", "mcpServers")
    if config_key not in existing:
        existing[config_key] = {}

    mcp_config = generate_mcp_config(config)
    existing[config_key].update(mcp_config)

    with open(target, "w") as f:
        json.dump(existing, f, indent=2)

    return str(target)


def generate_all(config_path: str = "skills.yaml") -> dict:
    """Generate skill files and configs for all platforms."""
    config = load_skills_config(config_path)
    results = {}

    for platform in ["openclaw", "cursor", "generic"]:
        path = write_skill_file(platform, config)
        if path:
            results[f"{platform}_skill"] = path

    for platform in ["openclaw", "claude_desktop", "gemini"]:
        path = update_json_config(platform, config)
        if path:
            results[f"{platform}_config"] = path

    return results


def main():
    print("MLSysEng MoE Skill Generator")
    print("=" * 40)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "skills.yaml"

    if not Path(config_path).exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    results = generate_all(config_path)

    print("\nGenerated files:")
    for key, path in sorted(results.items()):
        print(f"  {key}: {path}")

    print(f"\nTotal: {len(results)} files generated")


if __name__ == "__main__":
    main()
