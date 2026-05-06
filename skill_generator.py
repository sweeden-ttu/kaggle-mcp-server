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


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate a SKILL.md file content from skills.yaml config."""
    lines = [
        f"# {config['name']}",
        "",
        config["description"].strip(),
        "",
        "## MCP Tools",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]
    for skill in config.get("skills", []):
        lines.append(f"| `{skill['tool']}` | {skill['description'].strip()} |")

    lines.extend([
        "",
        "## Quick Start",
        "",
        "### Extract Knowledge",
        "```",
        "extract_knowledge(force_reindex=false)",
        "```",
        "",
        "### List Experts",
        "```",
        "list_experts()",
        "```",
        "",
        "### Build Competition Entry",
        "```",
        "build_entry(competition=\"titanic\")",
        "```",
        "",
        "### Run Convergence Loop",
        "```",
        "evolve(competition=\"titanic\")",
        "```",
        "",
        "## Expert Definition",
        "",
        "Each chapter expert has capabilities, skills, strategy, formula,",
        "and a convergence loop config with epsilon-based exit condition.",
        "",
        "## Configuration",
        "",
        "| Variable | Description |",
        "|----------|-------------|",
    ])

    env = config.get("mcp_server", {}).get("env", {})
    for var, val in env.items():
        lines.append(f"| `{var}` | Default: `{val}` |")

    return "\n".join(lines) + "\n"


def _mcp_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the MCP server JSON config block."""
    server = config.get("mcp_server", {})
    return {
        "command": server.get("command", "python"),
        "args": server.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": server.get("env", {}),
    }


def install_openclaw(config: Dict[str, Any]):
    """Install skill to OpenClaw."""
    platforms = config.get("platforms", {})
    oc = platforms.get("openclaw", {})

    skill_dir = _expand(oc.get("skill_dir", "~/.openclaw/workspace/skills/mlsyseng"))
    _ensure_dir(skill_dir)

    skill_md = _generate_skill_md(config)
    skill_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_path, "w") as f:
        f.write(skill_md)
    print(f"  [openclaw] Wrote {skill_path}")

    config_path = _expand(oc.get("config_path", "~/.openclaw/openclaw.json"))
    if os.path.exists(config_path):
        with open(config_path) as f:
            oc_config = json.load(f)
    else:
        _ensure_dir(os.path.dirname(config_path))
        oc_config = {}

    if "mcpServers" not in oc_config:
        oc_config["mcpServers"] = {}

    oc_config["mcpServers"]["mlsyseng-moe"] = _mcp_server_config(config)

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  [openclaw] Updated {config_path}")


def install_claude_desktop(config: Dict[str, Any]):
    """Install skill to Claude Desktop."""
    platforms = config.get("platforms", {})
    cd = platforms.get("claude_desktop", {})

    config_path = _expand(
        cd.get("config_path", "~/Library/Application Support/Claude/claude_desktop_config.json")
    )

    if os.path.exists(config_path):
        with open(config_path) as f:
            cd_config = json.load(f)
    else:
        _ensure_dir(os.path.dirname(config_path))
        cd_config = {}

    if "mcpServers" not in cd_config:
        cd_config["mcpServers"] = {}

    cd_config["mcpServers"]["mlsyseng-moe"] = _mcp_server_config(config)

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  [claude_desktop] Updated {config_path}")


def install_cursor(config: Dict[str, Any]):
    """Install skill to Cursor."""
    platforms = config.get("platforms", {})
    cur = platforms.get("cursor", {})

    skill_dir = _expand(cur.get("skill_dir", "~/.cursor/skills/mlsyseng"))
    _ensure_dir(skill_dir)

    skill_md = _generate_skill_md(config)
    skill_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_path, "w") as f:
        f.write(skill_md)
    print(f"  [cursor] Wrote {skill_path}")


def install_gemini(config: Dict[str, Any]):
    """Install skill to Gemini."""
    platforms = config.get("platforms", {})
    gem = platforms.get("gemini", {})

    config_path = _expand(gem.get("config_path", "~/.gemini/mcp_config.json"))

    if os.path.exists(config_path):
        with open(config_path) as f:
            gem_config = json.load(f)
    else:
        _ensure_dir(os.path.dirname(config_path))
        gem_config = {}

    if "mcpServers" not in gem_config:
        gem_config["mcpServers"] = {}

    gem_config["mcpServers"]["mlsyseng-moe"] = _mcp_server_config(config)

    with open(config_path, "w") as f:
        json.dump(gem_config, f, indent=2)
    print(f"  [gemini] Updated {config_path}")


def install_generic(config: Dict[str, Any]):
    """Install skill to generic location."""
    platforms = config.get("platforms", {})
    gen = platforms.get("generic", {})

    skill_dir = _expand(gen.get("skill_dir", "~/.skills/mlsyseng"))
    _ensure_dir(skill_dir)

    skill_md = _generate_skill_md(config)
    skill_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_path, "w") as f:
        f.write(skill_md)
    print(f"  [generic] Wrote {skill_path}")


def main():
    yaml_path = "skills.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    config = load_skills_yaml(yaml_path)
    print(f"Generating skills for: {config['name']} v{config['version']}")
    print()

    installers = [
        ("OpenClaw", install_openclaw),
        ("Claude Desktop", install_claude_desktop),
        ("Cursor", install_cursor),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in installers:
        try:
            print(f"Installing {name}...")
            installer(config)
        except Exception as e:
            print(f"  [warning] {name} install failed: {e}")

    print()
    print("Done! Restart your AI clients to pick up the changes.")


if __name__ == "__main__":
    main()
