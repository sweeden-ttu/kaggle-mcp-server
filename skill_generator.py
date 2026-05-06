"""Skill generator for MLSysEng MoE system.

Reads skills.yaml and generates platform-specific configurations
for OpenClaw, Claude Desktop, Cursor, Gemini, and generic platforms.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_skills_config(config_path: str = "skills.yaml") -> Dict[str, Any]:
    """Load the skills.yaml configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> str:
    """Expand ~ and environment variables in paths."""
    return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())


def generate_skill_md(config: Dict[str, Any]) -> str:
    """Generate SKILL.md content for skill-based platforms."""
    tools_section = ""
    for tool in config.get("tools", []):
        params_str = ""
        if "params" in tool:
            param_lines = []
            for pname, pinfo in tool["params"].items():
                req = " (required)" if pinfo.get("required") else ""
                default = f" [default: {pinfo.get('default')}]" if "default" in pinfo else ""
                param_lines.append(f"  - `{pname}`: {pinfo.get('description', pinfo.get('type', ''))}{req}{default}")
            params_str = "\n" + "\n".join(param_lines)

        tools_section += f"\n### `{tool['name']}`\n{tool['description']}{params_str}\n"

    defaults = config.get("expert_defaults", {})
    loop_cfg = defaults.get("loop_config", {})

    return f"""# MLSysEng MoE - Machine Learning Systems Expert Mixture of Experts

{config.get('description', '')}

## MCP Server

This skill provides an MCP server with tools for:
- Knowledge extraction from ML Principles PDFs
- Expert registration with skills, strategies, and formulas
- Kaggle competition entry building with RAG-informed skill selection
- State convergence loops with L2 norm exit conditions

## Tools
{tools_section}
## Expert System

Each ML Principles chapter becomes an expert with:
- **Capabilities**: Inferred from chapter content
- **Skills**: Mapped Kaggle skill paths
- **Strategy**: `{defaults.get('strategy', 'N/A')}`
- **Formula**: `{defaults.get('formula', {}).get('function', 'N/A')}`
- **Loop Config**: Exit when `{loop_cfg.get('exit_condition', 'N/A')}` (epsilon={loop_cfg.get('epsilon', 0.001)})

## Quick Start

1. Extract knowledge: `extract-knowledge(force_reindex=false)`
2. List experts: `list-experts()`
3. Build entry: `build-entry(competition="titanic")`
4. Evolve: `evolve(competition="titanic")`

## Configuration

| Variable | Description |
|----------|-------------|
| `ML_PRINCIPLES_PATH` | Path to PDF chapters |
| `SQLITE_DB_PATH` | SQLite database path |
| `CHROMA_DB_PATH` | ChromaDB vector store path |
| `KAGGLE_SKILLS_PATH` | Kaggle skills directory |
"""


def generate_mcp_config_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate MCP server configuration entry for JSON configs."""
    server_cfg = config.get("mcp_server", {})
    return {
        "command": server_cfg.get("command", "python"),
        "args": server_cfg.get("args", ["-m", "mlsyseng_mcp.server"]),
        "env": {
            k: os.path.expandvars(v.replace("${", "").split(":-")[1].rstrip("}") if ":-" in v else v)
            for k, v in server_cfg.get("env", {}).items()
        },
    }


def install_openclaw(config: Dict[str, Any]):
    """Install skill for OpenClaw platform."""
    platform = config["skill_platforms"]["openclaw"]
    install_path = expand_path(platform["install_path"])

    Path(install_path).mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    skill_file = Path(install_path) / platform["skill_file"]
    skill_file.write_text(skill_md)
    print(f"  ✓ OpenClaw skill: {skill_file}")

    config_file = expand_path(platform["config_file"])
    if Path(config_file).exists():
        with open(config_file) as f:
            openclaw_config = json.load(f)
    else:
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        openclaw_config = {}

    if "mcpServers" not in openclaw_config:
        openclaw_config["mcpServers"] = {}

    openclaw_config["mcpServers"]["mlsyseng-moe"] = generate_mcp_config_entry(config)

    with open(config_file, "w") as f:
        json.dump(openclaw_config, f, indent=2)
    print(f"  ✓ OpenClaw config: {config_file}")


def install_cursor(config: Dict[str, Any]):
    """Install skill for Cursor platform."""
    platform = config["skill_platforms"]["cursor"]
    install_path = expand_path(platform["install_path"])

    Path(install_path).mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    skill_file = Path(install_path) / platform["skill_file"]
    skill_file.write_text(skill_md)
    print(f"  ✓ Cursor skill: {skill_file}")


def install_claude_desktop(config: Dict[str, Any]):
    """Install MCP server config for Claude Desktop."""
    platform = config["skill_platforms"]["claude_desktop"]
    config_file = expand_path(platform["config_file"])

    if Path(config_file).exists():
        with open(config_file) as f:
            claude_config = json.load(f)
    else:
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        claude_config = {}

    if "mcpServers" not in claude_config:
        claude_config["mcpServers"] = {}

    claude_config["mcpServers"]["mlsyseng-moe"] = generate_mcp_config_entry(config)

    with open(config_file, "w") as f:
        json.dump(claude_config, f, indent=2)
    print(f"  ✓ Claude Desktop config: {config_file}")


def install_gemini(config: Dict[str, Any]):
    """Install MCP server config for Gemini."""
    platform = config["skill_platforms"]["gemini"]
    config_file = expand_path(platform["config_file"])

    if Path(config_file).exists():
        with open(config_file) as f:
            gemini_config = json.load(f)
    else:
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        gemini_config = {}

    if "mcpServers" not in gemini_config:
        gemini_config["mcpServers"] = {}

    gemini_config["mcpServers"]["mlsyseng-moe"] = generate_mcp_config_entry(config)

    with open(config_file, "w") as f:
        json.dump(gemini_config, f, indent=2)
    print(f"  ✓ Gemini config: {config_file}")


def install_generic(config: Dict[str, Any]):
    """Install skill for generic platforms."""
    platform = config["skill_platforms"]["generic"]
    install_path = expand_path(platform["install_path"])

    Path(install_path).mkdir(parents=True, exist_ok=True)

    skill_md = generate_skill_md(config)
    skill_file = Path(install_path) / platform["skill_file"]
    skill_file.write_text(skill_md)
    print(f"  ✓ Generic skill: {skill_file}")


def main():
    """Generate and install skills for all platforms."""
    config_path = Path(__file__).parent / "skills.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    config = load_skills_config(str(config_path))
    print(f"MLSysEng MoE Skill Generator v{config.get('version', '0.1.0')}")
    print("=" * 50)

    platforms = [
        ("OpenClaw", install_openclaw),
        ("Cursor", install_cursor),
        ("Claude Desktop", install_claude_desktop),
        ("Gemini", install_gemini),
        ("Generic", install_generic),
    ]

    for name, installer in platforms:
        print(f"\nInstalling for {name}...")
        try:
            installer(config)
        except Exception as e:
            print(f"  ⚠ {name} installation failed: {e}")

    print("\n" + "=" * 50)
    print("Done! Restart AI clients to load the new skills.")


if __name__ == "__main__":
    main()
