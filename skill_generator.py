#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Reads the canonical skills.yaml and writes:
  - OpenClaw: SKILL.md + openclaw.json MCP entry
  - Claude Desktop: claude_desktop_config.json MCP entry
  - Cursor: SKILL.md
  - Gemini: mcp_config.json MCP entry
  - Generic: SKILL.md
"""

import json
import os
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SKILLS_YAML = SCRIPT_DIR / "skills.yaml"


def load_skills() -> dict:
    with open(SKILLS_YAML) as f:
        return yaml.safe_load(f)


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def _generate_skill_md(config: dict) -> str:
    """Generate a Markdown skill file from the config."""
    tools_section = ""
    for tool in config.get("tools", []):
        params = tool.get("parameters", {})
        param_lines = ""
        for pname, pinfo in params.items():
            param_lines += (
                f"  - `{pname}` ({pinfo.get('type', 'string')}): "
                f"{pinfo.get('description', '')} "
                f"(default: {pinfo.get('default', 'required')})\n"
            )
        tools_section += f"### `{tool['name']}`\n\n{tool.get('description', '')}\n\n"
        if param_lines:
            tools_section += f"**Parameters:**\n{param_lines}\n"
        tools_section += "\n"

    return f"""# {config['name']}

{config['description']}

## Version

{config.get('version', '0.1.0')}

## MCP Server

Start the server:

```bash
{config['mcp_server']['command']} {' '.join(config['mcp_server']['args'])}
```

## Tools

{tools_section}

## Quick Start

1. Extract knowledge from PDFs:
   ```
   extract-knowledge(force_reindex=false)
   ```

2. List available experts:
   ```
   list-experts()
   ```

3. Build a competition entry:
   ```
   build-entry(competition="titanic")
   ```

4. Run the convergence loop:
   ```
   evolve(competition="titanic")
   ```
"""


def _mcp_entry(config: dict) -> dict:
    """Build MCP server configuration entry."""
    server = config["mcp_server"]
    return {
        "command": server["command"],
        "args": server["args"],
        "env": {k: _expand(v) for k, v in server.get("env", {}).items()},
    }


def install_openclaw(config: dict) -> None:
    plat = config["platforms"]["openclaw"]
    skill_dir = Path(_expand(plat["skill_dir"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    md = _generate_skill_md(config)
    skill_path = skill_dir / plat["skill_file"]
    skill_path.write_text(md)
    print(f"  [openclaw] Wrote {skill_path}")

    config_path = Path(_expand(plat["config_file"]))
    if config_path.exists():
        with open(config_path) as f:
            oc_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        oc_config = {}

    oc_config.setdefault("mcpServers", {})
    oc_config["mcpServers"]["mlsyseng-moe"] = _mcp_entry(config)

    with open(config_path, "w") as f:
        json.dump(oc_config, f, indent=2)
    print(f"  [openclaw] Updated {config_path}")


def install_claude_desktop(config: dict) -> None:
    plat = config["platforms"]["claude_desktop"]
    config_path = Path(_expand(plat["config_file"]))

    if config_path.exists():
        with open(config_path) as f:
            cd_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cd_config = {}

    cd_config.setdefault("mcpServers", {})
    cd_config["mcpServers"]["mlsyseng-moe"] = _mcp_entry(config)

    with open(config_path, "w") as f:
        json.dump(cd_config, f, indent=2)
    print(f"  [claude_desktop] Updated {config_path}")


def install_cursor(config: dict) -> None:
    plat = config["platforms"]["cursor"]
    skill_dir = Path(_expand(plat["skill_dir"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    md = _generate_skill_md(config)
    skill_path = skill_dir / plat["skill_file"]
    skill_path.write_text(md)
    print(f"  [cursor] Wrote {skill_path}")


def install_gemini(config: dict) -> None:
    plat = config["platforms"]["gemini"]
    config_path = Path(_expand(plat["config_file"]))

    if config_path.exists():
        with open(config_path) as f:
            gem_config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        gem_config = {}

    gem_config.setdefault("mcpServers", {})
    gem_config["mcpServers"]["mlsyseng-moe"] = _mcp_entry(config)

    with open(config_path, "w") as f:
        json.dump(gem_config, f, indent=2)
    print(f"  [gemini] Updated {config_path}")


def install_generic(config: dict) -> None:
    plat = config["platforms"]["generic"]
    skill_dir = Path(_expand(plat["skill_dir"]))
    skill_dir.mkdir(parents=True, exist_ok=True)

    md = _generate_skill_md(config)
    skill_path = skill_dir / plat["skill_file"]
    skill_path.write_text(md)
    print(f"  [generic] Wrote {skill_path}")


def main():
    config = load_skills()
    print(f"Generating skills for {config['name']} v{config.get('version', '?')}")
    print()

    installers = {
        "openclaw": install_openclaw,
        "claude_desktop": install_claude_desktop,
        "cursor": install_cursor,
        "gemini": install_gemini,
        "generic": install_generic,
    }

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(installers.keys())

    for target in targets:
        if target in installers:
            try:
                installers[target](config)
            except Exception as e:
                print(f"  [{target}] Error: {e}")
        else:
            print(f"  Unknown platform: {target}")

    print()
    print("Done. Restart AI clients to pick up changes.")


if __name__ == "__main__":
    main()
