"""Tests for skill_generator module."""

import json
import os

import pytest

from skill_generator import (
    load_skills_config,
    expand_path,
    generate_skill_md,
    build_mcp_server_entry,
    write_skill_file,
    update_json_config,
    generate_all,
)


@pytest.fixture
def config():
    return load_skills_config("skills.yaml")


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal skills.yaml for testing."""
    yaml_content = """
name: test-skill
version: "0.1.0"
description: Test skill
mcp_server:
  command: python
  args: ["-m", "test.server"]
  env:
    TEST_VAR: "test_value"
skill_content: |
  # Test Skill
  This is a test.
platforms:
  generic:
    skill_dir: "{skill_dir}"
    skill_file: "SKILL.md"
""".format(skill_dir=str(tmp_path / "skills"))
    yaml_path = tmp_path / "skills.yaml"
    yaml_path.write_text(yaml_content)
    return str(yaml_path)


class TestLoadSkillsConfig:
    def test_loads_from_file(self, config):
        assert config["name"] == "mlsyseng-moe"
        assert "mcp_server" in config
        assert "platforms" in config

    def test_has_tools(self, config):
        assert len(config["tools"]) >= 7


class TestExpandPath:
    def test_expands_tilde(self):
        result = expand_path("~/test")
        assert "~" not in str(result)

    def test_preserves_absolute(self):
        result = expand_path("/absolute/path")
        assert str(result) == "/absolute/path"


class TestGenerateSkillMd:
    def test_generates_content(self, config):
        content = generate_skill_md(config)
        assert "MLSysEng MoE" in content
        assert "extract-knowledge" in content


class TestBuildMcpServerEntry:
    def test_builds_entry(self, config):
        entry = build_mcp_server_entry(config)
        assert entry["command"] == "python"
        assert "-m" in entry["args"]
        assert "env" in entry


class TestWriteSkillFile:
    def test_writes_file(self, tmp_path):
        path = write_skill_file(tmp_path / "skills", "SKILL.md", "# Test")
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "# Test"

    def test_creates_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        path = write_skill_file(deep_path, "SKILL.md", "content")
        assert os.path.exists(path)


class TestUpdateJsonConfig:
    def test_creates_new_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        result = update_json_config(config_path, "mcpServers", {"test": True})
        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert "mlsyseng-moe" in data["mcpServers"]

    def test_updates_existing_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"mcpServers": {"other": {}}}))
        update_json_config(config_path, "mcpServers", {"test": True})
        data = json.loads(config_path.read_text())
        assert "other" in data["mcpServers"]
        assert "mlsyseng-moe" in data["mcpServers"]


class TestGenerateAll:
    def test_generates_for_minimal_config(self, minimal_config, tmp_path):
        results = generate_all(minimal_config)
        assert len(results) > 0
        assert any("generic" in k for k in results)
