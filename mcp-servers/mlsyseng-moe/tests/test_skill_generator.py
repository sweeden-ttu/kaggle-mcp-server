"""Tests for the skill generator."""

import json
import os
import tempfile

import pytest

import skill_generator as sg


def test_load_skills_yaml():
    config = sg.load_skills_yaml()
    assert config["name"] == "mlsyseng-moe"
    assert "tools" in config
    assert "platforms" in config
    assert len(config["tools"]) > 0


def test_generate_skill_md():
    config = sg.load_skills_yaml()
    md = sg.generate_skill_md(config)
    assert "# mlsyseng-moe" in md
    assert "extract-knowledge" in md
    assert "Convergence Loop" in md


def test_install_generic():
    config = sg.load_skills_yaml()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = dict(config)
        config["platforms"] = dict(config["platforms"])
        config["platforms"]["generic"] = {"skill_path": os.path.join(tmpdir, "skills/")}
        sg.install_generic(config)
        skill_path = os.path.join(tmpdir, "skills", "SKILL.md")
        assert os.path.exists(skill_path)
        content = open(skill_path).read()
        assert "mlsyseng-moe" in content
