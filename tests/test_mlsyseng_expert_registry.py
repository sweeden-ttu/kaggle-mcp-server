"""Tests for mlsyseng_mcp.expert_registry module."""

import os
import tempfile

import pytest

from mlsyseng_mcp import database as db
from mlsyseng_mcp import expert_registry


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    yield


class TestSlugify:
    def test_basic(self):
        assert expert_registry._slugify("08 ML Systems") == "08_ml_systems"

    def test_special_chars(self):
        slug = expert_registry._slugify("Chapter 1: Introduction (v2)")
        assert ":" not in slug
        assert "(" not in slug

    def test_dashes(self):
        slug = expert_registry._slugify("deep-learning-basics")
        assert slug == "deep_learning_basics"


class TestInferCapabilities:
    def test_with_optimization(self):
        concepts = [{"category": "optimization"}]
        caps = expert_registry._infer_capabilities(concepts)
        assert any("Optimize" in c for c in caps)

    def test_empty_concepts(self):
        caps = expert_registry._infer_capabilities([])
        assert len(caps) >= 1

    def test_multiple_categories(self):
        concepts = [
            {"category": "optimization"},
            {"category": "evaluation"},
        ]
        caps = expert_registry._infer_capabilities(concepts)
        assert len(caps) >= 2


class TestInferSkills:
    def test_returns_paths(self):
        concepts = [{"category": "optimization"}]
        skills = expert_registry._infer_skills(concepts)
        assert all(isinstance(s, str) for s in skills)
        assert len(skills) > 0

    def test_empty_concepts(self):
        skills = expert_registry._infer_skills([])
        assert len(skills) >= 2


class TestRegisterExperts:
    def test_no_chapters(self):
        result = expert_registry.register_experts_from_chapters()
        assert result == []

    def test_with_chapter(self):
        ch_id = db.upsert_chapter(
            "01_intro", "Introduction", "/a.pdf",
            "gradient descent and backpropagation", 5,
        )
        db.add_concepts(ch_id, [
            {"concept": "gradient descent", "category": "optimization"},
        ])
        result = expert_registry.register_experts_from_chapters()
        assert len(result) == 1
        assert result[0]["slug"] == "01_intro"

    def test_export_definitions(self):
        ch_id = db.upsert_chapter(
            "02_dl", "Deep Learning", "/b.pdf", "neural network content", 4,
        )
        db.add_concepts(ch_id, [
            {"concept": "neural network", "category": "architecture"},
        ])
        expert_registry.register_experts_from_chapters()
        exports = expert_registry.export_expert_definitions()
        assert len(exports) == 1
        assert "formula" in exports[0]
        assert "loop_config" in exports[0]


class TestGetExpertForQuery:
    def test_no_experts(self):
        assert expert_registry.get_expert_for_query("anything") is None

    def test_keyword_match(self):
        db.upsert_expert("ml_systems", "ML Systems", None,
                         ["Build baseline models"], [], "", {}, {})
        db.upsert_expert("deep_learning", "Deep Learning", None,
                         ["Design neural networks"], [], "", {}, {})
        result = expert_registry.get_expert_for_query("deep learning neural")
        assert result is not None
        assert result["slug"] == "deep_learning"
