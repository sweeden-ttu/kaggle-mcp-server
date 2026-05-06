"""Tests for mlsyseng_mcp.expert_registry module."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp.expert_registry import (
    build_expert_definition,
    infer_skills_from_concepts,
    register_experts_from_chapters,
    save_expert_json,
    load_expert_json,
    select_experts_for_competition,
    _slugify,
)


class TestSlugify:
    def test_basic_slugify(self):
        assert _slugify("08_ML Systems") == "08_ml_systems"

    def test_slugify_with_spaces(self):
        slug = _slugify("10 Deep Learning Fundamentals")
        assert "deep_learning_fundamentals" in slug

    def test_slugify_special_chars(self):
        slug = _slugify("03 - Feature Engineering!")
        assert "feature_engineering" in slug


class TestInferSkills:
    def test_known_concepts(self):
        concepts = ["neural network", "deep learning", "cross-validation"]
        skills = infer_skills_from_concepts(concepts, "/test/skills")
        assert len(skills) > 0
        assert any("model-trainer" in s for s in skills)

    def test_empty_concepts(self):
        skills = infer_skills_from_concepts([], "/test/skills")
        assert skills == []

    def test_unknown_concepts(self):
        skills = infer_skills_from_concepts(["quantum computing"], "/test/skills")
        assert skills == []


class TestBuildExpertDefinition:
    def test_basic_definition(self):
        expert = build_expert_definition(
            chapter_id="ch_08",
            title="08_ML Systems",
            concepts=["neural network", "deep learning"],
        )
        assert expert["expert_name"] == "08_ML Systems"
        assert expert["slug"] == "08_ml_systems"
        assert expert["chapter_id"] == "ch_08"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert "strategy" in expert
        assert "formula" in expert
        assert "loop_config" in expert

    def test_empty_concepts_get_defaults(self):
        expert = build_expert_definition("ch_01", "01_Intro", [])
        assert len(expert["capabilities"]) >= 1
        assert "Build baseline models quickly" in expert["capabilities"]

    def test_custom_strategy(self):
        expert = build_expert_definition(
            "ch_01", "01_Intro", [],
            strategy="Custom Strategy",
        )
        assert expert["strategy"] == "Custom Strategy"


class TestRegisterExperts:
    def test_register_from_chapters(self):
        chapters = [
            {
                "chapter_id": "ch_01",
                "title": "Intro to ML",
                "concepts": ["neural network", "gradient descent"],
            },
            {
                "chapter_id": "ch_02",
                "title": "Feature Engineering",
                "concepts": ["feature engineering", "pca"],
            },
        ]
        experts = register_experts_from_chapters(chapters)
        assert len(experts) == 2

    def test_register_with_json_concepts(self):
        chapters = [
            {
                "chapter_id": "ch_01",
                "title": "Test",
                "concepts": json.dumps(["deep learning"]),
            },
        ]
        experts = register_experts_from_chapters(chapters)
        assert len(experts) == 1


class TestExpertJsonIO:
    def test_save_and_load(self, tmp_path):
        expert = build_expert_definition("ch_01", "Test Expert", ["deep learning"])
        path = save_expert_json(expert, str(tmp_path))
        assert os.path.exists(path)

        loaded = load_expert_json(expert["slug"], str(tmp_path))
        assert loaded is not None
        assert loaded["expert_name"] == "Test Expert"

    def test_load_nonexistent(self, tmp_path):
        result = load_expert_json("nonexistent", str(tmp_path))
        assert result is None


class TestSelectExperts:
    def test_select_without_embeddings(self):
        experts = [
            {
                "expert_name": "DL Expert",
                "chapter_id": "ch_01",
                "capabilities": ["Build and train deep learning models"],
                "slug": "dl",
            },
            {
                "expert_name": "FE Expert",
                "chapter_id": "ch_02",
                "capabilities": ["Feature engineering and dimensionality reduction"],
                "slug": "fe",
            },
            {
                "expert_name": "NLP Expert",
                "chapter_id": "ch_03",
                "capabilities": ["Natural language processing pipelines"],
                "slug": "nlp",
            },
        ]
        selected = select_experts_for_competition(
            "image classification deep learning", experts, top_k=2
        )
        assert len(selected) <= 2
