"""Tests for MLSysEng MoE expert registry module."""

import json

import pytest

from src.mlsyseng_mcp.database import Database
from src.mlsyseng_mcp.expert_registry import ExpertRegistry


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def registry(db, tmp_path):
    return ExpertRegistry(db, str(tmp_path / "experts"))


class TestExpertCreation:
    def test_create_expert_from_chapter(self, db, registry):
        ch_id = db.upsert_chapter(
            "01_ML_Basics", "01_ml_basics", "/p", "ML content", 10
        )
        db.add_concepts(ch_id, [
            {"concept_name": "Gradient Descent", "category": "algorithm"},
            {"concept_name": "Neural Network", "category": "architecture"},
        ])

        expert = registry.create_expert_from_chapter(ch_id)
        assert expert is not None
        assert expert["expert_name"] == "01_ML_Basics"
        assert expert["slug"] == "01_ml_basics"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert expert["strategy"]
        assert expert["formula"]

    def test_create_all_experts(self, db, registry):
        for i in range(3):
            ch_id = db.upsert_chapter(f"Ch{i}", f"ch{i}", "/p", f"content {i}", 1)
            db.add_concepts(ch_id, [
                {"concept_name": f"Concept_{i}"},
            ])

        experts = registry.create_all_experts()
        assert len(experts) == 3

    def test_create_expert_nonexistent_chapter(self, registry):
        result = registry.create_expert_from_chapter(999)
        assert result is None


class TestExpertRetrieval:
    def test_get_expert_by_name(self, db, registry):
        ch_id = db.upsert_chapter("DL", "dl", "/p", "content", 1)
        db.add_concepts(ch_id, [{"concept_name": "Deep Learning"}])
        registry.create_expert_from_chapter(ch_id)

        expert = registry.get_expert("DL")
        assert expert is not None
        assert expert["expert_name"] == "DL"

    def test_get_expert_by_slug(self, db, registry):
        ch_id = db.upsert_chapter("Deep Learning", "deep_learning", "/p", "c", 1)
        db.add_concepts(ch_id, [{"concept_name": "Neural Network"}])
        registry.create_expert_from_chapter(ch_id)

        expert = registry.get_expert("deep_learning")
        assert expert is not None

    def test_list_experts(self, db, registry):
        for i in range(2):
            ch_id = db.upsert_chapter(f"Ch{i}", f"ch{i}", "/p", "c", 1)
            registry.create_expert_from_chapter(ch_id)

        experts = registry.list_experts()
        assert len(experts) == 2


class TestExpertQuery:
    def test_query_expert(self, db, registry):
        ch_id = db.upsert_chapter("ML_Systems", "ml_systems", "/p", "Systems content", 1)
        db.add_concepts(ch_id, [
            {"concept_name": "Optimization", "description": "Optimizing models"},
        ])
        registry.create_expert_from_chapter(ch_id)

        result = registry.query_expert("ml_systems", "How to optimize?")
        assert "error" not in result
        assert result["expert_name"] == "ML_Systems"
        assert len(result["relevant_concepts"]) > 0

    def test_query_nonexistent_expert(self, registry):
        result = registry.query_expert("nonexistent", "question")
        assert "error" in result


class TestRecommendations:
    def test_recommend_experts(self, db, registry):
        ch_id = db.upsert_chapter("Neural Nets", "neural_nets", "/p", "c", 1)
        db.add_concepts(ch_id, [
            {"concept_name": "Neural Network"},
            {"concept_name": "Deep Learning"},
        ])
        registry.create_expert_from_chapter(ch_id)

        recs = registry.recommend_experts_for_competition("neural network image classification")
        assert len(recs) > 0
        assert recs[0]["relevance_score"] > 0


class TestExpertPersistence:
    def test_save_and_load_json(self, db, registry, tmp_path):
        ch_id = db.upsert_chapter("Test", "test", "/p", "c", 1)
        db.add_concepts(ch_id, [{"concept_name": "Test Concept"}])
        registry.create_expert_from_chapter(ch_id)

        json_file = tmp_path / "experts" / "test.json"
        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)
        assert data["expert_name"] == "Test"
        assert data["slug"] == "test"

    def test_load_experts_from_dir(self, db, tmp_path):
        experts_dir = tmp_path / "load_experts"
        experts_dir.mkdir()
        expert_data = {
            "expert_name": "Loaded_Expert",
            "slug": "loaded_expert",
            "capabilities": ["cap1"],
            "skills": ["/path/skill"],
            "strategy": "test strategy",
            "formula": {"objective": "test"},
            "loop_config": {"epsilon": 0.001},
        }
        with open(experts_dir / "loaded_expert.json", "w") as f:
            json.dump(expert_data, f)

        registry = ExpertRegistry(db, str(experts_dir))
        loaded = registry.load_experts_from_dir()
        assert len(loaded) == 1
        assert loaded[0]["expert_name"] == "Loaded_Expert"
