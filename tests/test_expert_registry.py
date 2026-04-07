"""Tests for mlsyseng_moe.expert_registry module."""

import pytest

from src.mlsyseng_moe.database import Database
from src.mlsyseng_moe.expert_registry import (
    ExpertRegistry,
    create_expert_definition,
    _slug_from_title,
    _infer_skills,
    _infer_metrics,
    _build_capabilities,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    d = Database(db_path=db_path)
    d.initialize()
    yield d
    d.close()


class TestSlugFromTitle:
    def test_simple_title(self):
        assert _slug_from_title("ML Systems") == "ml_systems"

    def test_special_characters(self):
        assert _slug_from_title("Deep Learning & CNNs") == "deep_learning_cnns"

    def test_spaces_and_hyphens(self):
        assert _slug_from_title("Feature Engineering - Advanced") == "feature_engineering_advanced"


class TestInferSkills:
    def test_neural_network_concepts(self):
        skills = _infer_skills(["neural network", "deep learning"])
        assert "kaggle-model-trainer" in skills
        assert "kaggle-deep-learning" in skills

    def test_tabular_concepts(self):
        skills = _infer_skills(["xgboost", "gradient boosting"])
        assert "kaggle-tabular-optimizer" in skills

    def test_default_when_no_match(self):
        skills = _infer_skills(["something_unknown"])
        assert "kaggle-preprocessor" in skills
        assert "kaggle-model-trainer" in skills


class TestInferMetrics:
    def test_classification_concepts(self):
        metrics = _infer_metrics(["classification"])
        assert "accuracy" in metrics
        assert "f1_score" in metrics

    def test_default_metrics(self):
        metrics = _infer_metrics(["unknown_concept"])
        assert "accuracy" in metrics


class TestBuildCapabilities:
    def test_includes_title(self):
        caps = _build_capabilities([], "Test Title")
        assert any("Test Title" in c for c in caps)

    def test_deep_learning_cap(self):
        caps = _build_capabilities(["deep learning"], "DL Chapter")
        assert any("Deep learning" in c for c in caps)


class TestCreateExpertDefinition:
    def test_full_definition(self):
        defn = create_expert_definition(
            chapter_id="08",
            title="ML Systems",
            concepts=["neural network", "gradient descent", "feature engineering"],
        )
        assert defn["expert_name"] == "08_ml_systems"
        assert defn["slug"] == "ml_systems"
        assert defn["chapter_id"] == "08"
        assert len(defn["skills"]) > 0
        assert defn["strategy"] is not None
        assert defn["formula"]["objective"] == "minimize_validation_loss"
        assert defn["loop_config"]["epsilon"] == 0.001

    def test_empty_concepts(self):
        defn = create_expert_definition("01", "Intro", [])
        assert len(defn["skill_names"]) > 0


class TestExpertRegistry:
    def test_register_from_chapters(self, db):
        db.upsert_chapter("01", "Introduction", "/ch1", "content", ["neural network"])
        db.upsert_chapter("02", "Feature Eng", "/ch2", "content", ["feature engineering"])

        registry = ExpertRegistry(db)
        experts = registry.register_from_chapters()

        assert len(experts) == 2
        assert experts[0]["chapter_id"] == "01"

    def test_list_experts(self, db):
        db.upsert_chapter("01", "Chapter 1", "/ch1", "content", ["deep learning"])
        registry = ExpertRegistry(db)
        registry.register_from_chapters()
        experts = registry.list_experts()
        assert len(experts) == 1

    def test_get_expert(self, db):
        db.upsert_chapter("01", "Chapter 1", "/ch1", "content", ["deep learning"])
        registry = ExpertRegistry(db)
        registry.register_from_chapters()
        experts = registry.list_experts()
        defn = registry.get_expert(experts[0]["expert_name"])
        assert defn is not None
        assert defn["chapter_id"] == "01"

    def test_find_experts_for_competition(self, db):
        db.upsert_chapter("01", "Chapter 1", "/ch1", "content", ["deep learning"])
        db.upsert_chapter("02", "Chapter 2", "/ch2", "content", ["feature engineering"])
        registry = ExpertRegistry(db)
        registry.register_from_chapters()

        relevance = [{"chapter_id": "01", "relevance": 0.9}]
        matched = registry.find_experts_for_competition("titanic", relevance)
        assert len(matched) == 1
        assert matched[0]["chapter_id"] == "01"

    def test_find_experts_fallback(self, db):
        db.upsert_chapter("01", "Chapter 1", "/ch1", "content", [])
        registry = ExpertRegistry(db)
        registry.register_from_chapters()

        matched = registry.find_experts_for_competition("titanic", None)
        assert len(matched) == 1

    def test_save_expert_json(self, db, tmp_path):
        db.upsert_chapter("01", "Test", "/ch1", "content", ["deep learning"])
        registry = ExpertRegistry(db)
        registry.register_from_chapters()
        experts = registry.list_experts()
        path = registry.save_expert_json(
            experts[0]["expert_name"],
            output_dir=str(tmp_path / "experts"),
        )
        assert path.endswith(".json")
