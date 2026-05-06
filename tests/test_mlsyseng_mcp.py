"""Tests for the MLSysEng MoE system.

Tests database operations, expert registry, docling worker concept extraction,
loop controller convergence, and skill generator config loading.
"""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp import database as db
from mlsyseng_mcp import docling_worker
from mlsyseng_mcp import expert_registry
from mlsyseng_mcp import loop_controller


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db.init_db(path)
    yield path
    os.unlink(path)


class TestDatabase:
    def test_init_db(self, tmp_db):
        stats = db.get_stats(tmp_db)
        assert stats["chapters"] == 0
        assert stats["experts"] == 0
        assert stats["total_words"] == 0

    def test_upsert_and_get_chapter(self, tmp_db):
        chapter_id = db.upsert_chapter(
            folder_name="01_Introduction",
            chapter_number=1,
            title="Introduction",
            pdf_path="/tmp/ch01.pdf",
            markdown_content="This is the intro chapter about machine learning.",
            concepts=["machine learning", "neural network"],
            db_path=tmp_db,
        )
        assert chapter_id > 0

        chapter = db.get_chapter("01_Introduction", db_path=tmp_db)
        assert chapter is not None
        assert chapter["title"] == "Introduction"
        assert chapter["chapter_number"] == 1
        assert "machine learning" in chapter["concepts"]
        assert chapter["word_count"] == 8

    def test_list_chapters(self, tmp_db):
        db.upsert_chapter("01_Intro", 1, "Intro", "/p1", "text one", ["ml"], tmp_db)
        db.upsert_chapter("02_Deep", 2, "Deep", "/p2", "text two", ["dl"], tmp_db)
        chapters = db.list_chapters(tmp_db)
        assert len(chapters) == 2
        assert chapters[0]["chapter_number"] == 1

    def test_upsert_and_get_expert(self, tmp_db):
        ch_id = db.upsert_chapter("01_Intro", 1, "Intro", "/p", "txt", ["ml"], tmp_db)
        expert_id = db.upsert_expert(
            expert_name="01_Introduction",
            slug="01_introduction",
            chapter_id=ch_id,
            capabilities=["Build baselines"],
            skills=["~/skills/kaggle-preprocessor"],
            strategy="Baseline → Submit",
            formula={"objective": "minimize_loss"},
            loop_config={"epsilon": 0.001},
            db_path=tmp_db,
        )
        assert expert_id > 0

        expert = db.get_expert("01_introduction", db_path=tmp_db)
        assert expert is not None
        assert expert["expert_name"] == "01_Introduction"
        assert "Build baselines" in expert["capabilities"]
        assert expert["formula"]["objective"] == "minimize_loss"

    def test_list_experts(self, tmp_db):
        ch_id = db.upsert_chapter("01_A", 1, "A", "/p", "t", [], tmp_db)
        db.upsert_expert("A", "a", ch_id, ["cap"], ["sk"], "strat", {}, {}, tmp_db)
        db.upsert_expert("B", "b", ch_id, ["cap"], ["sk"], "strat", {}, {}, tmp_db)
        experts = db.list_experts(tmp_db)
        assert len(experts) == 2

    def test_convergence_state(self, tmp_db):
        db.save_state("titanic", 1, [0.5, 0.5], 0.1, False, tmp_db)
        db.save_state("titanic", 2, [0.6, 0.4], 0.05, False, tmp_db)

        latest = db.get_latest_state("titanic", tmp_db)
        assert latest is not None
        assert latest["iteration"] == 2
        assert latest["l2_norm"] == 0.05

        history = db.get_state_history("titanic", tmp_db)
        assert len(history) == 2

    def test_extraction_jobs(self, tmp_db):
        job_id = db.create_extraction_job("01_Intro", tmp_db)
        db.update_extraction_job(job_id, "running", db_path=tmp_db)
        db.update_extraction_job(job_id, "done", db_path=tmp_db)
        status = db.get_extraction_status(tmp_db)
        assert status.get("done", 0) == 1

    def test_upsert_chapter_updates(self, tmp_db):
        db.upsert_chapter("01_A", 1, "A", "/p", "old text", ["ml"], tmp_db)
        db.upsert_chapter("01_A", 1, "A Updated", "/p2", "new text here", ["ml", "dl"], tmp_db)
        chapter = db.get_chapter("01_A", db_path=tmp_db)
        assert chapter["title"] == "A Updated"
        assert chapter["word_count"] == 3


class TestDoclingWorker:
    def test_extract_concepts(self):
        text = """
        Neural networks use backpropagation for gradient descent optimization.
        Cross-validation helps prevent overfitting. Feature engineering and
        feature selection improve model performance. XGBoost is a popular
        gradient boosting framework.
        """
        concepts = docling_worker.extract_concepts(text)
        assert len(concepts) > 0
        assert any("neural network" in c for c in concepts)
        assert any("gradient descent" in c for c in concepts)
        assert any("cross-validation" in c or "cross validation" in c for c in concepts)

    def test_extract_concepts_empty(self):
        concepts = docling_worker.extract_concepts("")
        assert concepts == []

    def test_discover_chapters_missing_path(self):
        chapters = docling_worker.discover_chapters("/nonexistent/path")
        assert chapters == []


class TestExpertRegistry:
    def test_slugify(self):
        assert expert_registry._slugify("08 ML Systems") == "08_ml_systems"
        assert expert_registry._slugify("Deep Learning!") == "deep_learning"

    def test_infer_skills(self):
        concepts = ["neural network", "gradient boosting", "clustering"]
        skills = expert_registry._infer_skills(concepts)
        assert len(skills) > 2
        assert any("kaggle-model-trainer" in s for s in skills)

    def test_infer_capabilities(self):
        concepts = ["neural network", "feature engineering"]
        caps = expert_registry._infer_capabilities(concepts)
        assert "Build baseline models quickly" in caps
        assert any("neural" in c.lower() for c in caps)

    def test_infer_formula_default(self):
        formula = expert_registry._infer_formula(["neural network"])
        assert formula["objective"] == "minimize_validation_loss"

    def test_infer_formula_clustering(self):
        formula = expert_registry._infer_formula(["clustering"])
        assert formula["objective"] == "maximize_silhouette_score"

    def test_infer_formula_regression(self):
        formula = expert_registry._infer_formula(["linear regression"])
        assert formula["objective"] == "minimize_rmse"

    def test_register_experts(self, tmp_db):
        db.upsert_chapter(
            "01_Neural_Networks", 1, "Neural Networks", "/p",
            "Neural networks and deep learning for classification.",
            ["neural network", "deep learning"],
            tmp_db,
        )
        results = expert_registry.register_experts_from_chapters(db_path=tmp_db)
        assert len(results) == 1
        assert results[0]["capabilities_count"] > 0

        experts = db.list_experts(tmp_db)
        assert len(experts) == 1
        assert "neural" in experts[0]["expert_name"].lower() or "01" in experts[0]["expert_name"]


class TestLoopController:
    def test_l2_norm(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        norm = loop_controller._l2_norm(a, b)
        assert abs(norm - 1.4142135623730951) < 1e-6

    def test_l2_norm_identical(self):
        a = [0.5, 0.5]
        norm = loop_controller._l2_norm(a, a)
        assert norm == 0.0

    def test_l2_norm_different_lengths(self):
        a = [1.0, 0.0, 0.5]
        b = [0.0, 1.0]
        norm = loop_controller._l2_norm(a, b)
        assert norm > 0

    def test_initialize_state(self):
        experts = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        state = loop_controller._initialize_state(experts)
        assert len(state) == 3
        assert abs(sum(state) - 1.0) < 1e-6

    def test_convergence_loop_no_experts(self, tmp_db):
        result = loop_controller.run_convergence_loop(
            "titanic", db_path=tmp_db
        )
        assert result["status"] == "no_experts"

    def test_convergence_loop_with_experts(self, tmp_db):
        ch_id = db.upsert_chapter(
            "01_ML", 1, "ML", "/p", "machine learning text", ["ml"], tmp_db
        )
        db.upsert_expert(
            "01_ML", "01_ml", ch_id,
            ["Build baselines"], ["~/skills/kaggle-preprocessor"],
            "Baseline → Submit",
            {"objective": "minimize_loss", "metrics": ["accuracy"]},
            {"epsilon": 0.001, "max_iterations": 5, "patience": 2},
            tmp_db,
        )
        result = loop_controller.run_convergence_loop(
            "titanic",
            db_path=tmp_db,
            max_iterations=5,
            patience=2,
        )
        assert result["status"] in ("converged", "max_iterations")
        assert result["competition"] == "titanic"
        assert len(result["history"]) > 0
        assert len(result["expert_weights"]) > 0

    def test_build_competition_entry_no_experts(self, tmp_db):
        result = loop_controller.build_competition_entry("titanic", db_path=tmp_db)
        assert result["competition"] == "titanic"
        assert result["experts_selected"] == 0


class TestSkillGenerator:
    def test_load_skills_yaml(self):
        import yaml
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "skills.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            assert cfg["name"] == "mlsyseng-moe"
            assert "tools" in cfg
            assert len(cfg["tools"]) >= 8
            assert "platforms" in cfg
