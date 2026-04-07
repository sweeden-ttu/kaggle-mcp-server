"""Tests for the MLSysEng MoE system components."""

import json
import math
import os
import tempfile

import pytest

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.docling_worker import _extract_concepts, _parse_chapter_info
from mlsyseng_mcp.expert_registry import (
    _build_capabilities,
    _build_formula,
    _infer_skills_from_concepts,
    _slugify,
    create_expert_from_chapter,
    infer_experts_for_competition,
    register_experts_from_db,
    save_expert_json,
    COMMON_SKILLS,
)
from mlsyseng_mcp.loop_controller import (
    LoopController,
    LoopState,
    build_competition_step,
    l2_norm_diff,
)
from skill_generator import load_skills_yaml


class TestDatabase:
    """Tests for SQLite database operations."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_upsert_and_get_chapter(self):
        ch_id = self.db.upsert_chapter(
            chapter_num=1,
            title="Introduction to ML",
            source_path="/test/ch1.pdf",
            markdown_content="# Chapter 1\nIntro content",
            concepts=["neural network", "classification"],
            page_count=10,
        )
        assert ch_id > 0

        ch = self.db.get_chapter(1)
        assert ch is not None
        assert ch["title"] == "Introduction to ML"
        assert ch["chapter_num"] == 1
        assert ch["page_count"] == 10
        assert "neural network" in ch["concepts"]
        assert "classification" in ch["concepts"]

    def test_upsert_chapter_updates_existing(self):
        self.db.upsert_chapter(1, "Title A", "/a.pdf", "content A", ["concept1"])
        self.db.upsert_chapter(1, "Title B", "/b.pdf", "content B", ["concept2"])
        ch = self.db.get_chapter(1)
        assert ch["title"] == "Title B"
        assert ch["concepts"] == ["concept2"]

    def test_list_chapters_empty(self):
        result = self.db.list_chapters()
        assert result == []

    def test_list_chapters_ordered(self):
        self.db.upsert_chapter(3, "Ch3", "/3.pdf", "c3", [])
        self.db.upsert_chapter(1, "Ch1", "/1.pdf", "c1", [])
        self.db.upsert_chapter(2, "Ch2", "/2.pdf", "c2", [])
        chapters = self.db.list_chapters()
        assert len(chapters) == 3
        assert [c["chapter_num"] for c in chapters] == [1, 2, 3]

    def test_upsert_and_get_expert(self):
        expert_data = {
            "expert_name": "01_Intro_ML",
            "slug": "01_intro_ml",
            "chapter_id": None,
            "capabilities": ["Build models"],
            "skills": ["/path/skill1"],
            "strategy": "Baseline -> Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        eid = self.db.upsert_expert(expert_data)
        assert eid > 0

        expert = self.db.get_expert("01_intro_ml")
        assert expert is not None
        assert expert["expert_name"] == "01_Intro_ML"
        assert expert["capabilities"] == ["Build models"]
        assert expert["formula"] == {"objective": "minimize_loss"}

    def test_get_expert_not_found(self):
        assert self.db.get_expert("nonexistent") is None

    def test_list_experts(self):
        self.db.upsert_expert({
            "expert_name": "Expert B", "slug": "b", "capabilities": [],
            "skills": [], "strategy": "", "formula": {}, "loop_config": {},
        })
        self.db.upsert_expert({
            "expert_name": "Expert A", "slug": "a", "capabilities": [],
            "skills": [], "strategy": "", "formula": {}, "loop_config": {},
        })
        experts = self.db.list_experts()
        assert len(experts) == 2
        assert experts[0]["expert_name"] == "Expert A"

    def test_extraction_status(self):
        self.db.set_extraction_status(1, "started")
        self.db.set_extraction_status(1, "completed")
        statuses = self.db.get_extraction_status()
        assert len(statuses) == 1
        assert statuses[0]["status"] == "completed"

    def test_extraction_status_failed(self):
        self.db.set_extraction_status(2, "started")
        self.db.set_extraction_status(2, "failed", "PDF corrupted")
        statuses = self.db.get_extraction_status()
        found = [s for s in statuses if s["chapter_num"] == 2]
        assert found[0]["status"] == "failed"
        assert found[0]["error_message"] == "PDF corrupted"

    def test_get_stats(self):
        self.db.upsert_chapter(1, "Ch1", "/1.pdf", "content", ["a"], page_count=5)
        self.db.upsert_expert({
            "expert_name": "E1", "slug": "e1", "capabilities": [],
            "skills": [], "strategy": "", "formula": {}, "loop_config": {},
        })
        stats = self.db.get_stats()
        assert stats["chapters_indexed"] == 1
        assert stats["experts_registered"] == 1
        assert stats["total_pages_extracted"] == 5

    def test_get_all_chapter_content(self):
        self.db.upsert_chapter(1, "Ch1", "/1.pdf", "content 1", ["concept1"])
        self.db.upsert_chapter(2, "Ch2", "/2.pdf", "content 2", ["concept2"])
        content = self.db.get_all_chapter_content()
        assert len(content) == 2
        assert content[0]["markdown_content"] == "content 1"


class TestDoclingWorker:
    """Tests for concept extraction and chapter parsing."""

    def test_extract_concepts_neural(self):
        text = "This chapter covers neural network architectures and deep learning."
        concepts = _extract_concepts(text)
        assert any("Neural Network" in c for c in concepts)
        assert any("Deep Learning" in c for c in concepts)

    def test_extract_concepts_optimization(self):
        text = "We will study gradient descent and backpropagation techniques."
        concepts = _extract_concepts(text)
        assert any("Gradient Descent" in c for c in concepts)
        assert any("Backpropagation" in c for c in concepts)

    def test_extract_concepts_ensemble(self):
        text = "Random forest and xgboost are popular ensemble methods."
        concepts = _extract_concepts(text)
        assert any("Random Forest" in c for c in concepts)
        assert any("Xgboost" in c for c in concepts)

    def test_extract_concepts_empty(self):
        concepts = _extract_concepts("Nothing related here at all.")
        assert concepts == []

    def test_parse_chapter_info_dash(self):
        result = _parse_chapter_info("01 - Introduction to ML")
        assert result == (1, "Introduction to ML")

    def test_parse_chapter_info_underscore(self):
        result = _parse_chapter_info("02_Deep_Learning")
        assert result == (2, "Deep Learning")

    def test_parse_chapter_info_chapter_prefix(self):
        result = _parse_chapter_info("Chapter 3 - Optimization")
        assert result == (3, "Optimization")

    def test_parse_chapter_info_dot(self):
        result = _parse_chapter_info("04. Feature Engineering")
        assert result == (4, "Feature Engineering")

    def test_parse_chapter_info_invalid(self):
        assert _parse_chapter_info("random_folder") is None

    def test_parse_chapter_info_no_number(self):
        assert _parse_chapter_info("Appendix A") is None


class TestExpertRegistry:
    """Tests for expert creation and management."""

    def test_slugify(self):
        assert _slugify("Hello World") == "hello_world"
        assert _slugify("ML Systems!") == "ml_systems"
        assert _slugify("  spaces  ") == "spaces"

    def test_infer_skills_neural(self):
        skills = _infer_skills_from_concepts(["Neural Network"])
        assert "model-trainer" in skills
        assert "deep-learning-pipeline" in skills
        for common in COMMON_SKILLS:
            assert common in skills

    def test_infer_skills_ensemble(self):
        skills = _infer_skills_from_concepts(["Random Forest", "XGBoost"])
        assert "random-forest-trainer" in skills
        assert "xgboost-trainer" in skills

    def test_infer_skills_empty(self):
        skills = _infer_skills_from_concepts([])
        assert set(skills) == set(COMMON_SKILLS)

    def test_build_formula_classification(self):
        formula = _build_formula(["classification", "precision"])
        assert formula["objective"] == "minimize_validation_loss"
        assert "accuracy" in formula["metrics"]

    def test_build_formula_regression(self):
        formula = _build_formula(["regression"])
        assert "rmse" in formula["metrics"]

    def test_build_formula_clustering(self):
        formula = _build_formula(["clustering"])
        assert "silhouette_score" in formula["metrics"]

    def test_build_formula_default(self):
        formula = _build_formula(["general concept"])
        assert formula["metrics"] == ["accuracy"]

    def test_build_capabilities(self):
        caps = _build_capabilities("Deep Learning", ["deep learning", "feature engineering"])
        assert any("Deep Learning" in c for c in caps)
        assert any("Deep learning architecture" in c for c in caps)
        assert any("feature engineering" in c.lower() for c in caps)

    def test_create_expert_from_chapter(self):
        expert = create_expert_from_chapter(
            chapter_num=8,
            title="ML Systems",
            concepts=["neural network", "optimization"],
            chapter_id=42,
        )
        assert expert["expert_name"] == "08_ML_Systems"
        assert expert["slug"] == "08_ml_systems"
        assert expert["chapter_id"] == 42
        assert len(expert["capabilities"]) >= 3
        assert len(expert["skills"]) >= len(COMMON_SKILLS)
        assert expert["strategy"]
        assert expert["formula"]["objective"] == "minimize_validation_loss"
        assert expert["loop_config"]["epsilon"] == 0.001

    def test_register_experts_from_db(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            db = Database(db_path=tmp.name)
            db.upsert_chapter(1, "Intro", "/1.pdf", "Neural network content", ["neural network"])
            db.upsert_chapter(2, "Deep", "/2.pdf", "Deep learning content", ["deep learning"])
            results = register_experts_from_db(db)
            assert len(results) == 2
            experts = db.list_experts()
            assert len(experts) == 2
        finally:
            os.unlink(tmp.name)

    def test_save_expert_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert = create_expert_from_chapter(1, "Test", ["classification"])
            path = save_expert_json(expert, output_dir=tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["expert_name"] == expert["expert_name"]

    def test_infer_experts_without_embeddings(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            db = Database(db_path=tmp.name)
            db.upsert_chapter(1, "Neural Networks", "/1.pdf", "content", ["neural network"])
            register_experts_from_db(db)
            experts = infer_experts_for_competition("neural network classification", db, top_k=1)
            assert len(experts) <= 1
        finally:
            os.unlink(tmp.name)


class TestLoopController:
    """Tests for the convergence loop controller."""

    def test_l2_norm_diff_identical(self):
        assert l2_norm_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_l2_norm_diff_known(self):
        diff = l2_norm_diff([0.0, 0.0], [3.0, 4.0])
        assert abs(diff - 5.0) < 1e-10

    def test_l2_norm_diff_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            l2_norm_diff([1.0], [1.0, 2.0])

    def test_loop_converges(self):
        controller = LoopController(
            epsilon=0.5, max_iterations=20, patience=3
        )
        result = controller.run(build_competition_step_fn([]))
        assert result.converged
        assert result.total_iterations < 20
        assert "converged" in result.exit_reason

    def test_loop_max_iterations(self):
        controller = LoopController(
            epsilon=0.0000001, max_iterations=3, patience=3
        )

        def diverging_step(iteration, prev):
            return LoopState(
                iteration=iteration,
                state_vector=[float(iteration), float(iteration * 2)],
                metrics={"loss": float(iteration)},
                expert_contributions=[],
            )

        result = controller.run(diverging_step)
        assert not result.converged
        assert result.total_iterations == 3
        assert result.exit_reason == "max_iterations_reached"

    def test_loop_to_dict(self):
        controller = LoopController(epsilon=0.1, max_iterations=5, patience=2)
        result = controller.run(build_competition_step_fn([]))
        d = controller.to_dict(result)
        assert "converged" in d
        assert "total_iterations" in d
        assert "final_metrics" in d
        assert "config" in d
        assert d["config"]["epsilon"] == 0.1

    def test_from_config(self):
        config = {"epsilon": 0.01, "max_iterations": 5, "patience": 2}
        controller = LoopController.from_config(config)
        assert controller.epsilon == 0.01
        assert controller.max_iterations == 5
        assert controller.patience == 2

    def test_build_competition_step_initial(self):
        state = build_competition_step([], "titanic", 0, None)
        assert state.iteration == 0
        assert len(state.state_vector) == 3
        assert state.metrics["loss"] == 1.0
        assert state.metrics["accuracy"] == 0.5

    def test_build_competition_step_with_previous(self):
        state = build_competition_step([], "titanic", 1, [1.0, 0.5, 0.475])
        assert state.iteration == 1
        assert state.metrics["loss"] < 1.0
        assert state.metrics["accuracy"] > 0.5


class TestSkillGenerator:
    """Tests for skill YAML loading."""

    def test_load_skills_yaml(self):
        config = load_skills_yaml("skills.yaml")
        assert config["name"] == "mlsyseng-moe"
        assert config["version"] == "0.1.0"
        assert "mcp_server" in config
        assert "tools" in config
        assert "platforms" in config
        assert "skill_content" in config

    def test_skills_yaml_tools(self):
        config = load_skills_yaml("skills.yaml")
        tool_names = [t["name"] for t in config["tools"]]
        assert "extract-knowledge" in tool_names
        assert "evolve" in tool_names
        assert "search-concepts" in tool_names
        assert "list-experts" in tool_names
        assert "build-entry" in tool_names

    def test_skills_yaml_platforms(self):
        config = load_skills_yaml("skills.yaml")
        platforms = config["platforms"]
        assert "openclaw" in platforms
        assert "claude_desktop" in platforms
        assert "cursor" in platforms
        assert "gemini" in platforms
        assert "generic" in platforms


def build_competition_step_fn(experts):
    """Helper to create a step function for testing."""
    def step_fn(iteration, prev_state):
        return build_competition_step(experts, "test", iteration, prev_state)
    return step_fn
