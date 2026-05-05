"""Tests for MLSysEng MoE core modules."""

import json
import os
import tempfile

import pytest

from src.mlsyseng_mcp.database import Database
from src.mlsyseng_mcp.docling_worker import (
    extract_concepts,
    parse_chapter_title,
)
from src.mlsyseng_mcp.expert_registry import ExpertRegistry, _slugify
from src.mlsyseng_mcp.loop_controller import (
    ConvergenceResult,
    LoopController,
    LoopState,
    l2_norm,
)


# ── Database tests ──


class TestDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        database = Database(db_path=db_path)
        yield database
        database.close()

    def test_upsert_and_get_chapter(self, db):
        chapter_id = db.upsert_chapter(
            folder_name="01_intro",
            title="Introduction",
            content_md="# Introduction\nThis is the intro.",
            concepts=["gradient", "loss"],
            pdf_path="/path/to/01_intro.pdf",
        )
        assert chapter_id > 0

        chapter = db.get_chapter("01_intro")
        assert chapter is not None
        assert chapter["title"] == "Introduction"
        assert chapter["concepts"] == ["gradient", "loss"]
        assert chapter["content_md"] == "# Introduction\nThis is the intro."

    def test_upsert_chapter_updates_existing(self, db):
        db.upsert_chapter("01_intro", "Intro v1", "content v1", ["gradient"], "/v1.pdf")
        db.upsert_chapter("01_intro", "Intro v2", "content v2", ["loss"], "/v2.pdf")

        chapter = db.get_chapter("01_intro")
        assert chapter["title"] == "Intro v2"
        assert chapter["concepts"] == ["loss"]

    def test_list_chapters(self, db):
        db.upsert_chapter("01_intro", "Intro", "content1", [], "/1.pdf")
        db.upsert_chapter("02_linear", "Linear", "content2", [], "/2.pdf")

        chapters = db.list_chapters()
        assert len(chapters) == 2
        assert chapters[0]["folder_name"] == "01_intro"
        assert chapters[1]["folder_name"] == "02_linear"

    def test_upsert_and_get_expert(self, db):
        expert_data = {
            "slug": "ml_systems",
            "expert_name": "ML Systems",
            "chapter_id": None,
            "capabilities": ["Build baseline models"],
            "skills": ["/skills/kaggle-preprocessor"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001, "max_iterations": 10},
        }
        expert_id = db.upsert_expert(expert_data)
        assert expert_id > 0

        expert = db.get_expert("ml_systems")
        assert expert is not None
        assert expert["expert_name"] == "ML Systems"
        assert expert["capabilities"] == ["Build baseline models"]
        assert expert["formula"]["objective"] == "minimize_loss"

    def test_list_experts(self, db):
        db.upsert_expert({"slug": "a_expert", "expert_name": "A"})
        db.upsert_expert({"slug": "b_expert", "expert_name": "B"})

        experts = db.list_experts()
        assert len(experts) == 2

    def test_extraction_log(self, db):
        chapter_id = db.upsert_chapter("01", "Ch1", "c", [], "/1.pdf")
        db.log_extraction(chapter_id, "started")
        db.log_extraction(chapter_id, "completed")

        status = db.get_extraction_status()
        assert len(status) == 2

    def test_stats(self, db):
        db.upsert_chapter("01", "Ch1", "c", [], "/1.pdf")
        db.upsert_expert({"slug": "e1", "expert_name": "E1"})

        stats = db.get_stats()
        assert stats["total_chapters"] == 1
        assert stats["total_experts"] == 1


# ── Docling worker tests ──


class TestDoclingWorker:
    def test_extract_concepts(self):
        text = "We use gradient descent to minimize the loss function with regularization."
        concepts = extract_concepts(text)
        assert "gradient" in concepts
        assert "loss" in concepts
        assert "regularization" in concepts

    def test_extract_concepts_empty(self):
        concepts = extract_concepts("Nothing relevant here.")
        assert concepts == []

    def test_parse_chapter_title(self):
        assert parse_chapter_title("08_ML_Systems") == "ML Systems"
        assert parse_chapter_title("03 - Neural Networks") == "Neural Networks"
        assert parse_chapter_title("01_Introduction") == "Introduction"
        assert parse_chapter_title("12_Deep_Learning") == "Deep Learning"

    def test_parse_chapter_title_no_number(self):
        assert parse_chapter_title("Introduction") == "Introduction"


# ── Expert registry tests ──


class TestExpertRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        db = Database(db_path=str(tmp_path / "test.db"))
        reg = ExpertRegistry(db, skills_path="/test/skills")
        yield reg
        db.close()

    def test_slugify(self):
        assert _slugify("ML Systems") == "ml_systems"
        assert _slugify("Deep Learning") == "deep_learning"
        assert _slugify("08 Neural Networks!") == "08_neural_networks"

    def test_create_expert_from_chapter(self, registry):
        chapter_id = registry.db.upsert_chapter(
            "08_ml_systems", "ML Systems", "content", ["gradient", "loss", "optimization"],
            "/path.pdf"
        )

        expert = registry.create_expert_from_chapter(
            chapter_id=chapter_id,
            folder_name="08_ml_systems",
            title="ML Systems",
            concepts=["gradient", "loss", "optimization"],
        )

        assert expert["slug"] == "ml_systems"
        assert expert["expert_name"] == "ML Systems"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert "minimize_validation_loss" in expert["formula"]["objective"]

    def test_create_experts_from_chapters(self, registry):
        registry.db.upsert_chapter("01", "Intro", "content1", ["gradient"], "/1.pdf")
        registry.db.upsert_chapter("02", "Linear", "content2", ["regression"], "/2.pdf")

        experts = registry.create_experts_from_chapters()
        assert len(experts) == 2

    def test_list_experts(self, registry):
        registry.db.upsert_chapter("01", "Intro", "content", ["gradient"], "/1.pdf")
        registry.create_expert_from_chapter(1, "01", "Intro", ["gradient"])

        experts = registry.list_experts()
        assert len(experts) == 1

    def test_query_expert(self, registry):
        chapter_id = registry.db.upsert_chapter(
            "01", "Intro", "Some ML content", ["gradient"], "/1.pdf"
        )
        registry.create_expert_from_chapter(chapter_id, "01", "Intro", ["gradient"])

        result = registry.query_expert("intro", "How does gradient descent work?")
        assert "expert" in result
        assert result["question"] == "How does gradient descent work?"

    def test_query_nonexistent_expert(self, registry):
        result = registry.query_expert("nonexistent", "test")
        assert "error" in result

    def test_build_competition_entry(self, registry):
        chapter_id = registry.db.upsert_chapter(
            "01", "Intro", "content", ["gradient", "loss"], "/1.pdf"
        )
        registry.create_expert_from_chapter(chapter_id, "01", "Intro", ["gradient", "loss"])

        relevant = [{"folder_name": "01", "title": "Intro", "relevance": 0.9}]
        entry = registry.build_competition_entry("titanic", relevant)

        assert entry["competition"] == "titanic"
        assert len(entry["combined_strategy"]) > 0

    def test_export_expert_json(self, registry, tmp_path):
        registry.db.upsert_chapter("01", "Intro", "content", ["gradient"], "/1.pdf")
        registry.create_expert_from_chapter(1, "01", "Intro", ["gradient"])

        path = registry.export_expert_json("intro", str(tmp_path / "experts"))
        assert path is not None
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert data["slug"] == "intro"


# ── Loop controller tests ──


class TestLoopController:
    def test_l2_norm_identical(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_l2_norm_different(self):
        norm = l2_norm([0.0, 0.0], [3.0, 4.0])
        assert abs(norm - 5.0) < 1e-10

    def test_l2_norm_dimension_mismatch(self):
        with pytest.raises(ValueError):
            l2_norm([1.0], [1.0, 2.0])

    def test_loop_state_to_vector(self):
        state = LoopState(iteration=0, values={"a": 1.0, "b": 2.0, "c": 3.0})
        vec = state.to_vector()
        assert vec == [1.0, 2.0, 3.0]

    def test_convergence_immediate(self):
        controller = LoopController(epsilon=1.0, max_iterations=10, patience=1)

        def step(state, iteration):
            return LoopState(iteration=iteration, values=state.values.copy())

        result = controller.run(step, {"x": 1.0, "y": 2.0})
        assert result.converged is True
        assert result.iterations == 1

    def test_convergence_with_patience(self):
        controller = LoopController(epsilon=1.0, max_iterations=20, patience=3)

        def step(state, iteration):
            return LoopState(iteration=iteration, values=state.values.copy())

        result = controller.run(step, {"x": 1.0})
        assert result.converged is True
        assert result.iterations == 3

    def test_no_convergence_max_iterations(self):
        controller = LoopController(epsilon=0.0001, max_iterations=5, patience=3)

        def step(state, iteration):
            new_vals = {k: v + 1.0 for k, v in state.values.items()}
            return LoopState(iteration=iteration, values=new_vals)

        result = controller.run(step, {"x": 0.0})
        assert result.converged is False
        assert result.iterations == 5

    def test_gradual_convergence(self):
        controller = LoopController(epsilon=0.01, max_iterations=50, patience=3)

        def step(state, iteration):
            decay = 0.5 ** iteration
            new_vals = {k: v + decay for k, v in state.values.items()}
            return LoopState(iteration=iteration, values=new_vals)

        result = controller.run(step, {"x": 0.0, "y": 0.0})
        assert result.converged is True

    def test_run_competition(self):
        controller = LoopController(epsilon=0.01, max_iterations=20, patience=3)
        result = controller.run_competition("titanic", ["ml_systems", "deep_learning"])
        assert isinstance(result, ConvergenceResult)
        assert result.iterations > 0

    def test_from_config(self):
        config = {"epsilon": 0.01, "max_iterations": 20, "patience": 5}
        controller = LoopController.from_config(config)
        assert controller.epsilon == 0.01
        assert controller.max_iterations == 20
        assert controller.patience == 5

    def test_to_dict(self):
        controller = LoopController(epsilon=0.001, max_iterations=10, patience=3)
        d = controller.to_dict()
        assert d["epsilon"] == 0.001
        assert d["exit_condition"] == "||state[n] - state[n-1]||_2 < epsilon"

    def test_step_function_error_handling(self):
        controller = LoopController(epsilon=0.001, max_iterations=5, patience=3)

        def bad_step(state, iteration):
            raise ValueError("Step failed")

        result = controller.run(bad_step, {"x": 1.0})
        assert result.converged is False
        assert "error" in result.reason.lower()
