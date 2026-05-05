"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mlsyseng_mcp.database import MoEDatabase
from mlsyseng_mcp.expert_registry import (
    create_expert_from_chapter,
    register_experts_from_chapters,
    get_expert_for_query,
    _slugify,
    _map_concepts_to_skills,
    _infer_capabilities,
)
from mlsyseng_mcp.loop_controller import (
    LoopController,
    default_step_fn,
    _l2_norm,
    _vector_diff,
)
from mlsyseng_mcp.docling_worker import (
    _extract_concepts_from_text,
    _chapter_id_from_path,
)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = MoEDatabase(db_path)
    yield db
    db.close()
    os.unlink(db_path)


# ── Database tests ────────────────────────────────────────────────────


class TestMoEDatabase:
    def test_create_and_get_chapter(self, temp_db):
        temp_db.upsert_chapter(
            chapter_id="ch01",
            title="Chapter 1 - Basics",
            source_path="/tmp/ch01",
            markdown="# Chapter 1\nSome content about gradient descent",
            concepts=["gradient descent", "optimization"],
        )
        chapter = temp_db.get_chapter("ch01")
        assert chapter is not None
        assert chapter["title"] == "Chapter 1 - Basics"
        assert "gradient descent" in chapter["concepts"]

    def test_upsert_chapter_updates_existing(self, temp_db):
        temp_db.upsert_chapter("ch01", "V1", "/tmp", "old", ["a"])
        temp_db.upsert_chapter("ch01", "V2", "/tmp", "new", ["b"])
        chapter = temp_db.get_chapter("ch01")
        assert chapter["title"] == "V2"
        assert chapter["markdown"] == "new"

    def test_list_chapters(self, temp_db):
        temp_db.upsert_chapter("ch01", "Ch1", "/tmp", "md1", ["a"])
        temp_db.upsert_chapter("ch02", "Ch2", "/tmp", "md2", ["b"])
        chapters = temp_db.list_chapters()
        assert len(chapters) == 2

    def test_upsert_and_list_experts(self, temp_db):
        expert = {
            "expert_name": "ML Systems",
            "slug": "ml_systems",
            "chapter_id": "ch08",
            "capabilities": ["Train models"],
            "skills": ["/path/to/skill"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        temp_db.upsert_expert(expert)
        experts = temp_db.list_experts()
        assert len(experts) == 1
        assert experts[0]["expert_name"] == "ML Systems"

    def test_get_expert_by_slug(self, temp_db):
        expert = {
            "expert_name": "Deep Learning",
            "slug": "deep_learning",
            "capabilities": [],
            "skills": [],
            "strategy": "",
            "formula": {},
            "loop_config": {},
        }
        temp_db.upsert_expert(expert)
        found = temp_db.get_expert_by_slug("deep_learning")
        assert found is not None
        assert found["expert_name"] == "Deep Learning"

    def test_save_and_get_loop_state(self, temp_db):
        temp_db.save_loop_state("titanic", 1, [1.0, 0.5], {"loss": 0.3})
        temp_db.save_loop_state("titanic", 2, [0.5, 0.25], {"loss": 0.15})
        states = temp_db.get_loop_states("titanic")
        assert len(states) == 2
        assert states[0]["iteration"] == 1

    def test_get_latest_loop_state(self, temp_db):
        temp_db.save_loop_state("titanic", 1, [1.0], {"loss": 0.5})
        temp_db.save_loop_state("titanic", 2, [0.5], {"loss": 0.25})
        latest = temp_db.get_latest_loop_state("titanic")
        assert latest["iteration"] == 2

    def test_extraction_job_lifecycle(self, temp_db):
        temp_db.create_extraction_job("job1", total_files=5)
        temp_db.update_extraction_job("job1", processed=3)
        temp_db.update_extraction_job("job1", error="bad pdf")
        temp_db.update_extraction_job("job1", status="completed")
        job = temp_db.get_extraction_job("job1")
        assert job["processed"] == 3
        assert job["status"] == "completed"
        assert len(job["errors"]) == 1

    def test_get_stats(self, temp_db):
        temp_db.upsert_chapter("ch01", "Ch1", "/tmp", "md", [])
        stats = temp_db.get_stats()
        assert stats["chapters"] == 1
        assert stats["experts"] == 0


# ── Expert registry tests ────────────────────────────────────────────


class TestExpertRegistry:
    def test_slugify(self):
        assert _slugify("08 ML Systems") == "08_ml_systems"
        assert _slugify("Deep Learning!") == "deep_learning"
        assert _slugify("hello---world") == "hello_world"

    def test_map_concepts_to_skills(self):
        skills = _map_concepts_to_skills(["gradient descent", "neural network"])
        assert len(skills) > 0
        assert any("model-trainer" in s for s in skills)

    def test_infer_capabilities(self):
        caps = _infer_capabilities(["neural network", "gradient descent"])
        assert len(caps) > 0
        assert any("deep learning" in c.lower() for c in caps)

    def test_create_expert_from_chapter(self):
        chapter = {
            "chapter_id": "ch08",
            "title": "08 ML Systems",
            "concepts": ["gradient descent", "neural network", "cross-validation"],
            "markdown": "content",
        }
        expert = create_expert_from_chapter(chapter)
        assert expert["expert_name"] == "08 ML Systems"
        assert expert["slug"] == "08_ml_systems"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert "→" in expert["strategy"]
        assert expert["formula"]["objective"] in ("minimize_loss", "minimize_validation_loss")

    def test_register_experts_from_chapters(self, temp_db):
        temp_db.upsert_chapter(
            "ch01", "Basics", "/tmp", "md", ["gradient descent"]
        )
        temp_db.upsert_chapter(
            "ch02", "Deep Learning", "/tmp", "md", ["neural network", "cnn"]
        )
        experts = register_experts_from_chapters(temp_db)
        assert len(experts) == 2
        db_experts = temp_db.list_experts()
        assert len(db_experts) == 2

    def test_get_expert_for_query(self, temp_db):
        temp_db.upsert_chapter("ch01", "Basics", "/tmp", "md", ["gradient descent"])
        register_experts_from_chapters(temp_db)
        expert = get_expert_for_query("gradient descent optimization", temp_db)
        assert expert is not None


# ── Loop controller tests ────────────────────────────────────────────


class TestLoopController:
    def test_l2_norm(self):
        assert _l2_norm([3.0, 4.0]) == 5.0
        assert _l2_norm([0.0]) == 0.0

    def test_vector_diff(self):
        assert _vector_diff([5.0, 3.0], [2.0, 1.0]) == [3.0, 2.0]

    def test_vector_diff_different_lengths(self):
        result = _vector_diff([1.0, 2.0, 3.0], [1.0])
        assert len(result) == 3
        assert result[0] == 0.0

    def test_convergence_check(self, temp_db):
        controller = LoopController(temp_db, epsilon=0.01)
        converged, delta = controller._check_convergence([0.001], [0.0])
        assert converged
        assert delta < 0.01

    def test_non_convergence(self, temp_db):
        controller = LoopController(temp_db, epsilon=0.001)
        converged, delta = controller._check_convergence([1.0], [0.0])
        assert not converged
        assert delta == 1.0

    def test_run_loop_converges(self, temp_db):
        controller = LoopController(
            temp_db, epsilon=0.01, max_iterations=20, patience=2
        )

        def decaying_step(iteration, state, metrics):
            new_state = [x * 0.3 for x in state]
            return new_state, {"iteration": iteration}

        result = controller.run_loop(
            competition="test_comp",
            step_fn=decaying_step,
            initial_state=[10.0],
        )
        assert result["status"] == "converged"
        assert result["patience_met"] is True

    def test_run_loop_max_iterations(self, temp_db):
        controller = LoopController(
            temp_db, epsilon=0.0001, max_iterations=3, patience=2
        )

        call_count = [0]

        def oscillating_step(iteration, state, metrics):
            call_count[0] += 1
            sign = 1 if iteration % 2 == 0 else -1
            new_state = [x + sign * 0.5 for x in state]
            return new_state, {"iteration": iteration}

        result = controller.run_loop(
            competition="osc_comp",
            step_fn=oscillating_step,
            initial_state=[1.0],
        )
        assert result["status"] == "max_iterations_reached"
        assert call_count[0] == 3

    def test_run_loop_error_handling(self, temp_db):
        controller = LoopController(temp_db, max_iterations=5)

        def failing_step(iteration, state, metrics):
            if iteration == 2:
                raise ValueError("Boom")
            return [x * 0.5 for x in state], {}

        result = controller.run_loop(
            competition="fail_comp",
            step_fn=failing_step,
            initial_state=[1.0],
        )
        assert result["status"] == "error"
        assert "Boom" in result["error"]

    def test_default_step_fn(self):
        state, metrics = default_step_fn(1, [2.0, 4.0], {})
        assert state == [1.0, 2.0]
        assert "state_norm" in metrics

    def test_loop_summary(self, temp_db):
        temp_db.save_loop_state("comp", 1, [1.0], {"loss": 0.5})
        temp_db.save_loop_state("comp", 2, [0.5], {"loss": 0.25})
        controller = LoopController(temp_db)
        summary = controller.get_loop_summary("comp")
        assert summary["iterations"] == 2
        assert len(summary["deltas"]) == 1

    def test_loop_summary_not_started(self, temp_db):
        controller = LoopController(temp_db)
        summary = controller.get_loop_summary("nonexistent")
        assert summary["status"] == "not_started"


# ── Docling worker tests ─────────────────────────────────────────────


class TestDoclingWorker:
    def test_extract_concepts_from_text(self):
        text = "We use gradient descent and backpropagation for neural network training"
        concepts = _extract_concepts_from_text(text)
        assert "gradient descent" in concepts
        assert "backpropagation" in concepts
        assert "neural network" in concepts

    def test_extract_concepts_empty(self):
        concepts = _extract_concepts_from_text("Hello world")
        assert len(concepts) == 0

    def test_chapter_id_from_path(self):
        path = Path("/tmp/08_ML_Systems")
        assert _chapter_id_from_path(path) == "08_ml_systems"

    def test_chapter_id_from_path_with_spaces(self):
        path = Path("/tmp/12 Deep Learning Basics")
        cid = _chapter_id_from_path(path)
        assert "deep_learning" in cid


# ── Skill generator tests ────────────────────────────────────────────


class TestSkillGenerator:
    def test_load_skills_config(self):
        from skill_generator import load_skills_config

        config = load_skills_config("skills.yaml")
        assert config["name"] == "mlsyseng-moe"
        assert "tools" in config
        assert len(config["tools"]) > 0

    def test_generate_skill_md(self):
        from skill_generator import generate_skill_md, load_skills_config

        config = load_skills_config("skills.yaml")
        md = generate_skill_md(config, "/tmp/server")
        assert "mlsyseng-moe" in md
        assert "extract-knowledge" in md
        assert "convergence" in md.lower()
