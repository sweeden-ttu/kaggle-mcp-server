"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp.database import MoEDatabase
from mlsyseng_mcp.docling_worker import extract_concepts
from mlsyseng_mcp.expert_registry import (
    _slugify,
    create_expert_from_chapter,
    register_experts_from_db,
)
from mlsyseng_mcp.loop_controller import LoopController, initial_state_vector, l2_norm


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = MoEDatabase(path)
    yield db
    db.close()
    os.unlink(path)


# ── L2 norm tests ────────────────────────────────────────────────


class TestL2Norm:
    def test_zero_vectors(self):
        assert l2_norm([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_unit_vector(self):
        assert abs(l2_norm([1.0, 0.0], [0.0, 0.0]) - 1.0) < 1e-9

    def test_pythagorean(self):
        assert abs(l2_norm([3.0, 4.0], [0.0, 0.0]) - 5.0) < 1e-9

    def test_identical_vectors(self):
        assert l2_norm([1.5, 2.5, 3.5], [1.5, 2.5, 3.5]) == 0.0

    def test_different_lengths(self):
        result = l2_norm([1.0], [0.0, 1.0])
        assert result > 0


# ── Database tests ───────────────────────────────────────────────


class TestDatabase:
    def test_upsert_and_get_chapter(self, tmp_db):
        tmp_db.upsert_chapter("ch_01", 1, "Introduction", markdown="# Intro")
        ch = tmp_db.get_chapter("ch_01")
        assert ch is not None
        assert ch["title"] == "Introduction"
        assert ch["chapter_num"] == 1

    def test_list_chapters(self, tmp_db):
        tmp_db.upsert_chapter("ch_01", 1, "Intro")
        tmp_db.upsert_chapter("ch_02", 2, "Linear Models")
        chapters = tmp_db.list_chapters()
        assert len(chapters) == 2
        assert chapters[0]["chapter_num"] == 1

    def test_extraction_status(self, tmp_db):
        tmp_db.upsert_chapter("ch_01", 1, "Intro", status="extracted")
        tmp_db.upsert_chapter("ch_02", 2, "Models", status="pending")
        status = tmp_db.get_extraction_status()
        assert status["total"] == 2
        assert status["extracted"] == 1
        assert status["pending"] == 1

    def test_upsert_and_get_expert(self, tmp_db):
        expert = create_expert_from_chapter("ch_01", 1, "Intro", ["regression"])
        tmp_db.upsert_expert(expert)
        result = tmp_db.get_expert(expert["expert_id"])
        assert result is not None
        assert result["slug"] == "01_intro"

    def test_get_expert_by_slug(self, tmp_db):
        expert = create_expert_from_chapter("ch_05", 5, "Deep Learning", ["neural network"])
        tmp_db.upsert_expert(expert)
        result = tmp_db.get_expert_by_slug("05_deep_learning")
        assert result is not None
        assert result["expert_name"] == "05_Deep Learning"

    def test_loop_state(self, tmp_db):
        state = {
            "loop_id": "test_loop",
            "competition": "titanic",
            "iteration": 0,
            "state_vector": [0.0, 0.0],
            "prev_vector": [],
            "l2_norm": float("inf"),
            "converged": False,
            "patience_cnt": 0,
        }
        tmp_db.upsert_loop_state(state)
        result = tmp_db.get_loop_state("test_loop")
        assert result is not None
        assert result["competition"] == "titanic"
        assert not result["converged"]

    def test_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats["chapters"] == 0
        assert stats["experts"] == 0


# ── Concept extraction tests ─────────────────────────────────────


class TestConceptExtraction:
    def test_extracts_gradient_descent(self):
        concepts = extract_concepts("Apply gradient descent to optimize the loss")
        assert any("gradient" in c.lower() for c in concepts)

    def test_extracts_neural_network(self):
        concepts = extract_concepts("Train a neural network for image recognition")
        assert any("neural" in c.lower() for c in concepts)

    def test_no_concepts_in_unrelated_text(self):
        concepts = extract_concepts("The weather today is sunny and warm.")
        assert len(concepts) == 0

    def test_multiple_concepts(self):
        text = "Use cross-validation with XGBoost and feature engineering"
        concepts = extract_concepts(text)
        assert len(concepts) >= 2


# ── Expert registry tests ────────────────────────────────────────


class TestExpertRegistry:
    def test_slugify(self):
        assert _slugify("08 ML Systems") == "08_ml_systems"
        assert _slugify("Deep Learning") == "deep_learning"
        assert _slugify("A & B") == "a_b"

    def test_create_expert_basic(self):
        expert = create_expert_from_chapter("ch_01", 1, "Intro", [])
        assert expert["slug"] == "01_intro"
        assert expert["expert_name"] == "01_Intro"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0

    def test_create_expert_with_concepts(self):
        expert = create_expert_from_chapter(
            "ch_05", 5, "Deep Learning", ["neural network", "gradient descent"]
        )
        assert "05_deep_learning" == expert["slug"]
        assert any("neural" in c.lower() for c in expert["capabilities"])

    def test_register_experts_from_db(self, tmp_db):
        tmp_db.upsert_chapter(
            "ch_01", 1, "Intro",
            markdown="content",
            concepts=["regression"],
            status="extracted",
        )
        tmp_db.upsert_chapter(
            "ch_02", 2, "Linear",
            markdown="content",
            concepts=["gradient descent"],
            status="extracted",
        )
        experts = register_experts_from_db(tmp_db)
        assert len(experts) == 2


# ── Loop controller tests ────────────────────────────────────────


class TestLoopController:
    def test_create_loop(self, tmp_db):
        lc = LoopController(tmp_db)
        state = lc.create_loop("titanic", 3)
        assert state["competition"] == "titanic"
        assert state["state_vector"] == [0.0, 0.0, 0.0]
        assert state["iteration"] == 0

    def test_single_step(self, tmp_db):
        lc = LoopController(tmp_db)
        state = lc.create_loop("titanic", 2)
        result = lc.step(state["loop_id"], [0.5, 0.3])
        assert result["iteration"] == 1
        assert result["l2_norm"] > 0

    def test_convergence(self, tmp_db):
        lc = LoopController(tmp_db, epsilon=0.01, max_iterations=50, patience=3)
        experts = [
            {"relevance_score": 0.9},
            {"relevance_score": 0.7},
        ]

        def refine_fn(iteration, expert_list, current_state):
            return [
                (current_state[i] if i < len(current_state) else 0.0)
                + e.get("relevance_score", 0.5) * (0.5 ** (iteration + 1))
                for i, e in enumerate(expert_list)
            ]

        result = lc.run_loop("titanic", experts, refine_fn)
        assert result["converged"]
        assert result["l2_norm"] < 0.01

    def test_max_iterations_exit(self, tmp_db):
        lc = LoopController(tmp_db, epsilon=1e-20, max_iterations=5, patience=3)

        def refine_fn(iteration, experts, current):
            return [c + 1.0 for c in current]

        experts = [{"name": "A"}]
        result = lc.run_loop("test", experts, refine_fn)
        assert result["iteration"] == 5
        assert result["converged"]

    def test_initial_state_vector(self):
        assert initial_state_vector(0) == [0.0]
        assert initial_state_vector(5) == [0.0, 0.0, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
