"""Tests for the MLSysEng MoE system.

Covers database, docling_worker, expert_registry, embeddings,
loop_controller, and skill_generator.
"""

import json
import os
import tempfile
import shutil

import pytest
import numpy as np

# ── Database tests ─────────────────────────────────────────────


class TestMLSysEngDB:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")

        from mlsyseng_mcp.database import MLSysEngDB
        self.db = MLSysEngDB(db_path=self.db_path)

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_upsert_and_get_chapter(self):
        ch = self.db.upsert_chapter(
            chapter_id="ch001",
            folder_name="01_Intro",
            title="Introduction",
            content_md="# Intro\nSome content",
            concepts=["neural network", "gradient descent"],
            pdf_path="/tmp/test.pdf",
        )
        assert ch is not None
        assert ch["chapter_id"] == "ch001"
        assert ch["title"] == "Introduction"
        assert "neural network" in ch["concepts"]

    def test_list_chapters(self):
        self.db.upsert_chapter("a", "a_folder", "A")
        self.db.upsert_chapter("b", "b_folder", "B")
        chapters = self.db.list_chapters()
        assert len(chapters) == 2

    def test_upsert_and_get_expert(self):
        expert_data = {
            "expert_id": "exp001",
            "chapter_id": "ch001",
            "expert_name": "ML Systems Expert",
            "slug": "ml_systems",
            "capabilities": ["Build baseline models"],
            "skills": ["/path/to/skill"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        exp = self.db.upsert_expert(expert_data)
        assert exp["expert_name"] == "ML Systems Expert"
        assert exp["slug"] == "ml_systems"

    def test_get_expert_by_slug(self):
        self.db.upsert_expert({
            "expert_id": "exp002",
            "chapter_id": "ch002",
            "expert_name": "Deep Learning",
            "slug": "deep_learning",
        })
        exp = self.db.get_expert_by_slug("deep_learning")
        assert exp is not None
        assert exp["expert_name"] == "Deep Learning"

    def test_list_experts(self):
        self.db.upsert_expert({
            "expert_id": "e1", "chapter_id": "c1",
            "expert_name": "A", "slug": "a",
        })
        self.db.upsert_expert({
            "expert_id": "e2", "chapter_id": "c2",
            "expert_name": "B", "slug": "b",
        })
        experts = self.db.list_experts()
        assert len(experts) == 2

    def test_create_and_get_run(self):
        run = self.db.create_run("run001", "titanic", ["e1", "e2"])
        assert run["competition"] == "titanic"
        assert run["run_id"] == "run001"

    def test_update_run(self):
        self.db.create_run("run002", "titanic", ["e1"])
        self.db.update_run(
            "run002",
            converged=True,
            final_norm=0.0005,
            iterations=5,
            finished_at="2026-01-01T00:00:00",
        )
        run = self.db.get_run("run002")
        assert run["converged"] == 1
        assert run["final_norm"] == pytest.approx(0.0005)
        assert run["iterations"] == 5

    def test_get_stats(self):
        self.db.upsert_chapter("c1", "folder", "Title")
        stats = self.db.get_stats()
        assert stats["chapters"] == 1
        assert stats["experts"] == 0

    def test_add_embedding_meta(self):
        self.db.add_embedding_meta("emb1", "ch001", 0, "some chunk text")
        stats = self.db.get_stats()
        assert stats["embeddings_meta"] == 1


# ── Docling worker tests ──────────────────────────────────────


class TestDoclingWorker:
    def test_extract_concepts(self):
        from mlsyseng_mcp.docling_worker import _extract_concepts
        text = (
            "We use gradient descent for optimization. "
            "The neural network includes batch normalization and dropout. "
            "Cross-validation helps prevent overfitting."
        )
        concepts = _extract_concepts(text)
        assert len(concepts) > 0
        assert any("gradient descent" in c for c in concepts)
        assert any("neural network" in c for c in concepts)

    def test_chapter_id_deterministic(self):
        from mlsyseng_mcp.docling_worker import _chapter_id_from_folder
        id1 = _chapter_id_from_folder("08_ML Systems")
        id2 = _chapter_id_from_folder("08_ML Systems")
        assert id1 == id2

    def test_title_from_folder(self):
        from mlsyseng_mcp.docling_worker import _title_from_folder
        assert _title_from_folder("08_ML Systems") == "ML Systems"
        assert _title_from_folder("01_Introduction") == "Introduction"

    def test_scan_chapter_folders_missing_path(self):
        from mlsyseng_mcp.docling_worker import scan_chapter_folders
        chapters = scan_chapter_folders("/nonexistent/path")
        assert chapters == []


# ── Expert registry tests ─────────────────────────────────────


class TestExpertRegistry:
    def test_slugify(self):
        from mlsyseng_mcp.expert_registry import _slugify
        assert _slugify("ML Systems") == "ml_systems"
        assert _slugify("08 Deep Learning!") == "08_deep_learning"

    def test_infer_skills(self):
        from mlsyseng_mcp.expert_registry import _infer_skills
        skills = _infer_skills(["neural network", "gradient descent"])
        assert len(skills) > 0
        assert any("model-trainer" in s for s in skills)

    def test_infer_skills_empty(self):
        from mlsyseng_mcp.expert_registry import _infer_skills
        skills = _infer_skills([])
        assert len(skills) >= 2

    def test_infer_capabilities(self):
        from mlsyseng_mcp.expert_registry import _infer_capabilities
        caps = _infer_capabilities(["neural network", "optimization"], "Deep Learning")
        assert any("Deep Learning" in c for c in caps)
        assert any("neural network" in c.lower() for c in caps)

    def test_infer_formula(self):
        from mlsyseng_mcp.expert_registry import _infer_formula
        formula = _infer_formula(["classification", "precision"])
        assert formula["objective"] == "minimize_validation_loss"
        assert "precision" in formula["metrics"]

    def test_create_expert_from_chapter(self):
        from mlsyseng_mcp.expert_registry import create_expert_from_chapter
        chapter = {
            "chapter_id": "ch001",
            "title": "ML Systems",
            "folder_name": "08_ML Systems",
            "concepts": ["neural network", "gradient descent", "ensemble"],
        }
        expert = create_expert_from_chapter(chapter)
        assert expert["expert_name"] is not None
        assert expert["slug"] is not None
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert expert["formula"]["objective"] == "minimize_validation_loss"

    def test_register_all_experts(self):
        from mlsyseng_mcp.expert_registry import register_all_experts
        chapters = [
            {"chapter_id": "c1", "title": "A", "folder_name": "01_A", "concepts": []},
            {"chapter_id": "c2", "title": "B", "folder_name": "02_B", "concepts": ["ensemble"]},
        ]
        experts = register_all_experts(chapters)
        assert len(experts) == 2


# ── Loop controller tests ─────────────────────────────────────


class TestLoopController:
    def test_l2_norm(self):
        from mlsyseng_mcp.loop_controller import l2_norm
        assert l2_norm([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
        assert l2_norm([1.0, 0.0], [0.0, 0.0]) == pytest.approx(1.0)
        assert l2_norm([3.0, 4.0], [0.0, 0.0]) == pytest.approx(5.0)

    def test_l2_norm_different_lengths(self):
        from mlsyseng_mcp.loop_controller import l2_norm
        norm = l2_norm([1.0, 2.0, 3.0], [1.0, 2.0])
        assert norm == pytest.approx(0.0)

    def test_convergence_loop_converges(self):
        from mlsyseng_mcp.loop_controller import ConvergenceLoop

        experts = [
            {"expert_id": "e1", "formula": {"metrics": ["accuracy"]}, "metadata": {"concepts": ["nn"]}},
            {"expert_id": "e2", "formula": {"metrics": ["f1"]}, "metadata": {"concepts": ["ensemble"]}},
        ]

        def constant_step(state, iteration, experts):
            return state

        loop = ConvergenceLoop(
            competition="test",
            experts=experts,
            epsilon=0.01,
            max_iterations=10,
            patience=2,
        )
        result = loop.run(step_callback=constant_step)
        assert result["converged"] is True
        assert result["final_norm"] == pytest.approx(0.0)

    def test_convergence_loop_max_iterations(self):
        from mlsyseng_mcp.loop_controller import ConvergenceLoop

        experts = [{"expert_id": "e1", "formula": {"metrics": []}, "metadata": {"concepts": []}}]

        def diverging_step(state, iteration, experts):
            return [s + 1.0 for s in state]

        loop = ConvergenceLoop(
            competition="test",
            experts=experts,
            epsilon=0.001,
            max_iterations=3,
            patience=3,
        )
        result = loop.run(step_callback=diverging_step)
        assert result["converged"] is False
        assert result["iterations"] == 3

    def test_build_competition_entry(self):
        from mlsyseng_mcp.loop_controller import build_competition_entry

        experts = [
            {
                "expert_id": "e1",
                "expert_name": "ML Systems",
                "slug": "ml_systems",
                "chapter_id": "c1",
                "skills": ["/skills/kaggle-preprocessor"],
                "capabilities": ["Build models"],
                "strategy": "Baseline → Submit",
                "formula": {"metrics": ["accuracy"]},
            },
        ]
        plan = build_competition_entry("titanic", experts)
        assert plan["competition"] == "titanic"
        assert len(plan["experts_used"]) > 0
        assert len(plan["skills"]) > 0


# ── Skill generator tests ─────────────────────────────────────


class TestSkillGenerator:
    def test_load_skills_yaml(self):
        from skill_generator import load_skills_yaml
        skills_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
        if os.path.exists(skills_path):
            skills = load_skills_yaml(skills_path)
            assert skills["name"] == "mlsyseng-moe"
            assert "mcp_server" in skills
            assert "tools" in skills
            assert len(skills["tools"]) >= 8


# ── Integration test ──────────────────────────────────────────


class TestIntegration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_pipeline_no_pdfs(self):
        """Test the full pipeline when no PDF chapters are available."""
        from mlsyseng_mcp.database import MLSysEngDB
        from mlsyseng_mcp.expert_registry import register_all_experts
        from mlsyseng_mcp.loop_controller import ConvergenceLoop, build_competition_entry

        db = MLSysEngDB(db_path=self.db_path)

        db.upsert_chapter(
            chapter_id="ch01",
            folder_name="01_Introduction",
            title="Introduction to ML",
            content_md="Machine learning involves neural networks and optimization.",
            concepts=["neural network", "optimization", "gradient descent"],
        )
        db.upsert_chapter(
            chapter_id="ch02",
            folder_name="02_Deep_Learning",
            title="Deep Learning",
            content_md="Deep learning uses convolutional and recurrent networks.",
            concepts=["deep learning", "convolutional", "recurrent", "backpropagation"],
        )

        chapters = db.list_chapters()
        assert len(chapters) == 2

        experts = register_all_experts(chapters, db=db)
        assert len(experts) == 2

        stored_experts = db.list_experts()
        assert len(stored_experts) == 2

        plan = build_competition_entry("titanic", stored_experts)
        assert plan["competition"] == "titanic"
        assert len(plan["experts_used"]) > 0

        def converging_step(state, iteration, experts):
            decay = 0.3 ** iteration
            return [s + 0.001 * decay for s in state]

        loop = ConvergenceLoop(
            competition="titanic",
            experts=stored_experts,
            epsilon=0.01,
            max_iterations=20,
            patience=2,
            db=db,
        )
        result = loop.run(step_callback=converging_step)
        assert result["converged"] is True

        run = db.get_run(result["run_id"])
        assert run is not None
        assert run["converged"] == 1

        stats = db.get_stats()
        assert stats["chapters"] == 2
        assert stats["experts"] == 2
        assert stats["convergence_runs"] == 1

        db.close()
