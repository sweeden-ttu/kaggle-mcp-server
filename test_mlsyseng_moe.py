"""Tests for the MLSysEng MoE system core modules."""

import json
import os
import sqlite3
import tempfile

import pytest

from src.mlsyseng_moe import database as db
from src.mlsyseng_moe import docling_worker
from src.mlsyseng_moe import expert_registry
from src.mlsyseng_moe import loop_controller


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = db.get_connection(path)
    db.init_db(conn)
    yield conn
    conn.close()
    os.unlink(path)


class TestDatabase:
    def test_init_creates_tables(self, tmp_db):
        tables = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in tables}
        assert "chapters" in names
        assert "concepts" in names
        assert "experts" in names
        assert "extraction_log" in names

    def test_upsert_and_get_chapter(self, tmp_db):
        cid = db.upsert_chapter(tmp_db, "01", "Introduction", "/path/ch01", "content", 10)
        assert cid > 0
        ch = db.get_chapter(tmp_db, cid)
        assert ch["title"] == "Introduction"
        assert ch["page_count"] == 10

    def test_upsert_chapter_idempotent(self, tmp_db):
        id1 = db.upsert_chapter(tmp_db, "01", "Intro", "/path", "v1")
        id2 = db.upsert_chapter(tmp_db, "01", "Intro Updated", "/path", "v2")
        assert id1 == id2
        ch = db.get_chapter(tmp_db, id1)
        assert ch["title"] == "Intro Updated"

    def test_list_chapters(self, tmp_db):
        db.upsert_chapter(tmp_db, "02", "Chapter Two", "/p2")
        db.upsert_chapter(tmp_db, "01", "Chapter One", "/p1")
        chapters = db.list_chapters(tmp_db)
        assert len(chapters) == 2
        assert chapters[0]["chapter_number"] == "01"

    def test_concept_crud(self, tmp_db):
        cid = db.upsert_chapter(tmp_db, "01", "Intro", "/p")
        db.upsert_concept(tmp_db, cid, "Gradient Descent", "An opt algorithm", "optimization")
        concepts = db.list_concepts(tmp_db, cid)
        assert len(concepts) == 1
        assert concepts[0]["name"] == "Gradient Descent"

    def test_expert_crud(self, tmp_db):
        expert_data = {
            "slug": "01_intro",
            "expert_name": "01_Introduction",
            "capabilities": ["Build baseline models"],
            "skills": ["/skills/kaggle-preprocessor"],
            "strategy": "Baseline -> Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        eid = db.upsert_expert(tmp_db, expert_data)
        assert eid > 0

        expert = db.get_expert(tmp_db, "01_intro")
        assert expert["expert_name"] == "01_Introduction"
        assert isinstance(expert["capabilities"], list)
        assert expert["capabilities"] == ["Build baseline models"]

    def test_list_experts(self, tmp_db):
        db.upsert_expert(tmp_db, {"slug": "a", "expert_name": "A"})
        db.upsert_expert(tmp_db, {"slug": "b", "expert_name": "B"})
        experts = db.list_experts(tmp_db)
        assert len(experts) == 2

    def test_stats(self, tmp_db):
        stats = db.get_stats(tmp_db)
        assert stats["total_chapters"] == 0
        assert stats["total_experts"] == 0

        db.upsert_chapter(tmp_db, "01", "T", "/p", "content")
        db.upsert_expert(tmp_db, {"slug": "x", "expert_name": "X"})
        stats = db.get_stats(tmp_db)
        assert stats["total_chapters"] == 1
        assert stats["extracted_chapters"] == 1
        assert stats["total_experts"] == 1


class TestDoclingWorker:
    def test_discover_chapters_empty(self, tmp_path):
        chapters = docling_worker.discover_chapters(str(tmp_path))
        assert chapters == []

    def test_discover_chapters(self, tmp_path):
        (tmp_path / "01_Introduction").mkdir()
        (tmp_path / "02_Deep Learning").mkdir()
        (tmp_path / "random_file.txt").touch()
        chapters = docling_worker.discover_chapters(str(tmp_path))
        assert len(chapters) == 2
        assert chapters[0]["chapter_number"] == "01"
        assert chapters[1]["chapter_number"] == "02"

    def test_find_pdfs(self, tmp_path):
        (tmp_path / "file1.pdf").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file2.pdf").touch()
        (tmp_path / "notes.txt").touch()
        pdfs = docling_worker.find_pdfs(str(tmp_path))
        assert len(pdfs) == 2

    def test_extract_concepts(self):
        text = "This chapter covers gradient descent and backpropagation for neural network training."
        concepts = docling_worker.extract_concepts(text)
        names = [c["name"] for c in concepts]
        assert "Gradient Descent" in names
        assert "Backpropagation" in names
        assert "Neural Network" in names

    def test_categorize_concept(self):
        assert docling_worker._categorize_concept("gradient descent") == "optimization"
        assert docling_worker._categorize_concept("transformer") == "architecture"
        assert docling_worker._categorize_concept("dropout") == "training"
        assert docling_worker._categorize_concept("f1 score") == "evaluation"


class TestExpertRegistry:
    def test_slugify(self):
        assert expert_registry._slugify("ML Systems") == "ml_systems"
        assert expert_registry._slugify("Deep Learning!") == "deep_learning"

    def test_infer_capabilities(self):
        concepts = [
            {"name": "Gradient Descent", "category": "optimization"},
            {"name": "Neural Network", "category": "architecture"},
        ]
        caps = expert_registry._infer_capabilities(concepts)
        assert len(caps) >= 4  # 2 base + 2 inferred

    def test_infer_skills(self):
        concepts = [
            {"name": "X", "category": "optimization"},
            {"name": "Y", "category": "evaluation"},
        ]
        skills = expert_registry._infer_skills(concepts)
        assert any("model-trainer" in s for s in skills)
        assert any("evaluator" in s for s in skills)

    def test_create_expert_from_chapter(self):
        concepts = [
            {"name": "Gradient Descent", "category": "optimization"},
        ]
        expert = expert_registry.create_expert_from_chapter("08", "ML Systems", concepts)
        assert expert["slug"] == "08_ml_systems"
        assert expert["expert_name"] == "08_ML_Systems"
        assert len(expert["capabilities"]) > 0
        assert expert["loop_config"]["epsilon"] == 0.001

    def test_register_experts(self, tmp_db):
        results = [
            {
                "chapter_number": "01",
                "title": "Intro",
                "concepts": [{"name": "A", "category": "general"}],
            },
            {
                "chapter_number": "02",
                "title": "Advanced",
                "concepts": [],
            },
        ]
        db.upsert_chapter(tmp_db, "01", "Intro", "/p1")
        experts = expert_registry.register_experts_from_extraction(results, conn=tmp_db)
        assert len(experts) == 1  # chapter 02 skipped (no concepts)

    def test_query_expert(self, tmp_db):
        cid = db.upsert_chapter(tmp_db, "01", "Intro", "/p", "Some content")
        db.upsert_concept(tmp_db, cid, "Gradient Descent", "opt", "optimization")
        db.upsert_expert(tmp_db, {
            "slug": "01_intro",
            "expert_name": "01_Introduction",
            "chapter_id": cid,
            "capabilities": ["Build models"],
            "strategy": "Baseline",
        })
        result = expert_registry.query_expert(tmp_db, "01_intro", "How to optimize?")
        assert result["expert"] == "01_Introduction"
        assert "Gradient Descent" in result["concepts"]

    @pytest.fixture
    def tmp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        conn = db.get_connection(path)
        db.init_db(conn)
        yield conn
        conn.close()
        os.unlink(path)


class TestLoopController:
    def test_l2_norm_diff(self):
        assert loop_controller.l2_norm_diff([1, 0], [1, 0]) == 0.0
        assert abs(loop_controller.l2_norm_diff([0, 0], [3, 4]) - 5.0) < 1e-9

    def test_l2_norm_diff_unequal_lengths(self):
        diff = loop_controller.l2_norm_diff([1], [1, 2])
        assert abs(diff - 2.0) < 1e-9

    def test_single_step(self):
        loop = loop_controller.ConvergenceLoop(epsilon=0.01, max_iterations=5, patience=2)
        result = loop.step({"loss": 1.0})
        assert result["iteration"] == 0
        assert result["converged"] is False
        assert result["reason"] == "first_iteration"

    def test_convergence(self):
        loop = loop_controller.ConvergenceLoop(epsilon=0.01, max_iterations=20, patience=2)
        loop.step({"loss": 1.0, "acc": 0.5})
        loop.step({"loss": 0.999, "acc": 0.501})
        result = loop.step({"loss": 0.998, "acc": 0.502})
        assert result["converged"] is True
        assert result["should_stop"] is True

    def test_max_iterations_stop(self):
        loop = loop_controller.ConvergenceLoop(epsilon=0.0001, max_iterations=3, patience=2)
        for i in range(4):
            result = loop.step({"loss": 1.0 / (i + 1)})
        assert result["should_stop"] is True
        assert result["reason"] == "max_iterations_reached"

    def test_run_converges(self):
        loop = loop_controller.ConvergenceLoop(epsilon=0.01, max_iterations=20, patience=3)

        def step_fn(i, prev):
            return {"loss": 0.5 + 0.001 * (0.9 ** i)}

        result = loop.run(step_fn)
        assert result["converged"] is True

    def test_run_hits_max(self):
        loop = loop_controller.ConvergenceLoop(epsilon=0.0001, max_iterations=3, patience=2)

        def step_fn(i, prev):
            return {"loss": 1.0 / (i + 1)}

        result = loop.run(step_fn)
        assert result["reason"] == "max_iterations_reached"

    def test_create_loop(self):
        loop = loop_controller.create_loop({"epsilon": 0.05, "max_iterations": 5, "patience": 2})
        assert loop.epsilon == 0.05
        assert loop.max_iterations == 5
        assert loop.patience == 2

    def test_history_tracking(self):
        loop = loop_controller.ConvergenceLoop()
        loop.step({"loss": 1.0})
        loop.step({"loss": 0.5})
        assert len(loop.history) == 2
        assert loop.history[0]["metrics"]["loss"] == 1.0
