"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp.database import MLSysEngDB
from mlsyseng_mcp.docling_worker import extract_concepts, discover_chapters
from mlsyseng_mcp.expert_registry import (
    ExpertRegistry,
    _slugify,
    _infer_capabilities,
    _infer_skills,
    _infer_formula,
)
from mlsyseng_mcp.loop_controller import LoopController, l2_norm


@pytest.fixture
def tmp_db():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = MLSysEngDB(db_path)
    yield db
    db.close()


# ── Database tests ──────────────────────────────────────────────────


class TestMLSysEngDB:
    def test_upsert_and_get_chapter(self, tmp_db):
        ch_id = tmp_db.upsert_chapter(
            "Chapter1", "/tmp/ch1", "# Markdown", ["concept1", "concept2"], 5
        )
        assert ch_id > 0
        ch = tmp_db.get_chapter("Chapter1")
        assert ch is not None
        assert ch["chapter_name"] == "Chapter1"
        assert ch["concepts"] == ["concept1", "concept2"]
        assert ch["page_count"] == 5

    def test_upsert_chapter_updates(self, tmp_db):
        tmp_db.upsert_chapter("Ch1", "/a", "# Old", ["old"], 1)
        tmp_db.upsert_chapter("Ch1", "/b", "# New", ["new"], 2)
        ch = tmp_db.get_chapter("Ch1")
        assert ch["source_path"] == "/b"
        assert ch["concepts"] == ["new"]

    def test_list_chapters(self, tmp_db):
        tmp_db.upsert_chapter("B", "/b", "B", [], 0)
        tmp_db.upsert_chapter("A", "/a", "A", [], 0)
        chapters = tmp_db.list_chapters()
        assert len(chapters) == 2
        assert chapters[0]["chapter_name"] == "A"

    def test_upsert_and_get_expert(self, tmp_db):
        expert_id = tmp_db.upsert_expert({
            "expert_name": "TestExpert",
            "slug": "test_expert",
            "capabilities": ["cap1"],
            "skills": ["/skill1"],
            "strategy": "strat",
            "formula": {"objective": "min_loss"},
            "loop_config": {"epsilon": 0.001},
        })
        assert expert_id > 0
        expert = tmp_db.get_expert("test_expert")
        assert expert["capabilities"] == ["cap1"]
        assert expert["formula"]["objective"] == "min_loss"

    def test_list_experts(self, tmp_db):
        tmp_db.upsert_expert({
            "expert_name": "E1", "slug": "e1",
            "capabilities": [], "skills": [], "strategy": "",
            "formula": {}, "loop_config": {},
        })
        experts = tmp_db.list_experts()
        assert len(experts) == 1

    def test_convergence_runs(self, tmp_db):
        tmp_db.insert_convergence_run("comp", 1, [0.5, 0.5], 0.1, False)
        tmp_db.insert_convergence_run("comp", 2, [0.5, 0.5], 0.001, True)
        history = tmp_db.get_convergence_history("comp")
        assert len(history) == 2
        assert history[1]["converged"] == 1

    def test_extraction_status(self, tmp_db):
        tmp_db.update_extraction_status("ch1", "running", 50.0)
        tmp_db.update_extraction_status("ch1", "completed", 100.0)
        statuses = tmp_db.get_extraction_status()
        assert len(statuses) == 1
        assert statuses[0]["status"] == "completed"

    def test_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats["chapters"] == 0
        assert stats["experts"] == 0


# ── Docling worker tests ───────────────────────────────────────────


class TestDoclingWorker:
    def test_extract_concepts_headings(self):
        md = "# Deep Learning\n## Neural Networks\n### Loss Functions"
        concepts = extract_concepts(md)
        assert "Deep Learning" in concepts
        assert "Neural Networks" in concepts

    def test_extract_concepts_bold(self):
        md = "The **gradient descent** algorithm is used for **optimization**."
        concepts = extract_concepts(md)
        assert "gradient descent" in concepts

    def test_extract_concepts_ml_keywords(self):
        md = "We discuss gradient descent, backpropagation, and cross-validation."
        concepts = extract_concepts(md)
        assert any("Gradient Descent" in c for c in concepts)

    def test_extract_concepts_deduplication(self):
        md = "# Neural Network\nThe **neural network** is a **neural network** model."
        concepts = extract_concepts(md)
        count = sum(1 for c in concepts if c.lower() == "neural network")
        assert count == 1

    def test_discover_chapters_missing_path(self):
        chapters = discover_chapters("/nonexistent/path")
        assert chapters == []


# ── Expert registry tests ──────────────────────────────────────────


class TestExpertRegistry:
    def test_slugify(self):
        assert _slugify("08_ML Systems") == "08_ml_systems"
        assert _slugify("Deep Learning Fundamentals") == "deep_learning_fundamentals"
        assert _slugify("A & B!") == "a_b"

    def test_infer_capabilities(self):
        caps = _infer_capabilities(["Neural Network", "Regularization"])
        assert "Build baseline models quickly" in caps
        assert any("neural" in c.lower() for c in caps)

    def test_infer_skills(self):
        skills = _infer_skills(["Neural Network", "Deep Learning"])
        assert any("nn-trainer" in s for s in skills)

    def test_infer_skills_default(self):
        skills = _infer_skills(["Unknown Concept"])
        assert len(skills) >= 2

    def test_infer_formula_classification(self):
        formula = _infer_formula(["Classification", "SVM"])
        assert formula["metrics"] == ["accuracy", "f1_score", "auc_roc"]

    def test_infer_formula_regression(self):
        formula = _infer_formula(["Regression", "Linear"])
        assert "rmse" in formula["metrics"]

    def test_register_and_query(self, tmp_db):
        registry = ExpertRegistry(db=tmp_db)
        tmp_db.upsert_chapter("Ch1", "/tmp", "# Test", ["ML"], 1)
        expert = registry.register_expert_from_chapter("Ch1", ["Neural Network", "Classification"])
        assert expert["slug"] == "ch1"
        found = registry.get_expert("ch1")
        assert found is not None

    def test_ask_expert(self, tmp_db):
        registry = ExpertRegistry(db=tmp_db)
        tmp_db.upsert_expert({
            "expert_name": "TestE", "slug": "teste",
            "capabilities": ["cap"], "skills": [], "strategy": "s",
            "formula": {}, "loop_config": {},
        })
        resp = registry.ask_expert("teste", "How to optimize?")
        assert resp["expert"] == "TestE"

    def test_ask_nonexistent_expert(self, tmp_db):
        registry = ExpertRegistry(db=tmp_db)
        resp = registry.ask_expert("nonexistent", "question")
        assert "error" in resp


# ── Loop controller tests ──────────────────────────────────────────


class TestLoopController:
    def test_l2_norm_zero(self):
        assert l2_norm([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_l2_norm_value(self):
        result = l2_norm([1.0, 0.0], [0.0, 1.0])
        assert abs(result - 2**0.5) < 1e-6

    def test_l2_norm_mismatch(self):
        with pytest.raises(ValueError):
            l2_norm([1.0], [1.0, 2.0])

    def test_convergence(self, tmp_db):
        controller = LoopController(db=tmp_db, epsilon=0.01, max_iterations=50, patience=3)
        experts = [
            {"expert_name": "A", "slug": "a", "skills": [], "strategy": "s"},
            {"expert_name": "B", "slug": "b", "skills": [], "strategy": "s"},
        ]
        result = controller.run("test", experts)
        assert result["converged"] is True
        assert result["iterations"] <= 50

    def test_max_iterations_reached(self, tmp_db):
        controller = LoopController(db=tmp_db, epsilon=1e-20, max_iterations=3, patience=10)
        experts = [{"expert_name": "A", "slug": "a", "skills": [], "strategy": "s"}]
        result = controller.run("test", experts)
        assert result["iterations"] == 3

    def test_build_entry(self, tmp_db):
        controller = LoopController(db=tmp_db)
        experts = [
            {"expert_name": "A", "slug": "a", "skills": ["/s1"], "strategy": "S1", "capabilities": ["C1"]},
        ]
        entry = controller.build_competition_entry("comp", experts)
        assert entry["competition"] == "comp"
        assert "/s1" in entry["combined_skills"]
        assert "recommended_approach" in entry

    def test_expert_weights_sum_to_one(self, tmp_db):
        controller = LoopController(db=tmp_db, patience=3)
        experts = [
            {"expert_name": f"E{i}", "slug": f"e{i}", "skills": [], "strategy": "s"}
            for i in range(5)
        ]
        result = controller.run("test", experts)
        total = sum(w["weight"] for w in result["expert_weights"])
        assert abs(total - 1.0) < 1e-6
