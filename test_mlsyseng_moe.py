"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import numpy as np
import pytest

# ── Database tests ────────────────────────────────────────────────────

from mlsyseng_mcp.database import Database


class TestDatabase:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_upsert_chapter(self):
        cid = self.db.upsert_chapter("ch01", "Chapter 1", "/tmp/ch01", "# Heading")
        assert cid > 0

        ch = self.db.get_chapter("ch01")
        assert ch is not None
        assert ch["title"] == "Chapter 1"
        assert ch["status"] == "extracted"

    def test_list_chapters(self):
        self.db.upsert_chapter("ch01", "Chapter 1")
        self.db.upsert_chapter("ch02", "Chapter 2")
        chapters = self.db.list_chapters()
        assert len(chapters) == 2

    def test_add_and_search_concepts(self):
        cid = self.db.upsert_chapter("ch01", "Chapter 1")
        self.db.add_concept(cid, "neural network", "A model inspired by the brain", "ml_concept")
        self.db.add_concept(cid, "gradient descent", "An optimization algorithm", "ml_concept")

        concepts = self.db.get_concepts(cid)
        assert len(concepts) == 2

        results = self.db.search_concepts("neural")
        assert len(results) == 1
        assert results[0]["term"] == "neural network"

    def test_upsert_expert(self):
        cid = self.db.upsert_chapter("ch01", "Chapter 1")
        expert = {
            "slug": "ch01_expert",
            "expert_name": "Chapter 1 Expert",
            "capabilities": ["Build models", "Feature engineering"],
            "skills": ["/skills/a", "/skills/b"],
            "strategy": "Baseline -> Tune",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001, "max_iterations": 10},
            "chapter_id": cid,
        }
        eid = self.db.upsert_expert(expert)
        assert eid > 0

        loaded = self.db.get_expert("ch01_expert")
        assert loaded is not None
        assert loaded["capabilities"] == ["Build models", "Feature engineering"]
        assert loaded["formula"]["objective"] == "minimize_loss"

    def test_list_experts(self):
        self.db.upsert_expert({
            "slug": "e1", "expert_name": "E1",
            "capabilities": [], "skills": [], "strategy": "",
            "formula": {}, "loop_config": {},
        })
        self.db.upsert_expert({
            "slug": "e2", "expert_name": "E2",
            "capabilities": [], "skills": [], "strategy": "",
            "formula": {}, "loop_config": {},
        })
        assert len(self.db.list_experts()) == 2

    def test_extraction_status(self):
        self.db.upsert_chapter("ch01", "Chapter 1", status="extracted")
        self.db.upsert_chapter("ch02", "Chapter 2", status="pending")
        status = self.db.get_extraction_status()
        assert status["total_chapters"] == 2
        assert status["extracted"] == 1
        assert status["pending"] == 1

    def test_log_event(self):
        self.db.log_event(None, "test_event", "detail text")
        stats = self.db.get_stats()
        assert len(stats["recent_events"]) == 1
        assert stats["recent_events"][0]["event"] == "test_event"


# ── Expert registry tests ────────────────────────────────────────────

from mlsyseng_mcp.expert_registry import ExpertRegistry


class TestExpertRegistry:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)
        self.registry = ExpertRegistry(self.db)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_register_from_chapter(self):
        cid = self.db.upsert_chapter("ch01", "ML Systems")
        expert = self.registry.register_from_chapter(
            "ch01", "ML Systems",
            ["neural network", "deep learning", "hyperparameter"],
            chapter_id=cid,
        )
        assert expert["slug"] == "ch01"
        assert "Design and train deep learning architectures" in expert["capabilities"]
        assert any("deep-learning" in s for s in expert["skills"])

    def test_get_experts_for_concepts(self):
        self.db.upsert_chapter("ch01", "Neural Networks")
        self.registry.register_from_chapter(
            "ch01", "Neural Networks", ["neural network", "deep learning"]
        )
        self.db.upsert_chapter("ch02", "Feature Engineering")
        self.registry.register_from_chapter(
            "ch02", "Feature Engineering", ["feature engineering", "preprocessing"]
        )

        matches = self.registry.get_experts_for_concepts(["deep learning"])
        assert len(matches) >= 1

    def test_export_expert_json(self):
        self.registry.register_from_chapter("ch01", "Test", ["model"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.registry.export_expert_json("ch01", output_dir=tmpdir)
            assert path is not None
            with open(path) as f:
                data = json.load(f)
            assert data["slug"] == "ch01"


# ── Loop controller tests ────────────────────────────────────────────

from mlsyseng_mcp.loop_controller import LoopController, build_competition_step_fn, default_metric_fn


class TestLoopController:
    def test_converges(self):
        controller = LoopController(epsilon=0.1, max_iterations=50, patience=2)

        def decaying_step(state, iteration):
            return state * 0.5

        initial = np.ones(4) * 10.0
        result = controller.run(initial, decaying_step, default_metric_fn)
        assert result.converged is True
        assert result.iterations < 50

    def test_does_not_converge(self):
        controller = LoopController(epsilon=1e-10, max_iterations=3, patience=2)

        def noisy_step(state, iteration):
            return state + np.random.default_rng(iteration).standard_normal(state.shape)

        initial = np.zeros(4)
        result = controller.run(initial, noisy_step)
        assert result.converged is False
        assert result.iterations == 3

    def test_build_competition_step_fn(self):
        step_fn = build_competition_step_fn(["skill_a", "skill_b"], "titanic")
        state = np.zeros(8)
        new_state = step_fn(state, 1)
        assert new_state.shape == (8,)
        assert not np.allclose(new_state, state)

    def test_history_recorded(self):
        controller = LoopController(epsilon=0.1, max_iterations=5, patience=1)
        step_fn = build_competition_step_fn(["s1"], "test")
        initial = np.ones(4)
        result = controller.run(initial, step_fn, default_metric_fn)
        assert len(result.history) > 0
        assert "l2_delta" in result.history[0]["metrics"]


# ── Docling worker tests ─────────────────────────────────────────────

from mlsyseng_mcp.docling_worker import extract_concepts, _slugify


class TestDoclingWorker:
    def test_slugify(self):
        assert _slugify("ML Systems Design") == "ml_systems_design"
        assert _slugify("08 - Deep Learning") == "08_deep_learning"

    def test_extract_concepts(self):
        text = """
        This chapter covers neural network architectures and gradient descent
        optimization. We also discuss regularization techniques and
        cross validation methodologies for model evaluation.
        """
        concepts = extract_concepts(text)
        terms = {c["term"] for c in concepts}
        assert "neural network" in terms
        assert "gradient descent" in terms
        assert "regularization" in terms
        assert "cross validation" in terms

    def test_extract_concepts_empty(self):
        concepts = extract_concepts("Nothing related to ML here.")
        assert len(concepts) == 0


# ── Embeddings tests (basic, no model download) ──────────────────────

from mlsyseng_mcp.embeddings import EmbeddingStore


class TestEmbeddingStore:
    def test_chunk(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = EmbeddingStore._chunk(text, size=20, overlap=5)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 20 for c in chunks)

    def test_chunk_empty(self):
        chunks = EmbeddingStore._chunk("", size=20, overlap=5)
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
