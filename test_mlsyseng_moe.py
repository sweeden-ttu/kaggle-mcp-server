"""Tests for MLSysEng MoE system components."""

import json
import math
import os
import tempfile

import pytest

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.docling_worker import extract_concepts
from mlsyseng_mcp.embeddings import _chunk_text
from mlsyseng_mcp.expert_registry import ExpertRegistry, _slugify, _concepts_to_capabilities
from mlsyseng_mcp.loop_controller import ConvergenceLoop, l2_norm, default_step_fn


# ── Database Tests ────────────────────────────────────────────────────


class TestDatabase:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_upsert_and_get_chapter(self):
        self.db.upsert_chapter(
            chapter_name="01_Intro",
            source_path="/tmp/ch1",
            markdown_content="# Introduction\nThis is ML.",
            concepts=["neural network", "deep learning"],
        )
        ch = self.db.get_chapter("01_Intro")
        assert ch is not None
        assert ch["chapter_name"] == "01_Intro"
        assert "neural network" in ch["concepts"]
        assert ch["status"] == "extracted"

    def test_list_chapters(self):
        self.db.upsert_chapter("A", "/a", "content a", ["pca"], status="extracted")
        self.db.upsert_chapter("B", "/b", "content b", ["svm"], status="pending")
        all_ch = self.db.list_chapters()
        assert len(all_ch) == 2
        extracted = self.db.list_chapters(status="extracted")
        assert len(extracted) == 1
        assert extracted[0]["chapter_name"] == "A"

    def test_upsert_and_get_expert(self):
        expert = {
            "expert_name": "ML Systems",
            "slug": "ml_systems",
            "chapter_id": None,
            "capabilities": ["Build models"],
            "skills": ["/skills/kaggle-model-trainer"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        self.db.upsert_expert(expert)
        result = self.db.get_expert("ml_systems")
        assert result is not None
        assert result["expert_name"] == "ML Systems"
        assert result["capabilities"] == ["Build models"]

    def test_list_experts(self):
        for name in ["Expert A", "Expert B"]:
            self.db.upsert_expert({
                "expert_name": name,
                "slug": _slugify(name),
                "capabilities": [],
                "skills": [],
                "strategy": "",
                "formula": {},
                "loop_config": {},
            })
        experts = self.db.list_experts()
        assert len(experts) == 2

    def test_extraction_status(self):
        self.db.upsert_chapter("Ch1", "/c1", "text", [], status="extracted")
        self.db.upsert_chapter("Ch2", "/c2", "", [], status="pending")
        self.db.upsert_chapter("Ch3", "/c3", "", [], status="failed")
        status = self.db.get_extraction_status()
        assert status["total"] == 3
        assert status["extracted"] == 1
        assert status["pending"] == 1
        assert status["failed"] == 1

    def test_convergence_state(self):
        self.db.save_state("titanic", 1, [0.5, 0.6], {"loss": 0.3})
        self.db.save_state("titanic", 2, [0.51, 0.61], {"loss": 0.28})
        states = self.db.get_states("titanic")
        assert len(states) == 2
        latest = self.db.get_latest_state("titanic")
        assert latest["iteration"] == 2

    def test_stats(self):
        self.db.upsert_chapter("Ch1", "/c1", "text", [], status="extracted")
        self.db.upsert_expert({
            "expert_name": "E1",
            "slug": "e1",
            "capabilities": [],
            "skills": [],
            "strategy": "",
            "formula": {},
            "loop_config": {},
        })
        self.db.save_state("titanic", 1, [0.5], {"loss": 0.3})
        stats = self.db.get_stats()
        assert stats["chapters"]["total"] == 1
        assert stats["experts"] == 1
        assert "titanic" in stats["competitions"]


# ── Docling Worker Tests ──────────────────────────────────────────────


class TestDoclingWorker:
    def test_extract_concepts(self):
        text = "Deep learning uses neural networks with gradient descent for optimization."
        concepts = extract_concepts(text)
        assert "deep learning" in concepts
        assert "neural network" in concepts
        assert "gradient descent" in concepts

    def test_extract_concepts_empty(self):
        assert extract_concepts("") == []
        assert extract_concepts("No ML concepts here, just regular text.") == []

    def test_chunk_text(self):
        words = " ".join(f"word{i}" for i in range(1000))
        chunks = _chunk_text(words, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_chunk_text_short(self):
        short = "This is a short text."
        chunks = _chunk_text(short, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == short


# ── Expert Registry Tests ────────────────────────────────────────────


class TestExpertRegistry:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)
        self.registry = ExpertRegistry(db=self.db, skills_path="/tmp/skills")

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_slugify(self):
        assert _slugify("08 ML Systems") == "08_ml_systems"
        assert _slugify("Deep Learning!") == "deep_learning"

    def test_concepts_to_capabilities(self):
        caps = _concepts_to_capabilities(["neural network", "feature engineering"])
        assert "Build baseline models quickly" in caps
        assert any("neural" in c.lower() for c in caps)
        assert any("feature" in c.lower() for c in caps)

    def test_create_expert_from_chapter(self):
        expert = self.registry.create_expert_from_chapter(
            chapter_name="08_ML_Systems",
            concepts=["neural network", "gradient descent", "cross-validation"],
        )
        assert expert["slug"] == "08_ml_systems"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert expert["formula"]["objective"] == "minimize_validation_loss"

    def test_create_experts_from_db(self):
        self.db.upsert_chapter("Ch1", "/c1", "neural network text", ["neural network"])
        self.db.upsert_chapter("Ch2", "/c2", "ensemble text", ["ensemble", "boosting"])
        experts = self.registry.create_experts_from_db()
        assert len(experts) == 2

    def test_list_and_get_expert(self):
        self.registry.create_expert_from_chapter("Test_Expert", ["PCA", "clustering"])
        experts = self.registry.list_experts()
        assert len(experts) == 1
        exp = self.registry.get_expert("test_expert")
        assert exp is not None
        assert exp["expert_name"] == "Test_Expert"

    def test_export_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.registry.create_expert_from_chapter("Export_Test", ["regression"])
            path = self.registry.export_expert("export_test", output_dir=tmpdir)
            assert path is not None
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["expert_name"] == "Export_Test"


# ── Loop Controller Tests ────────────────────────────────────────────


class TestLoopController:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_l2_norm(self):
        assert l2_norm([1, 0], [1, 0]) == 0.0
        assert abs(l2_norm([1, 0], [0, 1]) - math.sqrt(2)) < 1e-10
        assert l2_norm([3, 4], [0, 0]) == 5.0

    def test_l2_norm_different_lengths(self):
        result = l2_norm([1, 2], [1, 2, 3])
        assert result == 3.0

    def test_convergence_loop_converges(self):
        call_count = {"n": 0}

        def converging_step(iteration, prev_state):
            call_count["n"] += 1
            decay = 0.5 ** iteration
            state = [1.0 - decay, 0.5 - decay * 0.5]
            return state, {"metric": decay}

        loop = ConvergenceLoop(
            competition="test",
            epsilon=0.01,
            max_iterations=20,
            patience=2,
            db=self.db,
        )
        result = loop.run(converging_step)
        assert result["converged"] is True
        assert result["iterations"] < 20

    def test_convergence_loop_max_iterations(self):
        def diverging_step(iteration, prev_state):
            state = [float(iteration), float(iteration * 2)]
            return state, {"metric": iteration}

        loop = ConvergenceLoop(
            competition="test_diverge",
            epsilon=0.001,
            max_iterations=5,
            patience=3,
            db=self.db,
        )
        result = loop.run(diverging_step)
        assert result["converged"] is False
        assert result["iterations"] == 5

    def test_default_step_fn(self):
        experts = [
            {"slug": "expert_a", "relevance_score": 0.9},
            {"slug": "expert_b", "relevance_score": 0.7},
        ]
        step = default_step_fn(experts, "titanic")
        state, metrics = step(1, None)
        assert len(state) == 2
        assert "mean_score" in metrics
        assert "max_score" in metrics

    def test_state_persisted(self):
        def simple_step(iteration, prev_state):
            return [0.5, 0.5], {"loss": 0.1}

        loop = ConvergenceLoop(
            competition="persist_test",
            epsilon=0.0001,
            max_iterations=3,
            patience=3,
            db=self.db,
        )
        loop.run(simple_step)
        states = self.db.get_states("persist_test")
        assert len(states) == 3


# ── Integration-like Tests ───────────────────────────────────────────


class TestEndToEnd:
    """Integration tests that exercise the full pipeline without external deps."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=self.tmp.name)

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_chapter_to_expert_pipeline(self):
        self.db.upsert_chapter(
            chapter_name="01_Introduction",
            source_path="/fake/path",
            markdown_content="Neural networks use gradient descent for training.",
            concepts=["neural network", "gradient descent", "backpropagation"],
        )

        registry = ExpertRegistry(db=self.db, skills_path="/tmp/skills")
        experts = registry.create_experts_from_db()
        assert len(experts) == 1

        exp = experts[0]
        assert exp["slug"] == "01_introduction"
        assert "Build baseline models quickly" in exp["capabilities"]
        assert exp["formula"]["objective"] == "minimize_validation_loss"

    def test_convergence_with_experts(self):
        self.db.upsert_chapter("ChA", "/a", "text a", ["ensemble", "boosting"])
        self.db.upsert_chapter("ChB", "/b", "text b", ["regression"])

        registry = ExpertRegistry(db=self.db, skills_path="/tmp/skills")
        experts = registry.create_experts_from_db()

        step_fn = default_step_fn(experts, "house-prices")
        loop = ConvergenceLoop(
            competition="house-prices",
            epsilon=0.01,
            max_iterations=15,
            patience=2,
            db=self.db,
        )
        result = loop.run(step_fn)
        assert result["iterations"] > 0
        assert "final_delta" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
