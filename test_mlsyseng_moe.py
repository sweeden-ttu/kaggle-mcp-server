"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile
import time

import pytest

from src.mlsyseng_moe.database import Chapter, Expert, CompetitionEntry, MoEDatabase
from src.mlsyseng_moe.docling_worker import extract_concepts, infer_skills, _slug_from_title, _chapter_id_from_folder
from src.mlsyseng_moe.expert_registry import ExpertRegistry
from src.mlsyseng_moe.loop_controller import LoopController, StateVector, l2_norm


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = MoEDatabase(db_path)
    yield db
    db.close()
    os.unlink(db_path)


# ─── Database Tests ──────────────────────────────────────────────────


class TestDatabase:
    def test_chapter_crud(self, tmp_db):
        ch = Chapter(
            chapter_id="ch01",
            title="Introduction to ML",
            folder_path="/tmp/ch01",
            content_md="Some content about ML",
            concepts='["gradient descent", "optimization"]',
            extracted_at=time.time(),
            status="done",
        )
        tmp_db.upsert_chapter(ch)

        loaded = tmp_db.get_chapter("ch01")
        assert loaded is not None
        assert loaded.title == "Introduction to ML"
        assert loaded.status == "done"
        assert "gradient descent" in loaded.concept_list

    def test_chapter_list(self, tmp_db):
        for i in range(3):
            ch = Chapter(
                chapter_id=f"ch{i:02d}",
                title=f"Chapter {i}",
                folder_path=f"/tmp/ch{i:02d}",
                status="done" if i < 2 else "pending",
            )
            tmp_db.upsert_chapter(ch)

        all_ch = tmp_db.list_chapters()
        assert len(all_ch) == 3

        done = tmp_db.list_chapters(status="done")
        assert len(done) == 2

    def test_expert_crud(self, tmp_db):
        ch = Chapter(chapter_id="ch01", title="ML", folder_path="/tmp/ch01", status="done")
        tmp_db.upsert_chapter(ch)

        expert = Expert(
            expert_name="ML Systems",
            slug="ml_systems",
            chapter_id="ch01",
            capabilities='["Build baseline models"]',
            skills='["/skills/kaggle-trainer"]',
            strategy="Baseline → Submit",
            formula='{"objective": "minimize_loss"}',
            loop_config='{"epsilon": 0.001}',
            created_at=time.time(),
        )
        tmp_db.upsert_expert(expert)

        loaded = tmp_db.get_expert("ML Systems")
        assert loaded is not None
        assert loaded.slug == "ml_systems"

        by_slug = tmp_db.get_expert_by_slug("ml_systems")
        assert by_slug is not None
        assert by_slug.expert_name == "ML Systems"

    def test_expert_to_dict(self, tmp_db):
        expert = Expert(
            expert_name="Deep Learning",
            slug="deep_learning",
            chapter_id="ch02",
            capabilities='["Neural networks", "Transformers"]',
            skills='["/skills/kaggle-deep"]',
            formula='{"objective": "minimize_validation_loss", "metrics": ["accuracy"]}',
            loop_config='{"epsilon": 0.001}',
        )
        d = expert.to_dict()
        assert isinstance(d["capabilities"], list)
        assert isinstance(d["formula"], dict)
        assert d["formula"]["objective"] == "minimize_validation_loss"

    def test_competition_entry(self, tmp_db):
        entry = CompetitionEntry(
            entry_id="titanic_abc123",
            competition="titanic",
            experts_used='["ML Systems", "Deep Learning"]',
            skills_used='["/skills/kaggle-trainer"]',
            state_history='[{"metrics": {"loss": 0.5}}]',
            converged=True,
            final_metric=0.123,
            created_at=time.time(),
        )
        tmp_db.save_entry(entry)

        loaded = tmp_db.get_entry("titanic_abc123")
        assert loaded is not None
        assert loaded.converged is True
        assert loaded.competition == "titanic"

    def test_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert "chapters_total" in stats
        assert "experts_registered" in stats
        assert stats["chapters_total"] == 0


# ─── Docling Worker Tests ────────────────────────────────────────────


class TestDoclingWorker:
    def test_extract_concepts(self):
        text = "This chapter covers gradient descent and neural network optimization with regularization."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "neural network" in concepts
        assert "regularization" in concepts

    def test_extract_concepts_empty(self):
        assert extract_concepts("") == []
        assert extract_concepts("Hello world, nothing ML here") == []

    def test_infer_skills(self):
        concepts = ["gradient descent", "neural network", "cross-validation"]
        skills = infer_skills(concepts)
        assert any("trainer" in s for s in skills)
        assert any("submitter" in s for s in skills)

    def test_slug_from_title(self):
        assert _slug_from_title("ML Systems") == "ml_systems"
        assert _slug_from_title("08 Deep Learning") == "08_deep_learning"

    def test_chapter_id_from_folder(self):
        assert _chapter_id_from_folder("08_ML_Systems") == "08_ml_systems"


# ─── Expert Registry Tests ───────────────────────────────────────────


class TestExpertRegistry:
    def test_register_and_list(self, tmp_db):
        ch = Chapter(chapter_id="ch01", title="ML", folder_path="/tmp", status="done")
        tmp_db.upsert_chapter(ch)

        registry = ExpertRegistry(tmp_db)
        registry.register_expert(
            expert_name="Test Expert",
            slug="test_expert",
            chapter_id="ch01",
            capabilities=["Fast prototyping"],
            skills=["/skills/kaggle-trainer"],
        )

        all_experts = registry.list_all()
        assert len(all_experts) == 1
        assert all_experts[0]["expert_name"] == "Test Expert"

    def test_ask_expert(self, tmp_db):
        ch = Chapter(
            chapter_id="ch01",
            title="ML",
            folder_path="/tmp",
            content_md="Content about ML",
            status="done",
        )
        tmp_db.upsert_chapter(ch)

        registry = ExpertRegistry(tmp_db)
        registry.register_expert(
            expert_name="ML Expert",
            slug="ml_expert",
            chapter_id="ch01",
            capabilities=["Optimization"],
            skills=["/skills/kaggle-opt"],
        )

        result = registry.ask_expert("ml_expert", "How to tune learning rate?")
        assert "error" not in result
        assert result["question"] == "How to tune learning rate?"
        assert "guidance" in result

    def test_ask_unknown_expert(self, tmp_db):
        registry = ExpertRegistry(tmp_db)
        result = registry.ask_expert("nonexistent", "question")
        assert "error" in result


# ─── Loop Controller Tests ───────────────────────────────────────────


class TestLoopController:
    def test_l2_norm(self):
        assert l2_norm([0, 0], [0, 0]) == 0.0
        assert abs(l2_norm([1, 0], [0, 0]) - 1.0) < 1e-9
        assert abs(l2_norm([3, 4], [0, 0]) - 5.0) < 1e-9

    def test_convergence_detection(self):
        controller = LoopController(epsilon=0.01, patience=2)

        s1 = StateVector(metrics={"loss": 1.0, "accuracy": 0.5})
        r1 = controller.step(s1)
        assert r1["iteration"] == 1
        assert not r1["should_stop"]

        s2 = StateVector(metrics={"loss": 0.5, "accuracy": 0.7})
        r2 = controller.step(s2)
        assert r2["iteration"] == 2
        assert not r2["converged"]

        s3 = StateVector(metrics={"loss": 0.5001, "accuracy": 0.7001})
        r3 = controller.step(s3)
        assert r3["converged"]
        assert not r3["should_stop"]

        s4 = StateVector(metrics={"loss": 0.5002, "accuracy": 0.7001})
        r4 = controller.step(s4)
        assert r4["converged"]
        assert r4["should_stop"]

    def test_max_iterations(self):
        controller = LoopController(epsilon=0.0001, max_iterations=3)

        for i in range(3):
            s = StateVector(metrics={"loss": 1.0 / (i + 1)})
            result = controller.step(s)

        assert result["should_stop"]
        assert "max_iterations" in result["reason"]

    def test_state_vector_dict(self):
        sv = StateVector(metrics={"loss": 0.5, "acc": 0.9})
        d = sv.to_dict()
        restored = StateVector.from_dict(d)
        assert restored.metrics == sv.metrics

    def test_reset(self):
        controller = LoopController()
        controller.step(StateVector(metrics={"loss": 1.0}))
        assert controller.iteration == 1
        controller.reset()
        assert controller.iteration == 0
        assert len(controller.history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
