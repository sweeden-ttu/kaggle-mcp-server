"""Tests for MLSysEng MoE expert registry module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlsyseng_mcp.database import ChapterRecord, MLSysEngDatabase
from mlsyseng_mcp.expert_registry import ExpertRegistry


@pytest.fixture
def db(tmp_path):
    return MLSysEngDatabase(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def populated_db(db):
    chapters = [
        ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Introduction",
            chapter_number=1,
            source_path="/fake/01.pdf",
            content_md="Intro to ML",
            concepts=["neural network", "gradient descent"],
        ),
        ChapterRecord(
            chapter_id="ch002",
            chapter_name="08_ML_Systems",
            chapter_number=8,
            source_path="/fake/08.pdf",
            content_md="ML Systems chapter",
            concepts=["pipeline", "hyperparameter", "cross-validation"],
        ),
        ChapterRecord(
            chapter_id="ch003",
            chapter_name="10_Deep_Learning",
            chapter_number=10,
            source_path="/fake/10.pdf",
            content_md="Deep learning chapter",
            concepts=["deep learning", "transformer", "transfer learning"],
        ),
    ]
    for ch in chapters:
        db.upsert_chapter(ch)
    return db


class TestExpertRegistration:
    def test_register_from_chapter(self, populated_db):
        registry = ExpertRegistry(populated_db)
        chapter = populated_db.get_chapter("ch001")
        expert = registry.register_from_chapter(chapter)

        assert expert.expert_name == "01_Introduction"
        assert expert.slug == "01_introduction"
        assert len(expert.capabilities) > 0
        assert len(expert.skills) > 0
        assert expert.strategy
        assert expert.formula["objective"] == "minimize_validation_loss"

    def test_register_all_chapters(self, populated_db):
        registry = ExpertRegistry(populated_db)
        experts = registry.register_all_chapters()

        assert len(experts) == 3

    def test_list_experts(self, populated_db):
        registry = ExpertRegistry(populated_db)
        registry.register_all_chapters()

        listed = registry.list_experts()
        assert len(listed) == 3

    def test_get_expert_by_slug(self, populated_db):
        registry = ExpertRegistry(populated_db)
        registry.register_all_chapters()

        expert = registry.get_expert("08_ml_systems")
        assert expert is not None
        assert expert.expert_name == "08_ML_Systems"

    def test_get_nonexistent_expert(self, populated_db):
        registry = ExpertRegistry(populated_db)
        assert registry.get_expert("nonexistent") is None


class TestExpertCapabilities:
    def test_deep_learning_expert_has_dl_capabilities(self, populated_db):
        registry = ExpertRegistry(populated_db)
        chapter = populated_db.get_chapter("ch003")
        expert = registry.register_from_chapter(chapter)

        assert any("deep learning" in c.lower() for c in expert.capabilities)

    def test_ml_systems_expert_strategy(self, populated_db):
        registry = ExpertRegistry(populated_db)
        chapter = populated_db.get_chapter("ch002")
        expert = registry.register_from_chapter(chapter)

        assert "Baseline" in expert.strategy


class TestExpertExport:
    def test_export_expert_json(self, populated_db):
        registry = ExpertRegistry(populated_db)
        registry.register_all_chapters()

        exported = registry.export_expert_json("01_introduction")
        assert exported is not None
        assert exported["slug"] == "01_introduction"

    def test_export_all_experts_json(self, populated_db):
        registry = ExpertRegistry(populated_db)
        registry.register_all_chapters()

        all_exported = registry.export_all_experts_json()
        assert len(all_exported) == 3

    def test_export_nonexistent_expert(self, populated_db):
        registry = ExpertRegistry(populated_db)
        assert registry.export_expert_json("fake") is None


class TestExpertForCompetition:
    def test_get_experts_without_embeddings(self, populated_db):
        registry = ExpertRegistry(populated_db)
        registry.register_all_chapters()

        matched = registry.get_experts_for_competition("titanic")
        assert len(matched) == 3
        assert all(m["relevance_score"] == 1.0 for m in matched)
