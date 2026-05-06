"""Tests for mlsyseng_mcp.database module."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp.database import MLSysEngDatabase


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_mlsyseng.db")
    return MLSysEngDatabase(db_path=db_path)


class TestChapterOperations:
    def test_upsert_and_get_chapter(self, db):
        db.upsert_chapter(
            chapter_id="ch_01",
            title="Introduction to ML",
            folder_path="/path/to/ch01",
            pdf_path="/path/to/ch01/intro.pdf",
            content_md="# Introduction\nThis is chapter 1.",
            concepts=["neural network", "deep learning"],
        )
        chapter = db.get_chapter("ch_01")
        assert chapter is not None
        assert chapter["title"] == "Introduction to ML"
        assert chapter["concepts"] == ["neural network", "deep learning"]
        assert "extracted_at" in chapter

    def test_list_chapters(self, db):
        db.upsert_chapter("ch_01", "Chapter 1")
        db.upsert_chapter("ch_02", "Chapter 2")
        chapters = db.list_chapters()
        assert len(chapters) == 2

    def test_chapter_exists(self, db):
        assert not db.chapter_exists("ch_01")
        db.upsert_chapter("ch_01", "Chapter 1")
        assert db.chapter_exists("ch_01")

    def test_upsert_overwrites(self, db):
        db.upsert_chapter("ch_01", "Old Title")
        db.upsert_chapter("ch_01", "New Title")
        chapter = db.get_chapter("ch_01")
        assert chapter["title"] == "New Title"

    def test_get_nonexistent_chapter(self, db):
        assert db.get_chapter("nonexistent") is None


class TestExpertOperations:
    def test_upsert_and_get_expert(self, db):
        db.upsert_expert(
            expert_name="ML Systems Expert",
            slug="08_ml_systems",
            chapter_id="ch_08",
            capabilities=["Build baseline models", "Hyperparameter search"],
            skills=["/path/to/skill1", "/path/to/skill2"],
            strategy="Baseline → Feature Eng → Model → Submit",
            formula={"objective": "minimize_validation_loss"},
            loop_config={"epsilon": 0.001, "max_iterations": 10},
        )
        expert = db.get_expert("ML Systems Expert")
        assert expert is not None
        assert expert["slug"] == "08_ml_systems"
        assert len(expert["capabilities"]) == 2
        assert expert["formula"]["objective"] == "minimize_validation_loss"

    def test_get_expert_by_slug(self, db):
        db.upsert_expert("Test Expert", "test_expert", chapter_id="ch_01")
        expert = db.get_expert_by_slug("test_expert")
        assert expert is not None
        assert expert["expert_name"] == "Test Expert"

    def test_list_experts(self, db):
        db.upsert_expert("Expert A", "a")
        db.upsert_expert("Expert B", "b")
        experts = db.list_experts()
        assert len(experts) == 2


class TestExtractionLog:
    def test_log_and_retrieve(self, db):
        db.log_extraction_event("ch_01", "start", "beginning extraction")
        db.log_extraction_event("ch_01", "complete", "done")
        db.log_extraction_event("ch_02", "start", "beginning extraction")

        all_logs = db.get_extraction_log()
        assert len(all_logs) == 3

        ch01_logs = db.get_extraction_log("ch_01")
        assert len(ch01_logs) == 2


class TestStateSnapshots:
    def test_save_and_get_snapshots(self, db):
        db.save_state_snapshot("titanic", 1, [0.1, 0.2, 0.3], {"accuracy": 0.75})
        db.save_state_snapshot("titanic", 2, [0.5, 0.6, 0.7], {"accuracy": 0.82})

        snapshots = db.get_state_snapshots("titanic")
        assert len(snapshots) == 2
        assert snapshots[0]["iteration"] == 1
        assert snapshots[1]["state_vector"] == [0.5, 0.6, 0.7]

    def test_empty_snapshots(self, db):
        snapshots = db.get_state_snapshots("nonexistent")
        assert len(snapshots) == 0


class TestStats:
    def test_get_stats(self, db):
        db.upsert_chapter("ch_01", "Chapter 1")
        db.upsert_expert("Expert 1", "e1")
        stats = db.get_stats()
        assert stats["chapters"] == 1
        assert stats["experts"] == 1
        assert "db_path" in stats
