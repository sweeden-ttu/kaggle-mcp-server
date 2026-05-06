"""Tests for mlsyseng_mcp.database module."""

import json
import os
import sqlite3
import tempfile

import pytest

os.environ["SQLITE_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(), "test_mlsyseng.db"
)

from mlsyseng_mcp import database as db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Use a fresh database for each test."""
    test_db = str(tmp_path / "test.db")
    os.environ["SQLITE_DB_PATH"] = test_db
    db.init_db()
    yield test_db


class TestInitDb:
    def test_creates_tables(self, fresh_db):
        conn = sqlite3.connect(fresh_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "chapters" in tables
        assert "concepts" in tables
        assert "experts" in tables
        assert "extraction_log" in tables
        assert "competition_entries" in tables

    def test_idempotent(self, fresh_db):
        db.init_db()
        db.init_db()


class TestChapters:
    def test_upsert_and_get(self):
        ch_id = db.upsert_chapter(
            folder_name="01_intro",
            title="Introduction to ML",
            pdf_path="/path/to/intro.pdf",
            markdown_content="# Introduction\nML basics.",
            word_count=3,
        )
        assert ch_id > 0
        chapters = db.get_all_chapters()
        assert len(chapters) == 1
        assert chapters[0]["folder_name"] == "01_intro"
        assert chapters[0]["word_count"] == 3

    def test_upsert_updates_existing(self):
        db.upsert_chapter("01", "Title1", "/a.pdf", "content1", 1)
        db.upsert_chapter("01", "Title2", "/b.pdf", "content2", 2)
        chapters = db.get_all_chapters()
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Title2"

    def test_get_chapter_by_id(self):
        ch_id = db.upsert_chapter("02", "Chapter 2", "/c.pdf", "text", 5)
        ch = db.get_chapter(ch_id)
        assert ch is not None
        assert ch["title"] == "Chapter 2"

    def test_get_chapter_missing(self):
        assert db.get_chapter(999) is None


class TestConcepts:
    def test_add_and_get(self):
        ch_id = db.upsert_chapter("01", "Ch1", "/a.pdf", "text", 1)
        concepts = [
            {"concept": "gradient descent", "description": "Optimization", "category": "optimization"},
            {"concept": "backprop", "description": "Training", "category": "training"},
        ]
        db.add_concepts(ch_id, concepts)
        result = db.get_concepts_for_chapter(ch_id)
        assert len(result) == 2
        names = {r["concept"] for r in result}
        assert "gradient descent" in names

    def test_duplicate_concepts_ignored(self):
        ch_id = db.upsert_chapter("01", "Ch1", "/a.pdf", "text", 1)
        db.add_concepts(ch_id, [{"concept": "PCA"}])
        db.add_concepts(ch_id, [{"concept": "PCA"}])
        result = db.get_concepts_for_chapter(ch_id)
        assert len(result) == 1


class TestExperts:
    def _make_expert(self, slug="test_expert"):
        return db.upsert_expert(
            slug=slug,
            expert_name="Test Expert",
            chapter_id=None,
            capabilities=["cap1", "cap2"],
            skills=["/skills/a", "/skills/b"],
            strategy="Baseline → Submit",
            formula={"objective": "min_loss", "metrics": ["acc"]},
            loop_config={"epsilon": 0.01, "max_iterations": 5},
        )

    def test_upsert_and_get(self):
        eid = self._make_expert()
        assert eid > 0
        expert = db.get_expert("test_expert")
        assert expert is not None
        assert expert["expert_name"] == "Test Expert"
        assert expert["capabilities"] == ["cap1", "cap2"]
        assert expert["formula"]["objective"] == "min_loss"

    def test_get_all(self):
        self._make_expert("a")
        self._make_expert("b")
        experts = db.get_all_experts()
        assert len(experts) == 2

    def test_get_missing(self):
        assert db.get_expert("nonexistent") is None


class TestStats:
    def test_empty_stats(self):
        stats = db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0

    def test_stats_after_inserts(self):
        ch_id = db.upsert_chapter("01", "Ch1", "/a.pdf", "text words here", 3)
        db.add_concepts(ch_id, [{"concept": "PCA"}])
        db.upsert_expert("e1", "Expert1", ch_id, [], [], "", {}, {})
        stats = db.get_stats()
        assert stats["chapters_indexed"] == 1
        assert stats["experts_registered"] == 1
        assert stats["concepts_extracted"] == 1
        assert stats["total_words_extracted"] == 3


class TestExtractionLog:
    def test_log_extraction(self):
        ch_id = db.upsert_chapter("01", "Ch1", "/a.pdf", "", 0)
        db.log_extraction(ch_id, "started")
        status = db.get_extraction_status()
        assert len(status) == 1
        assert status[0]["status"] == "started"

    def test_complete_extraction(self):
        ch_id = db.upsert_chapter("01", "Ch1", "/a.pdf", "", 0)
        db.log_extraction(ch_id, "started")
        db.log_extraction(ch_id, "completed")
        status = db.get_extraction_status()
        assert status[0]["status"] == "completed"


class TestCompetitionEntries:
    def test_save_entry(self):
        db.save_entry("titanic", "exp1", "/nb.ipynb", 0.85, 1, [0.1, 0.2])
        stats = db.get_stats()
        assert stats["competition_entries"] == 1
