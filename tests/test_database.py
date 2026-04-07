"""Tests for mlsyseng_moe.database module."""

import json
import os
import tempfile

import pytest

from src.mlsyseng_moe.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    d = Database(db_path=db_path)
    d.initialize()
    yield d
    d.close()


class TestDatabaseInit:
    def test_creates_tables(self, db):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "chapters" in names
        assert "experts" in names
        assert "extraction_status" in names
        assert "loop_states" in names

    def test_initialize_is_idempotent(self, db):
        db.initialize()
        db.initialize()
        stats = db.get_stats()
        assert stats["chapters"] == 0


class TestChapterOperations:
    def test_upsert_and_get(self, db):
        db.upsert_chapter(
            chapter_id="01",
            title="Introduction",
            folder_path="/chapters/01",
            content_md="# Introduction\nHello world",
            concepts=["neural network", "deep learning"],
        )
        ch = db.get_chapter("01")
        assert ch is not None
        assert ch["title"] == "Introduction"
        assert ch["concepts"] == ["neural network", "deep learning"]
        assert ch["content_md"] == "# Introduction\nHello world"

    def test_upsert_updates_existing(self, db):
        db.upsert_chapter("01", "Old Title", "/old", "", [])
        db.upsert_chapter("01", "New Title", "/new", "content", ["concept"])
        ch = db.get_chapter("01")
        assert ch["title"] == "New Title"
        assert ch["folder_path"] == "/new"

    def test_get_nonexistent(self, db):
        assert db.get_chapter("nonexistent") is None

    def test_list_chapters(self, db):
        db.upsert_chapter("02", "Chapter 2", "/ch2", "content2", ["a"])
        db.upsert_chapter("01", "Chapter 1", "/ch1", "content1", ["b"])
        chapters = db.list_chapters()
        assert len(chapters) == 2
        assert chapters[0]["chapter_id"] == "01"


class TestExpertOperations:
    def test_upsert_and_get(self, db):
        defn = {"slug": "intro", "capabilities": ["test"]}
        db.upsert_expert("01_intro", defn)
        row = db.get_expert("01_intro")
        assert row is not None
        assert row["definition"]["slug"] == "intro"

    def test_list_experts(self, db):
        db.upsert_expert("a_expert", {"slug": "a"})
        db.upsert_expert("b_expert", {"slug": "b"})
        experts = db.list_experts()
        assert len(experts) == 2


class TestExtractionStatus:
    def test_set_and_get(self, db):
        db.set_extraction_status("01", "extracting", "Processing")
        statuses = db.get_extraction_status()
        assert len(statuses) == 1
        assert statuses[0]["status"] == "extracting"

    def test_upsert_status(self, db):
        db.set_extraction_status("01", "extracting")
        db.set_extraction_status("01", "done", "Complete")
        statuses = db.get_extraction_status()
        assert len(statuses) == 1
        assert statuses[0]["status"] == "done"


class TestLoopState:
    def test_save_and_get(self, db):
        db.save_loop_state("titanic", 1, [0.1, 0.2, 0.3], {"diff": 0.5})
        db.save_loop_state("titanic", 2, [0.2, 0.3, 0.4], {"diff": 0.1})
        states = db.get_loop_states("titanic")
        assert len(states) == 2
        assert states[0]["iteration"] == 1
        assert states[1]["state_vector"] == [0.2, 0.3, 0.4]

    def test_get_empty(self, db):
        states = db.get_loop_states("nonexistent")
        assert states == []


class TestStats:
    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats == {"chapters": 0, "experts": 0, "loop_states": 0}

    def test_stats_after_inserts(self, db):
        db.upsert_chapter("01", "Ch1", "/ch1", "content", [])
        db.upsert_expert("expert1", {"slug": "e1"})
        db.save_loop_state("comp", 1, [1.0], {})
        stats = db.get_stats()
        assert stats["chapters"] == 1
        assert stats["experts"] == 1
        assert stats["loop_states"] == 1
