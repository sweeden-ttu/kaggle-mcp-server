"""Tests for the database module."""

import json
import os
import tempfile

import pytest

import database as db


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db.init_db(path)
    yield path
    os.unlink(path)


def test_init_db(temp_db):
    conn = db.get_connection(temp_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    table_names = {t["name"] for t in tables}
    assert "chapters" in table_names
    assert "experts" in table_names
    assert "extraction_log" in table_names
    assert "competition_entries" in table_names


def test_upsert_chapter(temp_db):
    cid = db.upsert_chapter("Test Chapter", "/path/to/folder", "/path/to/pdf", temp_db)
    assert cid > 0

    cid2 = db.upsert_chapter("Test Chapter", "/path/to/folder2", None, temp_db)
    assert cid2 == cid


def test_update_chapter_extraction(temp_db):
    cid = db.upsert_chapter("Ch1", "/p", None, temp_db)
    db.update_chapter_extraction(cid, "Extracted text", ["ML", "DL"], "completed", temp_db)

    chapters = db.get_chapters_by_status("completed", temp_db)
    assert len(chapters) == 1
    assert chapters[0]["chapter_name"] == "Ch1"
    assert "ML" in chapters[0]["concepts"]


def test_mark_chapter_failed(temp_db):
    cid = db.upsert_chapter("FailCh", "/p", None, temp_db)
    db.mark_chapter_failed(cid, "test error", temp_db)

    failed = db.get_chapters_by_status("failed", temp_db)
    assert len(failed) == 1


def test_upsert_expert(temp_db):
    expert_def = {
        "expert_name": "Test Expert",
        "slug": "test_expert",
        "chapter_id": None,
        "capabilities": ["cap1", "cap2"],
        "skills": ["/skills/a", "/skills/b"],
        "strategy": "A → B → C",
        "formula": {"objective": "min_loss", "metrics": ["acc"]},
        "loop_config": {"epsilon": 0.001, "max_iterations": 10, "patience": 3},
    }
    eid = db.upsert_expert(expert_def, temp_db)
    assert eid > 0

    expert = db.get_expert_by_slug("test_expert", temp_db)
    assert expert is not None
    assert expert["expert_name"] == "Test Expert"
    assert expert["capabilities"] == ["cap1", "cap2"]
    assert expert["formula"]["objective"] == "min_loss"


def test_get_all_experts(temp_db):
    for i in range(3):
        db.upsert_expert({
            "expert_name": f"Expert {i}",
            "slug": f"expert_{i}",
            "capabilities": [],
            "skills": [],
            "strategy": "test",
            "formula": {},
            "loop_config": {},
        }, temp_db)

    experts = db.get_all_experts(temp_db)
    assert len(experts) == 3


def test_get_stats(temp_db):
    db.upsert_chapter("Ch1", "/p", None, temp_db)
    stats = db.get_stats(temp_db)
    assert stats["chapters_total"] == 1
    assert stats["experts_total"] == 0


def test_upsert_competition_entry(temp_db):
    eid = db.upsert_competition_entry(
        "titanic", "test_expert",
        notebook_path="/path/to/nb.ipynb",
        state_vector=[1.0, 0.5, 0.3],
        db_path=temp_db,
    )
    assert eid > 0
