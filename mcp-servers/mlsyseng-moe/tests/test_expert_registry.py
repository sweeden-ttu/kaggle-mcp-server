"""Tests for the expert registry module."""

import os
import tempfile

import pytest

import database as db
import expert_registry as er


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db.init_db(path)
    yield path
    os.unlink(path)


def test_slugify():
    assert er._slugify("08_ML Systems") == "08_ml_systems"
    assert er._slugify("Deep Learning & CNNs") == "deep_learning_cnns"
    assert er._slugify("  spaces  ") == "spaces"


def test_match_chapter_type():
    assert er._match_chapter_type("08_ML Systems") == "system"
    assert er._match_chapter_type("Neural Networks") == "neural"
    assert er._match_chapter_type("Deep Learning") == "deep"
    assert er._match_chapter_type("Ensemble Methods") == "ensemble"
    assert er._match_chapter_type("Unknown Topic 99") == "system"


def test_create_expert_from_chapter(temp_db):
    cid = db.upsert_chapter("Neural Networks", "/path", None, temp_db)
    db.update_chapter_extraction(cid, "text", ["Neural Network", "Backpropagation"], "completed", temp_db)

    expert = er.create_expert_from_chapter(
        "Neural Networks", cid, ["Neural Network", "Backpropagation"], temp_db
    )
    assert expert["slug"] == "neural_networks"
    assert "Neural architecture design and optimization" in expert["capabilities"]
    assert expert["formula"]["objective"] == "minimize_validation_loss"


def test_register_all_experts(temp_db):
    for name in ["Neural Networks", "Ensemble Methods", "Feature Engineering"]:
        cid = db.upsert_chapter(name, f"/path/{name}", None, temp_db)
        db.update_chapter_extraction(cid, f"Content of {name}", ["ML"], "completed", temp_db)

    experts = er.register_all_experts(temp_db)
    assert len(experts) == 3


def test_build_competition_entry(temp_db):
    cid = db.upsert_chapter("Test Ch", "/p", None, temp_db)
    db.update_chapter_extraction(cid, "text", ["ML"], "completed", temp_db)
    er.create_expert_from_chapter("Test Ch", cid, ["ML"], temp_db)

    entry = er.build_competition_entry("titanic", db_path=temp_db)
    assert entry["competition"] == "titanic"
    assert len(entry["experts_used"]) > 0
    assert "combined_strategy" in entry


def test_combine_strategies():
    experts = [
        {"strategy": "A → B → C"},
        {"strategy": "B → D → E"},
    ]
    combined = er._combine_strategies(experts)
    assert "A" in combined
    assert "D" in combined
    assert "E" in combined


def test_combine_skills():
    experts = [
        {"skills": ["/a", "/b"]},
        {"skills": ["/b", "/c"]},
    ]
    combined = er._combine_skills(experts)
    assert combined == ["/a", "/b", "/c"]
