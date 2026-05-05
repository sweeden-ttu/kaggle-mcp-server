"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp.database import MoEDatabase
from mlsyseng_mcp.docling_worker import chapter_id_from_folder, extract_concepts
from mlsyseng_mcp.expert_registry import (
    ExpertRegistry,
    _infer_capabilities,
    _infer_skills,
    _slugify,
)
from mlsyseng_mcp.loop_controller import (
    ConvergenceLoop,
    l2_distance,
    l2_norm,
    run_evolve,
)

from pathlib import Path


# ── Database Tests ──────────────────────────────────────────────────


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = MoEDatabase(path)
    yield database
    database.close()
    os.unlink(path)


def test_chapter_crud(db):
    db.upsert_chapter("ch1", "Chapter_01", "Introduction", content_md="# Intro")
    chapter = db.get_chapter("ch1")
    assert chapter is not None
    assert chapter["title"] == "Introduction"
    assert chapter["content_md"] == "# Intro"

    chapters = db.list_chapters()
    assert len(chapters) == 1


def test_expert_crud(db):
    db.upsert_expert(
        {
            "expert_name": "Test Expert",
            "slug": "test_expert",
            "chapter_id": "ch1",
            "capabilities": ["build models"],
            "skills": ["/skills/test"],
            "strategy": "Baseline",
            "formula": {"objective": "min"},
            "loop_config": {"epsilon": 0.001},
        }
    )
    expert = db.get_expert("Test Expert")
    assert expert is not None
    assert expert["capabilities"] == ["build models"]

    expert_by_slug = db.get_expert_by_slug("test_expert")
    assert expert_by_slug is not None

    experts = db.list_experts()
    assert len(experts) == 1


def test_concepts(db):
    db.upsert_chapter("ch1", "Chapter_01", "Intro")
    db.add_concept("ch1", "gradient descent", "optimisation method")
    db.add_concept("ch1", "neural network", "computation graph")

    results = db.search_concepts("gradient")
    assert len(results) >= 1
    assert results[0]["term"] == "gradient descent"

    chapter_concepts = db.get_concepts_for_chapter("ch1")
    assert len(chapter_concepts) == 2


def test_convergence_log(db):
    db.log_convergence("titanic", 1, [0.1, 0.2], 0.05, False)
    db.log_convergence("titanic", 2, [0.15, 0.25], 0.02, True)
    history = db.get_convergence_history("titanic")
    assert len(history) == 2
    assert history[1]["converged"] is True


def test_stats(db):
    db.upsert_chapter("ch1", "Chapter_01", "Intro")
    db.upsert_expert(
        {
            "expert_name": "E1",
            "slug": "e1",
            "capabilities": [],
            "skills": [],
        }
    )
    db.add_concept("ch1", "term1")
    stats = db.get_stats()
    assert stats["chapters"] == 1
    assert stats["experts"] == 1
    assert stats["concepts"] == 1


def test_extraction_status(db):
    db.upsert_chapter("ch1", "Chapter_01", "Intro")
    db.set_extraction_status("ch1", "running", 0.5)
    statuses = db.get_extraction_status()
    assert len(statuses) == 1
    assert statuses[0]["status"] == "running"


# ── Loop Controller Tests ───────────────────────────────────────────


def test_l2_norm():
    assert abs(l2_norm([3, 4]) - 5.0) < 1e-10
    assert abs(l2_norm([0]) - 0.0) < 1e-10
    assert abs(l2_norm([1, 1, 1]) - 3**0.5) < 1e-10


def test_l2_distance():
    assert abs(l2_distance([1, 0], [0, 0]) - 1.0) < 1e-10
    assert abs(l2_distance([3, 4], [0, 0]) - 5.0) < 1e-10

    with pytest.raises(ValueError):
        l2_distance([1, 2], [1])


def test_convergence_loop():
    def step(state, iteration):
        return [s + 0.1 / iteration for s in state]

    loop = ConvergenceLoop(epsilon=0.01, max_iterations=30, patience=2)
    result = loop.run([0.0, 0.0], step, competition="test")
    assert result["converged"] is True
    assert result["iterations"] <= 30
    assert len(result["history"]) == result["iterations"]


def test_convergence_loop_max_iterations():
    def step(state, _):
        return [s + 1.0 for s in state]

    loop = ConvergenceLoop(epsilon=0.0001, max_iterations=3, patience=2)
    result = loop.run([0.0], step, competition="no_converge")
    assert result["converged"] is False
    assert result["iterations"] == 3


def test_run_evolve():
    experts = [{"expert_name": "E1", "slug": "e1"}, {"expert_name": "E2", "slug": "e2"}]
    result = run_evolve("test", experts, epsilon=0.01, max_iterations=30, patience=2)
    assert "converged" in result
    assert "final_state" in result
    assert len(result["final_state"]) == 2


# ── Expert Registry Tests ──────────────────────────────────────────


def test_slugify():
    assert _slugify("08_ML Systems") == "08_ml_systems"
    assert _slugify("Introduction to ML") == "introduction_to_ml"
    assert _slugify("Deep Learning & CNNs") == "deep_learning_cnns"


def test_infer_skills():
    skills = _infer_skills(["neural network", "feature engineering"])
    skill_names = [os.path.basename(s) for s in skills]
    assert "kaggle-deep-learning" in skill_names
    assert "kaggle-feature-engineer" in skill_names


def test_infer_capabilities():
    caps = _infer_capabilities(["neural network", "regularization"], "Deep Learning")
    assert any("neural" in c.lower() for c in caps)
    assert any("regularization" in c.lower() for c in caps)


def test_expert_registry_flow(db):
    registry = ExpertRegistry(db)
    db.upsert_chapter("ch1", "Chapter_01", "01_Intro", content_md="# Intro")
    db.add_concept("ch1", "neural network")
    db.add_concept("ch1", "gradient descent")

    expert = registry.create_expert_from_chapter(
        "ch1", "01_Intro", ["neural network", "gradient descent"]
    )
    assert expert["expert_name"] == "01_Intro"
    assert expert["slug"] == "01_intro"

    experts = registry.list_experts()
    assert len(experts) == 1

    answer = registry.ask_expert("01_intro", "How to train?")
    assert "recommendation" in answer

    entry = registry.build_competition_entry("titanic", "classify passengers")
    assert entry["competition"] == "titanic"
    assert "execution_plan" in entry


# ── Docling Worker Tests ────────────────────────────────────────────


def test_chapter_id_from_folder():
    assert chapter_id_from_folder(Path("/some/08_ML Systems")) == "08_ml_systems"
    assert (
        chapter_id_from_folder(Path("/path/01_Introduction to ML"))
        == "01_introduction_to_ml"
    )


def test_extract_concepts():
    text = (
        "This chapter covers gradient descent optimization.\n"
        "Regularization prevents overfitting.\n"
        "Cross-validation is used for evaluation.\n"
    )
    concepts = extract_concepts(text)
    terms = [c["term"].lower() for c in concepts]
    assert any("gradient" in t for t in terms)
    assert any("regularization" in t for t in terms)
    assert any("cross-validation" in t for t in terms)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
