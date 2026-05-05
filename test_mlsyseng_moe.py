"""Tests for the MLSysEng MoE system."""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from src.mlsyseng_mcp.database import ChapterRecord, ExpertRecord, MLSysEngDatabase
from src.mlsyseng_mcp.docling_worker import (
    _chapter_id_from_path,
    _chapter_title_from_folder,
    extract_concepts,
)
from src.mlsyseng_mcp.expert_registry import (
    _slugify,
    _capabilities_from_concepts,
    _formula_for_concepts,
    _skill_paths_for_concepts,
    build_competition_entry,
    get_expert_definition,
    register_experts_from_chapters,
)
from src.mlsyseng_mcp.loop_controller import (
    IterationState,
    LoopConfig,
    LoopController,
    l2_norm,
    run_evolve_loop,
)


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = MLSysEngDatabase(db_path=db_path)
    yield db
    db.close()
    os.unlink(db_path)


@pytest.fixture
def sample_chapter():
    return ChapterRecord(
        chapter_id="ch_08_ml_systems",
        title="ML Systems",
        source_pdf="/tmp/08_ml_systems.pdf",
        content_md="This chapter covers gradient descent, backpropagation, and neural network optimization.",
        concepts=["gradient descent", "backpropagation", "neural network"],
    )


@pytest.fixture
def sample_expert():
    return ExpertRecord(
        expert_name="ML Systems",
        slug="ml_systems",
        chapter_id="ch_08_ml_systems",
        capabilities=["Build baseline models quickly"],
        skills=["/tmp/skills/kaggle-model-trainer"],
        strategy="Baseline → EDA → Feature Engineering → Model Selection → Submit",
    )


class TestDatabase:
    def test_create_database(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0

    def test_upsert_and_get_chapter(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        result = tmp_db.get_chapter("ch_08_ml_systems")
        assert result is not None
        assert result.title == "ML Systems"
        assert result.word_count > 0
        assert "gradient descent" in result.concepts

    def test_list_chapters(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        chapters = tmp_db.list_chapters()
        assert len(chapters) == 1
        assert chapters[0].chapter_id == "ch_08_ml_systems"

    def test_upsert_replaces_chapter(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        sample_chapter.content_md = "Updated content about deep learning"
        tmp_db.upsert_chapter(sample_chapter)
        chapters = tmp_db.list_chapters()
        assert len(chapters) == 1
        assert "Updated" in chapters[0].content_md

    def test_upsert_and_get_expert(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        result = tmp_db.get_expert("ML Systems")
        assert result is not None
        assert result.slug == "ml_systems"

    def test_get_expert_by_slug(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        result = tmp_db.get_expert_by_slug("ml_systems")
        assert result is not None
        assert result.expert_name == "ML Systems"

    def test_list_experts(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        experts = tmp_db.list_experts()
        assert len(experts) == 1

    def test_convergence_state(self, tmp_db):
        state_id = tmp_db.save_convergence_state(
            competition="titanic",
            iteration=1,
            state_vector=[0.5, 0.3, 0.2],
            l2_norm=0.01,
            converged=False,
        )
        assert state_id > 0

        history = tmp_db.get_convergence_history("titanic")
        assert len(history) == 1
        assert history[0]["iteration"] == 1
        assert history[0]["l2_norm"] == 0.01

    def test_stats(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        stats = tmp_db.get_stats()
        assert stats["chapters_indexed"] == 1
        assert stats["experts_registered"] == 1
        assert stats["total_words_extracted"] > 0


class TestDoclingWorker:
    def test_chapter_id_from_path(self):
        assert _chapter_id_from_path("08_ML_Systems") == "ch_08_08_ml_systems"
        assert _chapter_id_from_path("01_Introduction") == "ch_01_01_introduction"

    def test_chapter_title_from_folder(self):
        assert _chapter_title_from_folder("08_ML_Systems") == "ML Systems"
        assert _chapter_title_from_folder("01 Introduction to ML") == "Introduction to ML"

    def test_extract_concepts(self):
        text = "This discusses gradient descent and backpropagation in neural networks."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "backpropagation" in concepts

    def test_extract_concepts_case_insensitive(self):
        text = "GRADIENT DESCENT is used for optimization with RELU activations."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "relu" in concepts

    def test_extract_concepts_empty(self):
        text = "This text has no ML concepts."
        concepts = extract_concepts(text)
        assert len(concepts) == 0


class TestExpertRegistry:
    def test_slugify(self):
        assert _slugify("ML Systems") == "ml_systems"
        assert _slugify("08 Deep Learning!") == "08_deep_learning"

    def test_capabilities_from_concepts(self):
        caps = _capabilities_from_concepts(["gradient descent", "ensemble"])
        assert len(caps) >= 2
        assert "Build baseline models quickly" in caps

    def test_formula_for_concepts(self):
        formula = _formula_for_concepts(["f1 score", "precision", "recall"])
        assert "f1_score" in formula["metrics"]

    def test_formula_default(self):
        formula = _formula_for_concepts([])
        assert formula["objective"] == "minimize_validation_loss"
        assert "accuracy" in formula["metrics"]

    def test_skill_paths_for_concepts(self):
        skills = _skill_paths_for_concepts(
            ["gradient descent", "ensemble"], "/tmp/skills"
        )
        assert any("kaggle-model-trainer" in s for s in skills)
        assert any("kaggle-ensembler" in s for s in skills)
        assert any("kaggle-preprocessor" in s for s in skills)
        assert any("kaggle-submitter" in s for s in skills)

    def test_register_experts_from_chapters(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        result = register_experts_from_chapters(tmp_db, skills_base_path="/tmp/skills")
        assert result["experts_created"] == 1
        experts = tmp_db.list_experts()
        assert len(experts) == 1
        assert experts[0].chapter_id == "ch_08_ml_systems"

    def test_get_expert_definition(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        defn = get_expert_definition(tmp_db, "ML Systems")
        assert defn is not None
        assert defn["slug"] == "ml_systems"

    def test_get_expert_definition_by_slug(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        defn = get_expert_definition(tmp_db, "ml_systems")
        assert defn is not None

    def test_build_competition_entry(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        entry = build_competition_entry(tmp_db, "titanic")
        assert entry["competition"] == "titanic"
        assert len(entry["experts_consulted"]) >= 1
        assert len(entry["pipeline_steps"]) == 7


class TestLoopController:
    def test_l2_norm_zero(self):
        assert l2_norm([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_l2_norm_basic(self):
        result = l2_norm([0.0, 0.0], [3.0, 4.0])
        assert abs(result - 5.0) < 1e-10

    def test_l2_norm_length_mismatch(self):
        with pytest.raises(ValueError):
            l2_norm([1.0], [1.0, 2.0])

    def test_convergence_loop_converges(self, tmp_db):
        config = LoopConfig(epsilon=0.01, max_iterations=50, patience=2)
        controller = LoopController(config=config, db=tmp_db)

        call_count = [0]

        def step_fn(iteration, prev):
            call_count[0] += 1
            if prev is None:
                return [1.0, 0.5, 0.25]
            decay = 0.5 ** iteration
            return [v + decay * 0.001 for v in prev]

        result = controller.run("test-conv", step_fn, initial_state=[1.0, 0.5, 0.25])
        assert result["converged"] is True
        assert result["competition"] == "test-conv"

    def test_convergence_loop_max_iterations(self, tmp_db):
        config = LoopConfig(epsilon=1e-20, max_iterations=3, patience=2)
        controller = LoopController(config=config, db=tmp_db)

        def step_fn(iteration, prev):
            if prev is None:
                return [0.0]
            return [prev[0] + 1.0]

        result = controller.run("test-max", step_fn, initial_state=[0.0])
        assert result["converged"] is False
        assert result["total_iterations"] == 3

    def test_run_evolve_loop(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        experts = tmp_db.list_experts()

        result = run_evolve_loop(
            competition="titanic",
            db=tmp_db,
            experts=experts,
            epsilon=0.1,
            max_iterations=20,
            patience=2,
        )
        assert "converged" in result
        assert "total_iterations" in result
        assert result["competition"] == "titanic"

    def test_convergence_persisted(self, tmp_db):
        config = LoopConfig(epsilon=0.01, max_iterations=5, patience=1)
        controller = LoopController(config=config, db=tmp_db)

        def step_fn(iteration, prev):
            if prev is None:
                return [1.0]
            return [prev[0] + 0.001 * (0.5 ** iteration)]

        controller.run("persist-test", step_fn, initial_state=[1.0])
        history = tmp_db.get_convergence_history("persist-test")
        assert len(history) > 0


class TestChapterRecord:
    def test_auto_word_count(self):
        ch = ChapterRecord(
            chapter_id="test",
            title="Test",
            source_pdf="/tmp/test.pdf",
            content_md="one two three four five",
        )
        assert ch.word_count == 5

    def test_auto_timestamp(self):
        ch = ChapterRecord(
            chapter_id="test",
            title="Test",
            source_pdf="/tmp/test.pdf",
            content_md="content",
        )
        assert ch.extracted_at != ""


class TestExpertRecord:
    def test_default_formula(self):
        expert = ExpertRecord(
            expert_name="Test",
            slug="test",
            chapter_id="ch_test",
        )
        assert expert.formula["objective"] == "minimize_validation_loss"

    def test_default_loop_config(self):
        expert = ExpertRecord(
            expert_name="Test",
            slug="test",
            chapter_id="ch_test",
        )
        assert expert.loop_config["epsilon"] == 0.001
        assert expert.loop_config["max_iterations"] == 10
        assert expert.loop_config["patience"] == 3
