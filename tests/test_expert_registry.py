"""Tests for mlsyseng_moe.expert_registry module."""

import json

import pytest

from mlsyseng_moe.database import Chapter, Database
from mlsyseng_moe.expert_registry import (
    _infer_formula_type,
    _infer_strategy_type,
    _slugify,
    create_expert_from_chapter,
    get_experts_for_competition,
    register_experts_from_db,
)


@pytest.fixture
def db(tmp_path):
    return Database(db_path=str(tmp_path / "test.db"))


class TestSlugify:
    def test_basic(self):
        assert _slugify("ML Systems") == "ml_systems"

    def test_chapter_prefix(self):
        assert _slugify("Chapter 8: ML Systems") == "chapter_8_ml_systems"

    def test_special_chars(self):
        assert _slugify("Deep Learning!") == "deep_learning"


class TestInferStrategyType:
    def test_deep_learning(self):
        assert _infer_strategy_type(["neural network", "deep learning"]) == "deep_learning"

    def test_tabular(self):
        assert _infer_strategy_type(["decision tree", "random forest", "boosting"]) == "tabular"

    def test_nlp(self):
        assert _infer_strategy_type(["transformer", "attention mechanism"]) == "nlp"

    def test_default(self):
        assert _infer_strategy_type(["unknown concept"]) == "default"

    def test_empty(self):
        assert _infer_strategy_type([]) == "default"


class TestInferFormulaType:
    def test_optimization(self):
        assert _infer_formula_type(["gradient descent", "optimization"]) == "optimization"

    def test_classification(self):
        assert _infer_formula_type(["decision tree", "random forest"]) == "classification"

    def test_default(self):
        assert _infer_formula_type([]) == "default"


class TestCreateExpertFromChapter:
    def test_creates_expert(self, db):
        ch_id = db.upsert_chapter(Chapter(
            folder_name="08_ML_Systems",
            title="Chapter 8: ML Systems",
            status="extracted",
            concepts=json.dumps(["neural network", "gradient descent"]),
        ))

        expert = create_expert_from_chapter(
            chapter_id=ch_id,
            title="Chapter 8: ML Systems",
            concepts=["neural network", "gradient descent"],
            db=db,
        )

        assert expert.id is not None
        assert expert.expert_name == "Chapter 8: ML Systems"
        assert expert.slug == "chapter_8_ml_systems"
        assert expert.chapter_id == ch_id
        assert len(expert.capabilities_list) > 0
        assert len(expert.skills_list) > 0
        assert expert.strategy != ""

    def test_different_strategies_for_different_concepts(self, db):
        ch1_id = db.upsert_chapter(Chapter(
            folder_name="01_DL", title="Deep Learning", status="extracted",
        ))
        e1 = create_expert_from_chapter(ch1_id, "Deep Learning", ["neural network", "deep learning"], db)

        ch2_id = db.upsert_chapter(Chapter(
            folder_name="02_Tab", title="Tabular", status="extracted",
        ))
        e2 = create_expert_from_chapter(ch2_id, "Tabular", ["decision tree", "feature engineering"], db)

        assert e1.strategy != e2.strategy


class TestRegisterExpertsFromDb:
    def test_registers_all_extracted(self, db):
        db.upsert_chapter(Chapter(
            folder_name="01", title="Ch1", status="extracted",
            concepts=json.dumps(["neural network"]),
        ))
        db.upsert_chapter(Chapter(
            folder_name="02", title="Ch2", status="extracted",
            concepts=json.dumps(["decision tree"]),
        ))
        db.upsert_chapter(Chapter(
            folder_name="03", title="Ch3", status="pending",
        ))

        experts = register_experts_from_db(db)
        assert len(experts) == 2


class TestGetExpertsForCompetition:
    def test_keyword_matching(self, db):
        db.upsert_chapter(Chapter(folder_name="01", title="Ch1", status="extracted"))
        from mlsyseng_moe.expert_registry import Expert
        db.upsert_expert(Expert(
            expert_name="Neural Networks",
            slug="neural_networks",
            capabilities=json.dumps(["Expert in: neural network, deep learning"]),
        ))
        db.upsert_expert(Expert(
            expert_name="Decision Trees",
            slug="decision_trees",
            capabilities=json.dumps(["Expert in: decision tree, random forest"]),
        ))

        results = get_experts_for_competition("neural network classification", db)
        assert len(results) > 0

    def test_empty_when_no_match(self, db):
        results = get_experts_for_competition("xyz_no_match", db)
        assert len(results) == 0
