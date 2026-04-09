"""Tests for MLSysEng MoE database module."""

import json
import os
import tempfile

import pytest

from src.mlsyseng_mcp.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_mlsyseng.db")
    return Database(db_path)


class TestChapterOperations:
    def test_upsert_and_get_chapter(self, db):
        chapter_id = db.upsert_chapter(
            chapter_name="01_Introduction",
            slug="01_introduction",
            source_path="/path/to/pdf",
            markdown_content="# Introduction\nThis is the intro.",
            page_count=10,
        )
        assert chapter_id > 0

        chapter = db.get_chapter("01_Introduction")
        assert chapter is not None
        assert chapter["chapter_name"] == "01_Introduction"
        assert chapter["slug"] == "01_introduction"
        assert chapter["page_count"] == 10
        assert "Introduction" in chapter["markdown_content"]

    def test_get_chapter_by_slug(self, db):
        db.upsert_chapter("02_Supervised", "02_supervised", "/path", "content", 5)
        chapter = db.get_chapter_by_slug("02_supervised")
        assert chapter is not None
        assert chapter["chapter_name"] == "02_Supervised"

    def test_list_chapters(self, db):
        db.upsert_chapter("Ch1", "ch1", "/p1", "c1", 5)
        db.upsert_chapter("Ch2", "ch2", "/p2", "c2", 10)
        chapters = db.list_chapters()
        assert len(chapters) == 2
        assert chapters[0]["chapter_name"] == "Ch1"

    def test_upsert_updates_existing(self, db):
        db.upsert_chapter("Ch1", "ch1", "/p1", "old content", 5)
        db.upsert_chapter("Ch1", "ch1", "/p1", "new content", 15)
        chapter = db.get_chapter("Ch1")
        assert chapter["markdown_content"] == "new content"
        assert chapter["page_count"] == 15


class TestConceptOperations:
    def test_add_and_get_concepts(self, db):
        ch_id = db.upsert_chapter("Ch1", "ch1", "/p", "content", 1)
        db.add_concepts(ch_id, [
            {"concept_name": "Gradient Descent", "description": "Optimization", "category": "algorithm"},
            {"concept_name": "Neural Network", "description": "Architecture", "category": "architecture"},
        ])
        concepts = db.get_concepts_for_chapter(ch_id)
        assert len(concepts) == 2
        names = {c["concept_name"] for c in concepts}
        assert "Gradient Descent" in names
        assert "Neural Network" in names

    def test_search_concepts(self, db):
        ch_id = db.upsert_chapter("Ch1", "ch1", "/p", "content", 1)
        db.add_concepts(ch_id, [
            {"concept_name": "Gradient Descent", "description": "Optimization method"},
            {"concept_name": "Dropout", "description": "Regularization technique"},
        ])
        results = db.search_concepts("gradient")
        assert len(results) >= 1
        assert results[0]["concept_name"] == "Gradient Descent"

    def test_search_concepts_by_description(self, db):
        ch_id = db.upsert_chapter("Ch1", "ch1", "/p", "content", 1)
        db.add_concepts(ch_id, [
            {"concept_name": "LR", "description": "Learning rate scheduling"},
        ])
        results = db.search_concepts("scheduling")
        assert len(results) >= 1


class TestExpertOperations:
    def test_upsert_and_get_expert(self, db):
        expert = {
            "expert_name": "ML_Systems",
            "slug": "ml_systems",
            "capabilities": ["Build baselines", "Hyperparameter search"],
            "skills": ["/path/to/skill1"],
            "strategy": "Baseline -> Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001, "max_iterations": 10},
        }
        expert_id = db.upsert_expert(expert)
        assert expert_id > 0

        retrieved = db.get_expert("ML_Systems")
        assert retrieved is not None
        assert retrieved["slug"] == "ml_systems"
        assert isinstance(retrieved["capabilities"], list)
        assert len(retrieved["capabilities"]) == 2
        assert isinstance(retrieved["formula"], dict)

    def test_get_expert_by_slug(self, db):
        db.upsert_expert({
            "expert_name": "DL_Expert",
            "slug": "dl_expert",
            "capabilities": [],
            "skills": [],
            "strategy": "",
            "formula": {},
            "loop_config": {},
        })
        expert = db.get_expert_by_slug("dl_expert")
        assert expert is not None
        assert expert["expert_name"] == "DL_Expert"

    def test_list_experts(self, db):
        for i in range(3):
            db.upsert_expert({
                "expert_name": f"Expert_{i}",
                "slug": f"expert_{i}",
                "capabilities": [f"cap_{i}"],
                "skills": [],
                "strategy": "test",
                "formula": {},
                "loop_config": {},
            })
        experts = db.list_experts()
        assert len(experts) == 3


class TestExtractionStatus:
    def test_set_and_get_status(self, db):
        db.set_extraction_status("Ch1", "extracting")
        status = db.get_extraction_status()
        assert len(status) == 1
        assert status[0]["chapter_name"] == "Ch1"
        assert status[0]["status"] == "extracting"

    def test_update_status(self, db):
        db.set_extraction_status("Ch1", "extracting")
        db.set_extraction_status("Ch1", "completed", pages_extracted=10)
        status = db.get_extraction_status()
        assert len(status) == 1
        assert status[0]["status"] == "completed"
        assert status[0]["pages_extracted"] == 10


class TestConvergenceLog:
    def test_log_and_retrieve(self, db):
        db.log_convergence("titanic", 1, [0.5, 0.3, 0.1], 0.5, False)
        db.log_convergence("titanic", 2, [0.4, 0.3, 0.1], 0.1, False)
        db.log_convergence("titanic", 3, [0.39, 0.3, 0.1], 0.01, True)

        history = db.get_convergence_history("titanic")
        assert len(history) == 3
        assert history[0]["iteration"] == 1
        assert history[2]["converged"] == 1
        assert isinstance(history[0]["state_vector"], list)


class TestStats:
    def test_get_stats(self, db):
        db.upsert_chapter("Ch1", "ch1", "/p", "c", 1)
        ch_id = db.upsert_chapter("Ch2", "ch2", "/p2", "c2", 2)
        db.add_concepts(ch_id, [
            {"concept_name": "test_concept"},
        ])
        db.upsert_expert({
            "expert_name": "E1", "slug": "e1",
            "capabilities": [], "skills": [], "strategy": "",
            "formula": {}, "loop_config": {},
        })

        stats = db.get_stats()
        assert stats["total_chapters"] == 2
        assert stats["total_concepts"] == 1
        assert stats["total_experts"] == 1
