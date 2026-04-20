"""Tests for mlsyseng_moe.database module."""

import json
import os
import tempfile

import pytest

from mlsyseng_moe.database import Chapter, ConceptEntry, Database, Expert


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Database(db_path=db_path)


class TestDatabase:
    def test_init_creates_tables(self, db):
        stats = db.get_stats()
        assert stats["chapters_total"] == 0
        assert stats["experts_total"] == 0
        assert stats["concepts_total"] == 0

    def test_upsert_chapter_insert(self, db):
        ch = Chapter(
            folder_name="01_Introduction",
            title="Chapter 1: Introduction",
            pdf_path="/path/to/ch1.pdf",
            status="pending",
        )
        ch_id = db.upsert_chapter(ch)
        assert ch_id > 0

        retrieved = db.get_chapter(ch_id)
        assert retrieved is not None
        assert retrieved.folder_name == "01_Introduction"
        assert retrieved.title == "Chapter 1: Introduction"

    def test_upsert_chapter_update(self, db):
        ch = Chapter(
            folder_name="01_Introduction",
            title="Chapter 1: Introduction",
            status="pending",
        )
        id1 = db.upsert_chapter(ch)

        ch.content_md = "Updated content"
        ch.status = "extracted"
        id2 = db.upsert_chapter(ch)

        assert id1 == id2
        retrieved = db.get_chapter(id1)
        assert retrieved.content_md == "Updated content"
        assert retrieved.status == "extracted"

    def test_get_chapter_by_folder(self, db):
        ch = Chapter(folder_name="02_LinearAlgebra", title="Chapter 2")
        db.upsert_chapter(ch)

        found = db.get_chapter_by_folder("02_LinearAlgebra")
        assert found is not None
        assert found.title == "Chapter 2"

        assert db.get_chapter_by_folder("nonexistent") is None

    def test_list_chapters(self, db):
        db.upsert_chapter(Chapter(folder_name="01_Intro", title="Ch1", status="extracted"))
        db.upsert_chapter(Chapter(folder_name="02_LA", title="Ch2", status="pending"))
        db.upsert_chapter(Chapter(folder_name="03_Prob", title="Ch3", status="extracted"))

        all_chapters = db.list_chapters()
        assert len(all_chapters) == 3

        extracted = db.list_chapters(status="extracted")
        assert len(extracted) == 2

        pending = db.list_chapters(status="pending")
        assert len(pending) == 1

    def test_upsert_expert(self, db):
        ch_id = db.upsert_chapter(Chapter(folder_name="08_ML", title="ML Systems"))
        expert = Expert(
            expert_name="ML Systems",
            slug="ml_systems",
            chapter_id=ch_id,
            capabilities=json.dumps(["Build baseline models"]),
            skills=json.dumps(["kaggle-preprocessor"]),
            strategy="Baseline → Submit",
            formula=json.dumps({"objective": "minimize_loss"}),
            loop_config=json.dumps({"epsilon": 0.001}),
        )
        exp_id = db.upsert_expert(expert)
        assert exp_id > 0

        retrieved = db.get_expert("ml_systems")
        assert retrieved is not None
        assert retrieved.expert_name == "ML Systems"
        assert "Build baseline models" in retrieved.capabilities_list

    def test_list_experts(self, db):
        db.upsert_expert(Expert(expert_name="Expert A", slug="expert_a"))
        db.upsert_expert(Expert(expert_name="Expert B", slug="expert_b"))

        experts = db.list_experts()
        assert len(experts) == 2

    def test_add_and_get_concepts(self, db):
        ch_id = db.upsert_chapter(Chapter(folder_name="01", title="Ch1"))
        db.add_concept(ConceptEntry(chapter_id=ch_id, concept="neural network"))
        db.add_concept(ConceptEntry(chapter_id=ch_id, concept="gradient descent"))

        concepts = db.get_concepts(chapter_id=ch_id)
        assert len(concepts) == 2

        all_concepts = db.get_concepts()
        assert len(all_concepts) == 2

    def test_log_event(self, db):
        ch_id = db.upsert_chapter(Chapter(folder_name="01", title="Ch1"))
        db.log_event(ch_id, "test_event", "test details")

    def test_get_stats(self, db):
        db.upsert_chapter(Chapter(folder_name="01", title="Ch1", status="extracted"))
        db.upsert_chapter(Chapter(folder_name="02", title="Ch2", status="pending"))
        db.upsert_expert(Expert(expert_name="E1", slug="e1"))
        ch_id = db.upsert_chapter(Chapter(folder_name="03", title="Ch3", status="extracted"))
        db.add_concept(ConceptEntry(chapter_id=ch_id, concept="test"))

        stats = db.get_stats()
        assert stats["chapters_total"] == 3
        assert stats["chapters_extracted"] == 2
        assert stats["chapters_pending"] == 1
        assert stats["experts_total"] == 1
        assert stats["concepts_total"] == 1

    def test_chapter_concept_list(self):
        ch = Chapter(concepts=json.dumps(["a", "b", "c"]))
        assert ch.concept_list == ["a", "b", "c"]

    def test_expert_to_dict(self):
        expert = Expert(
            expert_name="Test",
            slug="test",
            capabilities=json.dumps(["cap1"]),
            skills=json.dumps(["skill1"]),
            strategy="Baseline",
            formula=json.dumps({"objective": "min_loss"}),
            loop_config=json.dumps({"epsilon": 0.01}),
        )
        d = expert.to_dict()
        assert d["expert_name"] == "Test"
        assert d["capabilities"] == ["cap1"]
        assert d["skills"] == ["skill1"]
        assert d["formula"]["objective"] == "min_loss"
