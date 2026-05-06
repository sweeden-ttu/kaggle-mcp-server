"""Tests for MLSysEng MoE database module."""

import json
import os
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlsyseng_mcp.database import (
    ChapterRecord,
    ConvergenceState,
    ExpertRecord,
    MLSysEngDatabase,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_mlsyseng.db")
    return MLSysEngDatabase(db_path=db_path)


class TestChapterOperations:
    def test_upsert_and_get_chapter(self, db):
        ch = ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Introduction",
            chapter_number=1,
            source_path="/fake/path.pdf",
            content_md="# Chapter 1\n\nIntroduction to ML.",
            concepts=["neural network", "gradient descent"],
        )
        db.upsert_chapter(ch)

        result = db.get_chapter("ch001")
        assert result is not None
        assert result.chapter_name == "01_Introduction"
        assert result.chapter_number == 1
        assert "neural network" in result.concepts
        assert result.word_count > 0

    def test_list_chapters(self, db):
        for i in range(3):
            ch = ChapterRecord(
                chapter_id=f"ch{i:03d}",
                chapter_name=f"{i:02d}_Chapter",
                chapter_number=i,
                source_path=f"/fake/{i}.pdf",
                content_md=f"Content for chapter {i}",
            )
            db.upsert_chapter(ch)

        chapters = db.list_chapters()
        assert len(chapters) == 3
        assert chapters[0].chapter_number == 0

    def test_upsert_overwrites(self, db):
        ch = ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Introduction",
            chapter_number=1,
            source_path="/fake/path.pdf",
            content_md="Original content",
        )
        db.upsert_chapter(ch)

        ch.content_md = "Updated content"
        db.upsert_chapter(ch)

        result = db.get_chapter("ch001")
        assert result.content_md == "Updated content"

    def test_get_nonexistent_chapter(self, db):
        assert db.get_chapter("nonexistent") is None


class TestExpertOperations:
    def test_upsert_and_get_expert(self, db):
        ch = ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Introduction",
            chapter_number=1,
            source_path="/fake/path.pdf",
            content_md="Content",
        )
        db.upsert_chapter(ch)

        expert = ExpertRecord(
            expert_id="exp001",
            expert_name="01_Introduction",
            slug="01_introduction",
            chapter_id="ch001",
            capabilities=["Build baseline models"],
            skills=["/path/to/skill"],
        )
        db.upsert_expert(expert)

        result = db.get_expert("exp001")
        assert result is not None
        assert result.slug == "01_introduction"
        assert "Build baseline models" in result.capabilities

    def test_get_expert_by_slug(self, db):
        ch = ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Introduction",
            chapter_number=1,
            source_path="/fake/path.pdf",
            content_md="Content",
        )
        db.upsert_chapter(ch)

        expert = ExpertRecord(
            expert_id="exp001",
            expert_name="01_Introduction",
            slug="01_introduction",
            chapter_id="ch001",
        )
        db.upsert_expert(expert)

        result = db.get_expert_by_slug("01_introduction")
        assert result is not None
        assert result.expert_id == "exp001"

    def test_list_experts(self, db):
        ch = ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Chapter",
            chapter_number=1,
            source_path="/fake/path.pdf",
            content_md="Content",
        )
        db.upsert_chapter(ch)

        for i in range(3):
            expert = ExpertRecord(
                expert_id=f"exp{i:03d}",
                expert_name=f"{i:02d}_Expert",
                slug=f"{i:02d}_expert",
                chapter_id="ch001",
            )
            db.upsert_expert(expert)

        experts = db.list_experts()
        assert len(experts) == 3


class TestConvergenceState:
    def test_save_and_get_history(self, db):
        for i in range(5):
            state = ConvergenceState(
                iteration=i,
                state_vector=[0.5 + i * 0.1, 0.3 + i * 0.05],
                delta_norm=0.1 / (i + 1),
                converged=i >= 3,
            )
            db.save_convergence_state("titanic", state)

        history = db.get_convergence_history("titanic")
        assert len(history) == 5
        assert history[0].iteration == 0
        assert history[4].converged is True


class TestExtractionStatus:
    def test_set_and_get_status(self, db):
        db.set_status("extraction_status", "running")
        assert db.get_status("extraction_status") == "running"

        db.set_status("extraction_status", "completed")
        assert db.get_status("extraction_status") == "completed"

    def test_get_nonexistent_status(self, db):
        assert db.get_status("nonexistent") is None


class TestStats:
    def test_get_stats(self, db):
        stats = db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0
        assert stats["total_words_extracted"] == 0

        ch = ChapterRecord(
            chapter_id="ch001",
            chapter_name="01_Introduction",
            chapter_number=1,
            source_path="/fake/path.pdf",
            content_md="hello world foo bar",
        )
        db.upsert_chapter(ch)

        stats = db.get_stats()
        assert stats["chapters_indexed"] == 1
        assert stats["total_words_extracted"] == 4
