"""Tests for mlsyseng_mcp.docling_worker module."""

import os
import tempfile

import pytest

from mlsyseng_mcp import database as db
from mlsyseng_mcp import docling_worker


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    yield


class TestExtractConcepts:
    def test_finds_gradient_descent(self):
        text = "We use gradient descent to optimize the loss function."
        concepts = docling_worker._extract_concepts_from_text(text)
        names = [c["concept"].lower() for c in concepts]
        assert any("gradient descent" in n for n in names)

    def test_finds_multiple(self):
        text = (
            "Backpropagation computes gradients. "
            "We apply dropout for regularization. "
            "The transformer uses self-attention."
        )
        concepts = docling_worker._extract_concepts_from_text(text)
        assert len(concepts) >= 3

    def test_empty_text(self):
        assert docling_worker._extract_concepts_from_text("") == []

    def test_no_ml_content(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert docling_worker._extract_concepts_from_text(text) == []


class TestCategorizeConcept:
    def test_optimization(self):
        assert docling_worker._categorize_concept("gradient descent") == "optimization"

    def test_architecture(self):
        assert docling_worker._categorize_concept("neural network") == "architecture"

    def test_unknown(self):
        assert docling_worker._categorize_concept("foobar") == "general"


class TestDiscoverChapters:
    def test_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_PRINCIPLES_PATH", str(tmp_path))
        chapters = docling_worker.discover_chapters()
        assert chapters == []

    def test_finds_pdfs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_PRINCIPLES_PATH", str(tmp_path))
        ch1 = tmp_path / "01_intro"
        ch1.mkdir()
        (ch1 / "chapter1.pdf").write_text("fake pdf")
        chapters = docling_worker.discover_chapters()
        assert len(chapters) == 1
        assert chapters[0][0] == "01_intro"

    def test_skips_non_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_PRINCIPLES_PATH", str(tmp_path))
        (tmp_path / "readme.txt").write_text("not a chapter")
        chapters = docling_worker.discover_chapters()
        assert chapters == []

    def test_nonexistent_path(self, monkeypatch):
        monkeypatch.setenv("ML_PRINCIPLES_PATH", "/nonexistent/path")
        chapters = docling_worker.discover_chapters()
        assert chapters == []


class TestExtractChapter:
    def test_extract_skips_existing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
        db.init_db()
        db.upsert_chapter("01", "Ch1", "/a.pdf", "content", 1)
        result = docling_worker.extract_chapter("01", "/a.pdf", force=False)
        assert result["status"] == "skipped"

    def test_extract_force_reindex(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
        db.init_db()
        db.upsert_chapter("01", "Ch1", "/a.pdf", "content", 1)
        result = docling_worker.extract_chapter("01", "/nonexistent.pdf", force=True)
        assert result["status"] in ("completed", "failed")
