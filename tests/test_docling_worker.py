"""Tests for mlsyseng_moe.docling_worker module."""

import os
import tempfile

import pytest

from src.mlsyseng_moe.database import Database
from src.mlsyseng_moe.docling_worker import (
    discover_chapter_folders,
    extract_concepts,
    process_chapter,
    run_extraction,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    d = Database(db_path=db_path)
    d.initialize()
    yield d
    d.close()


@pytest.fixture
def chapter_tree(tmp_path):
    """Create a fake chapter directory structure."""
    base = tmp_path / "chapters"
    base.mkdir()

    ch01 = base / "01_Introduction_to_ML"
    ch01.mkdir()
    (ch01 / "notes.md").write_text(
        "# Introduction to ML\n\n"
        "This chapter covers neural network basics and gradient descent optimization.\n"
        "Deep learning is a subset of machine learning using multiple layers."
    )

    ch02 = base / "02_Feature_Engineering"
    ch02.mkdir()
    (ch02 / "content.txt").write_text(
        "Feature engineering involves creating new features from data.\n"
        "Feature selection helps reduce dimensionality.\n"
        "Normalization and standardization are preprocessing steps.\n"
        "Cross-validation ensures model generalization."
    )

    ch03 = base / "03_Empty_Chapter"
    ch03.mkdir()

    return str(base)


class TestDiscoverChapterFolders:
    def test_discovers_chapters(self, chapter_tree):
        folders = discover_chapter_folders(chapter_tree)
        assert len(folders) == 3
        assert folders[0]["chapter_id"] == "01"
        assert folders[0]["title"] == "Introduction to ML"

    def test_nonexistent_path(self, tmp_path):
        folders = discover_chapter_folders(str(tmp_path / "nonexistent"))
        assert folders == []

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        folders = discover_chapter_folders(str(empty))
        assert folders == []


class TestExtractConcepts:
    def test_finds_concepts(self):
        text = "This text discusses neural network architectures and gradient descent optimization."
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "gradient descent" in concepts

    def test_empty_text(self):
        assert extract_concepts("") == []

    def test_no_concepts(self):
        concepts = extract_concepts("The weather is nice today.")
        assert concepts == []

    def test_deduplication(self):
        text = "neural network and neural network again"
        concepts = extract_concepts(text)
        assert concepts.count("neural network") == 1


class TestProcessChapter:
    def test_process_with_md_content(self, db, chapter_tree):
        chapter_info = {
            "chapter_id": "01",
            "title": "Introduction to ML",
            "folder_path": os.path.join(chapter_tree, "01_Introduction_to_ML"),
            "pdf_path": None,
        }
        result = process_chapter(chapter_info, db)
        assert result["status"] == "done"
        assert result["concepts_count"] > 0

        ch = db.get_chapter("01")
        assert ch is not None
        assert "neural network" in ch["concepts"]

    def test_skip_already_extracted(self, db, chapter_tree):
        chapter_info = {
            "chapter_id": "01",
            "title": "Introduction to ML",
            "folder_path": os.path.join(chapter_tree, "01_Introduction_to_ML"),
            "pdf_path": None,
        }
        process_chapter(chapter_info, db)
        result = process_chapter(chapter_info, db, force=False)
        assert result["status"] == "skipped"

    def test_force_reextract(self, db, chapter_tree):
        chapter_info = {
            "chapter_id": "01",
            "title": "Introduction to ML",
            "folder_path": os.path.join(chapter_tree, "01_Introduction_to_ML"),
            "pdf_path": None,
        }
        process_chapter(chapter_info, db)
        result = process_chapter(chapter_info, db, force=True)
        assert result["status"] == "done"

    def test_empty_chapter(self, db, chapter_tree):
        chapter_info = {
            "chapter_id": "03",
            "title": "Empty Chapter",
            "folder_path": os.path.join(chapter_tree, "03_Empty_Chapter"),
            "pdf_path": None,
        }
        result = process_chapter(chapter_info, db)
        assert result["status"] == "empty"


class TestRunExtraction:
    def test_full_extraction(self, db, chapter_tree):
        results = run_extraction(db, base_path=chapter_tree)
        assert len(results) == 3
        done = [r for r in results if r["status"] == "done"]
        assert len(done) == 2

    def test_no_chapters(self, db, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = run_extraction(db, base_path=str(empty))
        assert results[0]["status"] == "no_chapters"
