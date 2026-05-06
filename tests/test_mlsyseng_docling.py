"""Tests for MLSysEng MoE docling worker module."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlsyseng_mcp.docling_worker import (
    DoclingWorker,
    extract_concepts,
    scan_chapter_folders,
)
from mlsyseng_mcp.database import MLSysEngDatabase


class TestConceptExtraction:
    def test_extract_basic_concepts(self):
        text = (
            "Neural network architectures use gradient descent for optimization. "
            "Regularization helps prevent overfitting."
        )
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "gradient descent" in concepts
        assert "regularization" in concepts

    def test_extract_no_concepts(self):
        text = "This is a completely unrelated document about cooking."
        concepts = extract_concepts(text)
        assert len(concepts) == 0

    def test_extract_deduplicates(self):
        text = "Neural network neural network neural network"
        concepts = extract_concepts(text)
        assert concepts.count("neural network") == 1


class TestScanChapterFolders:
    def test_scan_empty_dir(self, tmp_path):
        chapters = scan_chapter_folders(str(tmp_path))
        assert len(chapters) == 0

    def test_scan_with_pdfs(self, tmp_path):
        ch1 = tmp_path / "01_Introduction"
        ch1.mkdir()
        (ch1 / "chapter1.pdf").write_bytes(b"fake pdf content")

        ch2 = tmp_path / "02_Deep_Learning"
        ch2.mkdir()
        (ch2 / "chapter2.pdf").write_bytes(b"fake pdf content")

        chapters = scan_chapter_folders(str(tmp_path))
        assert len(chapters) == 2
        assert chapters[0]["chapter_number"] == 1
        assert chapters[1]["chapter_number"] == 2

    def test_scan_nonexistent_dir(self):
        chapters = scan_chapter_folders("/nonexistent/path")
        assert len(chapters) == 0

    def test_scan_top_level_pdfs(self, tmp_path):
        (tmp_path / "chapter1.pdf").write_bytes(b"fake pdf")
        (tmp_path / "chapter2.pdf").write_bytes(b"fake pdf")
        (tmp_path / "readme.txt").write_text("not a pdf")

        chapters = scan_chapter_folders(str(tmp_path))
        assert len(chapters) == 2


class TestDoclingWorker:
    def test_worker_no_chapters(self, tmp_path):
        db = MLSysEngDatabase(db_path=str(tmp_path / "test.db"))
        worker = DoclingWorker(db, ml_principles_path=str(tmp_path / "empty"))

        result = worker.run()
        assert result["status"] == "no_chapters"
        assert result["chapters_processed"] == 0

    def test_worker_progress_tracking(self, tmp_path):
        db = MLSysEngDatabase(db_path=str(tmp_path / "test.db"))
        worker = DoclingWorker(db, ml_principles_path=str(tmp_path / "empty"))

        assert worker.progress == {}
