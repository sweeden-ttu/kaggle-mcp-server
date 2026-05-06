"""Tests for mlsyseng_mcp.docling_worker module."""

import os
import tempfile

import pytest

from mlsyseng_mcp.docling_worker import (
    extract_concepts,
    scan_chapters,
    _parse_chapter_number,
)


class TestExtractConcepts:
    def test_basic_extraction(self):
        text = "This chapter covers neural network architectures and gradient descent optimization."
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "gradient descent" in concepts

    def test_case_insensitive(self):
        text = "DEEP LEARNING is a subset of machine learning."
        concepts = extract_concepts(text)
        assert "deep learning" in concepts

    def test_no_concepts(self):
        text = "This is a plain text with no ML concepts."
        concepts = extract_concepts(text)
        assert len(concepts) == 0

    def test_multiple_concepts(self):
        text = """
        The course covers neural network training with gradient descent,
        regularization to prevent overfitting, and cross-validation
        for model evaluation. We also discuss feature engineering
        and hyperparameter tuning.
        """
        concepts = extract_concepts(text)
        assert len(concepts) >= 4
        assert "neural network" in concepts
        assert "regularization" in concepts

    def test_sorted_and_unique(self):
        text = "neural network neural network deep learning deep learning"
        concepts = extract_concepts(text)
        assert concepts == sorted(set(concepts))


class TestParseChapterNumber:
    def test_two_digit(self):
        assert _parse_chapter_number("08_ML Systems") == "08"

    def test_single_digit(self):
        assert _parse_chapter_number("3 Feature Engineering") == "03"

    def test_no_number(self):
        assert _parse_chapter_number("Introduction") == "00"

    def test_leading_number(self):
        assert _parse_chapter_number("12_Advanced Topics") == "12"


class TestScanChapters:
    def test_scan_empty_directory(self, tmp_path):
        chapters = scan_chapters(str(tmp_path))
        assert chapters == []

    def test_scan_nonexistent_directory(self, tmp_path):
        chapters = scan_chapters(str(tmp_path / "nonexistent"))
        assert chapters == []

    def test_scan_with_pdf_files(self, tmp_path):
        pdf_dir = tmp_path / "01_Intro"
        pdf_dir.mkdir()
        (pdf_dir / "chapter1.pdf").write_bytes(b"%PDF-1.4 dummy")

        chapters = scan_chapters(str(tmp_path))
        assert len(chapters) == 1
        assert chapters[0]["chapter_id"] == "ch_01"

    def test_scan_with_root_pdfs(self, tmp_path):
        (tmp_path / "05_Models.pdf").write_bytes(b"%PDF-1.4 dummy")
        chapters = scan_chapters(str(tmp_path))
        assert len(chapters) == 1
        assert chapters[0]["chapter_id"] == "ch_05"

    def test_scan_skips_dirs_without_pdfs(self, tmp_path):
        empty_dir = tmp_path / "03_Empty"
        empty_dir.mkdir()
        (empty_dir / "notes.txt").write_text("no pdfs here")

        chapters = scan_chapters(str(tmp_path))
        assert len(chapters) == 0
