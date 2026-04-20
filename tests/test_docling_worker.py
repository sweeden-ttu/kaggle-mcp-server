"""Tests for mlsyseng_moe.docling_worker module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from mlsyseng_moe.database import Database
from mlsyseng_moe.docling_worker import (
    _make_title,
    _parse_chapter_number,
    extract_concepts,
    scan_chapters,
)


class TestMakeTitle:
    def test_numbered_folder(self):
        assert _make_title("08_ML_Systems") == "Chapter 8: ML Systems"

    def test_no_number(self):
        assert _make_title("Introduction") == "Introduction"

    def test_leading_zero(self):
        assert _make_title("01_Introduction") == "Chapter 1: Introduction"


class TestParseChapterNumber:
    def test_with_number(self):
        assert _parse_chapter_number("08_ML_Systems") == "08"

    def test_without_number(self):
        assert _parse_chapter_number("Introduction") is None


class TestExtractConcepts:
    def test_finds_known_concepts(self):
        text = "This chapter covers neural network architectures and gradient descent optimization."
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "gradient descent" in concepts

    def test_case_insensitive(self):
        text = "NEURAL NETWORK and Deep Learning"
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "deep learning" in concepts

    def test_empty_text(self):
        assert extract_concepts("") == []

    def test_no_concepts(self):
        text = "This is a simple paragraph with no ML concepts."
        concepts = extract_concepts(text)
        assert len(concepts) == 0


class TestScanChapters:
    def test_scan_nonexistent_directory(self):
        result = scan_chapters("/nonexistent/path")
        assert result == []

    def test_scan_empty_directory(self, tmp_path):
        result = scan_chapters(str(tmp_path))
        assert result == []

    def test_scan_with_chapters(self, tmp_path):
        ch1_dir = tmp_path / "01_Introduction"
        ch1_dir.mkdir()
        (ch1_dir / "chapter1.pdf").write_bytes(b"fake pdf content")

        ch2_dir = tmp_path / "02_LinearAlgebra"
        ch2_dir.mkdir()
        (ch2_dir / "chapter2.pdf").write_bytes(b"fake pdf content")

        no_pdf_dir = tmp_path / "03_NoPDF"
        no_pdf_dir.mkdir()

        result = scan_chapters(str(tmp_path))
        assert len(result) == 2
        assert result[0]["folder_name"] == "01_Introduction"
        assert result[1]["folder_name"] == "02_LinearAlgebra"
