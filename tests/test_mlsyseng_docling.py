"""Tests for MLSysEng MoE docling worker module."""

import os
import tempfile

import pytest

from src.mlsyseng_mcp.docling_worker import (
    _categorize_concept,
    _slugify,
    extract_concepts,
    scan_chapters,
)


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello_world"

    def test_with_numbers(self):
        assert _slugify("01 Introduction to ML") == "01_introduction_to_ml"

    def test_special_chars(self):
        assert _slugify("ML/AI - Concepts!") == "mlai_concepts"

    def test_multiple_spaces(self):
        assert _slugify("deep   learning") == "deep_learning"


class TestExtractConcepts:
    def test_finds_gradient_descent(self):
        text = "We use gradient descent to minimize the loss function."
        concepts = extract_concepts(text)
        names = {c["concept_name"] for c in concepts}
        assert "Gradient Descent" in names

    def test_finds_neural_network(self):
        text = "A neural network consists of layers of connected nodes."
        concepts = extract_concepts(text)
        names = {c["concept_name"] for c in concepts}
        assert "Neural Network" in names

    def test_finds_multiple_concepts(self):
        text = (
            "We apply regularization to the neural network "
            "and use cross-validation for evaluation."
        )
        concepts = extract_concepts(text)
        names = {c["concept_name"] for c in concepts}
        assert len(names) >= 2

    def test_empty_text(self):
        assert extract_concepts("") == []

    def test_no_concepts(self):
        text = "The weather is nice today."
        assert extract_concepts(text) == []


class TestCategorizeConcept:
    def test_algorithm(self):
        assert _categorize_concept("Gradient Descent") == "algorithm"

    def test_architecture(self):
        assert _categorize_concept("Neural Network") == "architecture"

    def test_technique(self):
        assert _categorize_concept("Regularization") == "technique"

    def test_metric(self):
        assert _categorize_concept("F1 Score") == "metric"

    def test_theory(self):
        assert _categorize_concept("Bayesian Inference") == "theory"

    def test_general(self):
        assert _categorize_concept("Unknown Thing") == "general"


class TestScanChapters:
    def test_scan_empty_dir(self, tmp_path):
        chapters = scan_chapters(str(tmp_path))
        assert chapters == []

    def test_scan_with_pdfs(self, tmp_path):
        pdf = tmp_path / "chapter1.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        chapters = scan_chapters(str(tmp_path))
        assert len(chapters) == 1
        assert chapters[0]["name"] == "chapter1"
        assert chapters[0]["slug"] == "chapter1"

    def test_scan_with_subdirs(self, tmp_path):
        subdir = tmp_path / "01_Introduction"
        subdir.mkdir()
        (subdir / "intro.pdf").write_bytes(b"%PDF-1.4 fake")
        chapters = scan_chapters(str(tmp_path))
        assert len(chapters) == 1
        assert chapters[0]["name"] == "01_Introduction"

    def test_scan_nonexistent(self):
        chapters = scan_chapters("/nonexistent/path")
        assert chapters == []
