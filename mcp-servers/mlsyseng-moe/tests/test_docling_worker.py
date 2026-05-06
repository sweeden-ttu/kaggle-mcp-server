"""Tests for the docling worker module."""

import os
import tempfile

import pytest

import database as db
import docling_worker as dw


def test_extract_concepts():
    text = """
    This chapter covers gradient descent and backpropagation
    in neural networks. We discuss learning rate schedules
    and regularization techniques like dropout and batch normalization.
    """
    concepts = dw.extract_concepts(text)
    assert len(concepts) > 0
    concept_lower = [c.lower() for c in concepts]
    assert any("gradient" in c for c in concept_lower)
    assert any("neural" in c for c in concept_lower)


def test_extract_concepts_empty():
    concepts = dw.extract_concepts("")
    assert concepts == []


def test_infer_concepts_from_name():
    concepts = dw._infer_concepts_from_name("Neural Networks and Deep Learning")
    assert "Deep Learning" in concepts or "Neural Network" in concepts

    concepts_unknown = dw._infer_concepts_from_name("Unknown Topic 42")
    assert "Machine Learning" in concepts_unknown


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db.init_db(path)
    yield path
    os.unlink(path)


def test_extract_chapter_no_pdf(temp_db):
    result = dw.extract_chapter(
        "Test Chapter", "/nonexistent/path", None, db_path=temp_db
    )
    assert result["status"] == "completed"
    assert "No PDF" in result["message"]


def test_scan_chapter_folders_nonexistent():
    folders = dw.scan_chapter_folders("/definitely/does/not/exist")
    assert folders == []
