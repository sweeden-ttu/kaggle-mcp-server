"""Tests for the embeddings module."""

import os
import tempfile

import pytest

import database as db
import embeddings as emb


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db.init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def temp_chroma(tmp_path):
    return str(tmp_path / "chroma_test")


def test_chunk_text():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = emb._chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_text_short():
    chunks = emb._chunk_text("short text", chunk_size=100)
    assert len(chunks) == 1


def test_chunk_text_empty():
    chunks = emb._chunk_text("", chunk_size=100)
    assert chunks == []


def test_index_chapter(temp_chroma):
    emb._chroma_client = None
    emb._collection = None
    emb._embedding_fn = None
    n = emb.index_chapter(
        chapter_id=1,
        chapter_name="Test Chapter",
        text="This is about neural networks and deep learning optimization",
        concepts=["Neural Network", "Optimization"],
        chroma_path=temp_chroma,
    )
    assert n > 0


def test_search_empty(temp_chroma):
    emb._chroma_client = None
    emb._collection = None
    emb._embedding_fn = None
    results = emb.search("test query", n_results=5, chroma_path=temp_chroma)
    assert isinstance(results, list)


def test_index_and_search(temp_chroma):
    emb._chroma_client = None
    emb._collection = None
    emb._embedding_fn = None
    emb.index_chapter(
        chapter_id=1,
        chapter_name="Neural Networks",
        text="Neural networks are computational models inspired by biological neurons. They use backpropagation for training.",
        concepts=["Neural Network", "Backpropagation"],
        chroma_path=temp_chroma,
    )
    results = emb.search("how do neural networks learn", n_results=3, chroma_path=temp_chroma)
    assert len(results) > 0
    assert results[0]["chapter_name"] == "Neural Networks"
