"""Tests for mlsyseng_moe.embeddings module.

These tests focus on the text chunking logic, which does not require
external dependencies (sentence-transformers, chromadb). Full integration
tests are skipped when those libraries are not installed.
"""

import pytest

from src.mlsyseng_moe.embeddings import _chunk_text, EmbeddingStore


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world this is a short text"
        chunks = _chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_long_text(self):
        words = ["word"] * 100
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 30

    def test_overlap_creates_redundancy(self):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) >= 3
        chunk1_words = set(chunks[0].split())
        chunk2_words = set(chunks[1].split())
        assert len(chunk1_words & chunk2_words) > 0

    def test_empty_text(self):
        chunks = _chunk_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_exact_chunk_size(self):
        words = ["word"] * 512
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1


class TestEmbeddingStoreInit:
    def test_default_model(self):
        store = EmbeddingStore(chroma_path="/tmp/test_chroma")
        assert store.model_name == "all-MiniLM-L6-v2"

    def test_custom_model(self):
        store = EmbeddingStore(
            chroma_path="/tmp/test_chroma",
            model_name="custom-model",
        )
        assert store.model_name == "custom-model"

    def test_custom_chroma_path(self):
        store = EmbeddingStore(chroma_path="/custom/path")
        assert store.chroma_path == "/custom/path"


class TestEmbeddingStoreIntegration:
    """Integration tests that require sentence-transformers and chromadb.
    Skipped if libraries are not available."""

    @pytest.fixture
    def store(self, tmp_path):
        try:
            import sentence_transformers
            import chromadb
        except ImportError:
            pytest.skip("sentence-transformers or chromadb not installed")
        return EmbeddingStore(chroma_path=str(tmp_path / "chroma"))

    def test_embed_texts(self, store):
        embeddings = store.embed_texts(["hello world", "test query"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0

    def test_index_and_search(self, store):
        store.index_chapter(
            chapter_id="01",
            title="Test Chapter",
            content="Neural networks use gradient descent for optimization. " * 20,
            concepts=["neural network", "gradient descent"],
        )
        results = store.search("gradient optimization", n_results=3)
        assert len(results) > 0
        assert results[0]["chapter_id"] == "01"

    def test_infer_skills(self, store):
        store.index_chapter(
            chapter_id="01",
            title="Deep Learning",
            content="Deep learning with neural networks and transformers. " * 20,
            concepts=["deep learning", "neural network"],
        )
        skills = store.infer_skills("image classification with neural nets")
        assert len(skills) > 0
        assert skills[0]["chapter_id"] == "01"

    def test_get_stats(self, store):
        stats = store.get_stats()
        assert "total_chunks" in stats
