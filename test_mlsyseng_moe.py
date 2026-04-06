"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def db(tmp_db):
    from src.mlsyseng_mcp.database import Database
    return Database(db_path=tmp_db)


class TestDatabase:
    def test_init_creates_tables(self, db):
        stats = db.get_stats()
        assert stats["total_chapters"] == 0
        assert stats["total_concepts"] == 0
        assert stats["total_experts"] == 0

    def test_upsert_chapter(self, db):
        chapter_id = db.upsert_chapter(
            chapter_name="Test Chapter",
            source_path="/tmp/test",
            markdown_content="# Test\nSome content",
            concept_count=3,
        )
        assert chapter_id > 0

        chapter = db.get_chapter("Test Chapter")
        assert chapter is not None
        assert chapter["chapter_name"] == "Test Chapter"
        assert chapter["concept_count"] == 3

    def test_upsert_chapter_updates(self, db):
        db.upsert_chapter("Ch1", "/a", "old content", 1)
        db.upsert_chapter("Ch1", "/b", "new content", 5)
        ch = db.get_chapter("Ch1")
        assert ch["markdown_content"] == "new content"
        assert ch["concept_count"] == 5

    def test_add_and_get_concepts(self, db):
        ch_id = db.upsert_chapter("Ch", "/tmp", "content")
        concepts = [
            {"name": "Gradient Descent", "description": "Optimization", "category": "optimization"},
            {"name": "Neural Network", "description": "Model", "category": "model"},
        ]
        db.add_concepts(ch_id, concepts)
        result = db.get_concepts(ch_id)
        assert len(result) == 2

    def test_search_concepts(self, db):
        ch_id = db.upsert_chapter("Ch", "/tmp", "content")
        db.add_concepts(ch_id, [
            {"name": "Gradient Descent", "description": "An optimization technique"},
        ])
        results = db.search_concepts("gradient")
        assert len(results) == 1
        assert results[0]["concept_name"] == "Gradient Descent"

    def test_upsert_expert(self, db):
        expert = {
            "expert_name": "Test Expert",
            "slug": "test_expert",
            "capabilities": ["Build models"],
            "skills": ["/path/to/skill"],
            "strategy": "Baseline -> Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        eid = db.upsert_expert(expert)
        assert eid > 0

        stored = db.get_expert("test_expert")
        assert stored is not None
        assert stored["expert_name"] == "Test Expert"
        assert stored["capabilities"] == ["Build models"]

    def test_list_experts(self, db):
        db.upsert_expert({"expert_name": "E1", "slug": "e1"})
        db.upsert_expert({"expert_name": "E2", "slug": "e2"})
        experts = db.list_experts()
        assert len(experts) == 2

    def test_extraction_log(self, db):
        db.log_extraction("Ch1", "running")
        db.log_extraction("Ch1", "completed", pages_extracted=10)
        status = db.get_extraction_status()
        assert len(status) == 2
        assert status[0]["status"] == "completed"

    def test_state_history(self, db):
        db.save_state("titanic", 0, [0.0, 0.0], float("inf"), False)
        db.save_state("titanic", 1, [0.1, 0.2], 0.224, False)
        history = db.get_state_history("titanic")
        assert len(history) == 2
        assert history[1]["l2_norm"] == 0.224


class TestDoclingWorker:
    def test_extract_concepts(self):
        from src.mlsyseng_mcp.docling_worker import extract_concepts

        text = """
        # Neural Network Optimization
        Gradient descent is a key optimization algorithm.
        Regularization helps prevent overfitting.
        Cross-validation is used for model evaluation.
        """
        concepts = extract_concepts(text)
        names = [c["name"].lower() for c in concepts]
        assert "gradient descent" in names
        assert "regularization" in names
        assert "overfitting" in names

    def test_categorize_concepts(self):
        from src.mlsyseng_mcp.docling_worker import extract_concepts

        text = "We apply gradient descent for optimization and use dropout for regularization."
        concepts = extract_concepts(text)
        categories = {c["name"].lower(): c["category"] for c in concepts}
        assert categories.get("gradient descent") == "optimization"
        assert categories.get("dropout") == "regularization"

    def test_discover_chapters_missing_path(self):
        from src.mlsyseng_mcp.docling_worker import discover_chapters

        chapters = discover_chapters("/nonexistent/path")
        assert chapters == []


class TestEmbeddingStore:
    def test_chunk_text(self):
        from src.mlsyseng_mcp.embeddings import EmbeddingStore

        text = " ".join([f"word{i}" for i in range(100)])
        chunks = EmbeddingStore._chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        assert "word0" in chunks[0]

    def test_hash_fallback_embed(self):
        from src.mlsyseng_mcp.embeddings import EmbeddingStore

        store = EmbeddingStore(chroma_path="/tmp/test_chroma_nonexistent")
        store._model = None
        vecs = store._embed(["hello world", "test text"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    def test_stats_unavailable(self):
        from src.mlsyseng_mcp.embeddings import EmbeddingStore

        store = EmbeddingStore()
        store._client = None
        store._collection = None
        stats = store.get_collection_stats()
        assert stats["status"] == "unavailable"


class TestExpertRegistry:
    def test_slugify(self):
        from src.mlsyseng_mcp.expert_registry import slugify

        assert slugify("08 ML Systems") == "08_ml_systems"
        assert slugify("Deep Learning & CNNs") == "deep_learning_cnns"

    def test_create_expert(self, db, tmp_path):
        from src.mlsyseng_mcp.expert_registry import ExpertRegistry

        reg = ExpertRegistry(db, skills_path=str(tmp_path / "skills"))

        concepts = [
            {"name": "Neural Network", "category": "model"},
            {"name": "Gradient Descent", "category": "optimization"},
            {"name": "Cross Validation", "category": "evaluation"},
        ]
        expert = reg.create_expert_from_chapter("08 ML Systems", concepts, chapter_id=1)

        assert expert["slug"] == "08_ml_systems"
        assert len(expert["capabilities"]) > 2
        assert len(expert["skills"]) > 0
        assert expert["strategy"] is not None

    def test_ask_expert(self, db, tmp_path):
        from src.mlsyseng_mcp.expert_registry import ExpertRegistry

        reg = ExpertRegistry(db, skills_path=str(tmp_path / "skills"))
        reg.create_expert_from_chapter("Test Ch", [{"name": "PCA", "category": "unsupervised"}])
        result = reg.ask_expert("test_ch", "How to reduce dimensions?")
        assert "Test Ch" in result["expert"]

    def test_ask_missing_expert(self, db, tmp_path):
        from src.mlsyseng_mcp.expert_registry import ExpertRegistry

        reg = ExpertRegistry(db, skills_path=str(tmp_path / "skills"))
        result = reg.ask_expert("nonexistent", "question")
        assert "error" in result


class TestLoopController:
    def test_l2_norm(self):
        from src.mlsyseng_mcp.loop_controller import l2_norm

        assert l2_norm([0, 0], [3, 4]) == 5.0
        assert l2_norm([1, 1], [1, 1]) == 0.0

    def test_l2_norm_unequal_lengths(self):
        from src.mlsyseng_mcp.loop_controller import l2_norm

        result = l2_norm([1.0], [1.0, 0.0])
        assert result == 0.0

    def test_convergence_loop(self, db):
        from src.mlsyseng_mcp.loop_controller import LoopController

        loop = LoopController(db, epsilon=0.01, max_iterations=20, patience=2)

        converge_val = 0.5

        def generator(iteration, current):
            factor = 1.0 / (iteration + 1)
            return [converge_val + factor * v for v in [0.1, 0.2]]

        result = loop.run_loop("test_comp", generator, expert_count=2)
        assert result["total_iterations"] > 0
        assert "stop_reason" in result

    def test_immediate_convergence(self, db):
        from src.mlsyseng_mcp.loop_controller import LoopController

        loop = LoopController(db, epsilon=1.0, max_iterations=10, patience=1)

        def generator(iteration, current):
            return [v + 0.0001 for v in current]

        result = loop.run_loop("immediate", generator, expert_count=2)
        assert result["converged"]

    def test_max_iterations_stop(self, db):
        from src.mlsyseng_mcp.loop_controller import LoopController

        loop = LoopController(db, epsilon=0.0000001, max_iterations=3, patience=10)

        def generator(iteration, current):
            return [v + 1.0 for v in current]

        result = loop.run_loop("maxiter", generator, expert_count=2)
        assert result["total_iterations"] == 3
        assert "max_iterations" in result["stop_reason"]


class TestSkillGenerator:
    def test_load_skills_yaml(self):
        from skill_generator import load_skills_yaml

        config = load_skills_yaml("skills.yaml")
        assert config["name"] == "mlsyseng-moe"
        assert "mcp_server" in config
        assert "platforms" in config

    def test_build_mcp_server_config(self):
        from skill_generator import build_mcp_server_config

        config = {
            "mcp_server": {
                "command": "python",
                "args": ["-m", "mlsyseng_mcp"],
                "env": {"TEST_VAR": "~/test"},
            }
        }
        result = build_mcp_server_config(config)
        assert result["command"] == "python"
        assert "~" not in result["env"]["TEST_VAR"]
