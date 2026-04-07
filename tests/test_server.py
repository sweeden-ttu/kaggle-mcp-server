"""Tests for mlsyseng_moe.server module.

Tests the MCP tool functions directly without starting a server.
"""

import json
import os

import pytest

import src.mlsyseng_moe.server as server_module
from src.mlsyseng_moe.database import Database


@pytest.fixture(autouse=True)
def reset_server_globals(tmp_path):
    """Reset server globals before each test and use temp DB."""
    db = Database(db_path=str(tmp_path / "test.db"))
    db.initialize()

    server_module._db = db
    server_module._embeddings = None
    server_module._registry = None
    server_module._loop = None

    yield

    db.close()
    server_module._db = None
    server_module._embeddings = None
    server_module._registry = None
    server_module._loop = None


class TestExtractKnowledge:
    def test_returns_json(self, tmp_path):
        os.environ["ML_PRINCIPLES_PATH"] = str(tmp_path / "empty_chapters")
        (tmp_path / "empty_chapters").mkdir()
        result = json.loads(server_module.extract_knowledge(force_reindex=False))
        assert "extraction" in result
        assert "experts_created" in result


class TestListExperts:
    def test_empty_initially(self):
        result = json.loads(server_module.list_experts())
        assert result == []

    def test_after_registration(self):
        db = server_module._get_db()
        db.upsert_chapter("01", "Test", "/ch1", "neural network content", ["neural network"])
        registry = server_module._get_registry()
        registry.register_from_chapters()

        result = json.loads(server_module.list_experts())
        assert len(result) == 1
        assert result[0]["expert_name"] is not None


class TestBuildEntry:
    def test_no_experts(self):
        result = json.loads(server_module.build_entry("titanic"))
        assert "error" in result

    def test_with_experts(self):
        db = server_module._get_db()
        db.upsert_chapter("01", "ML Basics", "/ch1", "content", ["neural network"])
        registry = server_module._get_registry()
        registry.register_from_chapters()

        result = json.loads(server_module.build_entry("titanic"))
        assert result["competition"] == "titanic"
        assert result["experts_selected"] > 0
        assert "recommended_skills" in result


class TestEvolve:
    def test_with_registered_experts(self):
        db = server_module._get_db()
        db.upsert_chapter("01", "ML Basics", "/ch1", "content", ["neural network"])
        registry = server_module._get_registry()
        registry.register_from_chapters()

        result = json.loads(server_module.evolve("titanic"))
        assert result["competition"] == "titanic"
        assert "iterations" in result


class TestAskExpert:
    def test_nonexistent_expert(self):
        result = json.loads(server_module.ask_expert("nonexistent", "How to train?"))
        assert "error" in result

    def test_existing_expert(self):
        db = server_module._get_db()
        db.upsert_chapter("01", "Deep Learning", "/ch1", "content", ["deep learning"])
        registry = server_module._get_registry()
        registry.register_from_chapters()
        experts = registry.list_experts()

        result = json.loads(
            server_module.ask_expert(experts[0]["expert_name"], "How to train?")
        )
        assert "capabilities" in result
        assert "answer_context" in result

    def test_fuzzy_match(self):
        db = server_module._get_db()
        db.upsert_chapter("01", "Deep Learning", "/ch1", "content", ["deep learning"])
        registry = server_module._get_registry()
        registry.register_from_chapters()

        result = json.loads(server_module.ask_expert("deep", "How to train?"))
        assert "capabilities" in result


class TestGetStats:
    def test_returns_stats(self):
        result = json.loads(server_module.get_stats())
        assert "database" in result
        assert result["database"]["chapters"] == 0


class TestGetExtractionStatus:
    def test_empty_status(self):
        result = json.loads(server_module.get_extraction_status())
        assert result == []


class TestRunRdagent:
    def test_returns_context(self):
        result = json.loads(server_module.run_rdagent("titanic", "Predict survival"))
        assert result["competition"] == "titanic"
        assert "rdagent_command" in result
