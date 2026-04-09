"""Tests for the MLSysEng MoE system."""

import json
import math
import os
import tempfile

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def db(tmp_db):
    from mlsyseng_mcp.database import Database

    return Database(db_path=tmp_db)


class TestDatabase:
    def test_init_creates_tables(self, db):
        stats = db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0

    def test_upsert_chapter(self, db):
        chapter_id = db.upsert_chapter(
            chapter_num=1,
            title="Test Chapter",
            source_path="/tmp/test.pdf",
            markdown_content="# Test\nSome content about neural networks",
            concepts=["Neural Network", "Gradient Descent"],
            page_count=10,
        )
        assert chapter_id > 0

        chapter = db.get_chapter(1)
        assert chapter is not None
        assert chapter["title"] == "Test Chapter"
        assert chapter["page_count"] == 10
        assert "Neural Network" in chapter["concepts"]

    def test_upsert_chapter_updates(self, db):
        db.upsert_chapter(1, "Ch1", "/a", "old content", ["concept1"], 5)
        db.upsert_chapter(1, "Ch1 Updated", "/b", "new content", ["concept2"], 10)
        ch = db.get_chapter(1)
        assert ch["markdown_content"] == "new content"
        assert ch["title"] == "Ch1 Updated"
        assert ch["page_count"] == 10

    def test_list_chapters(self, db):
        db.upsert_chapter(1, "Chapter 1", "/a", "c1", ["a"])
        db.upsert_chapter(2, "Chapter 2", "/b", "c2", ["b"])
        chapters = db.list_chapters()
        assert len(chapters) == 2
        assert chapters[0]["chapter_num"] == 1
        assert chapters[1]["chapter_num"] == 2

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
        assert stored["skills"] == ["/path/to/skill"]

    def test_list_experts(self, db):
        db.upsert_expert({"expert_name": "E1", "slug": "e1"})
        db.upsert_expert({"expert_name": "E2", "slug": "e2"})
        experts = db.list_experts()
        assert len(experts) == 2

    def test_get_expert_not_found(self, db):
        result = db.get_expert("nonexistent")
        assert result is None

    def test_extraction_status(self, db):
        db.set_extraction_status(1, "started")
        db.set_extraction_status(1, "completed")
        statuses = db.get_extraction_status()
        assert len(statuses) >= 1

    def test_get_all_chapter_content(self, db):
        db.upsert_chapter(1, "Ch1", "/a", "content one", ["c1"])
        db.upsert_chapter(2, "Ch2", "/b", "content two", ["c2"])
        content = db.get_all_chapter_content()
        assert len(content) == 2
        assert content[0]["markdown_content"] == "content one"

    def test_stats(self, db):
        db.upsert_chapter(1, "Ch1", "/a", "content", ["c1"], 5)
        db.upsert_expert({"expert_name": "E1", "slug": "e1"})
        stats = db.get_stats()
        assert stats["chapters_indexed"] == 1
        assert stats["experts_registered"] == 1
        assert stats["total_pages_extracted"] == 5


class TestDoclingWorker:
    def test_extract_concepts(self):
        from mlsyseng_mcp.docling_worker import _extract_concepts

        text = """
        Neural Network Optimization
        Gradient descent is a key optimization algorithm.
        Regularization helps prevent overfitting.
        Cross-validation is used for model evaluation.
        """
        concepts = _extract_concepts(text)
        concepts_lower = [c.lower() for c in concepts]
        assert "gradient descent" in concepts_lower
        assert "regularization" in concepts_lower
        assert "overfitting" in concepts_lower

    def test_extract_concepts_deep_learning(self):
        from mlsyseng_mcp.docling_worker import _extract_concepts

        text = "Deep learning with convolutional neural networks and a transformer model"
        concepts = _extract_concepts(text)
        concepts_lower = [c.lower() for c in concepts]
        assert "deep learning" in concepts_lower
        assert "convolutional" in concepts_lower
        assert "transformer" in concepts_lower

    def test_parse_chapter_info(self):
        from mlsyseng_mcp.docling_worker import _parse_chapter_info

        result = _parse_chapter_info("01 - Introduction to ML")
        assert result is not None
        assert result[0] == 1
        assert "Introduction" in result[1]

        result2 = _parse_chapter_info("Chapter 3 - Optimization")
        assert result2 is not None
        assert result2[0] == 3

    def test_parse_chapter_info_underscore(self):
        from mlsyseng_mcp.docling_worker import _parse_chapter_info

        result = _parse_chapter_info("02_Deep_Learning")
        assert result is not None
        assert result[0] == 2
        assert "Deep" in result[1]

    def test_parse_chapter_info_invalid(self):
        from mlsyseng_mcp.docling_worker import _parse_chapter_info

        result = _parse_chapter_info("random_folder")
        assert result is None

    def test_scan_chapters_missing_path(self):
        from mlsyseng_mcp.docling_worker import scan_chapters

        chapters = scan_chapters("/nonexistent/path")
        assert chapters == []

    def test_extract_all_chapters_no_chapters(self, db):
        from mlsyseng_mcp.docling_worker import extract_all_chapters

        results = extract_all_chapters("/nonexistent/path", db)
        assert len(results) == 1
        assert results[0]["status"] == "no_chapters"


class TestEmbeddings:
    def test_chunk_text(self):
        from mlsyseng_mcp.embeddings import _chunk_text

        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        assert "word0" in chunks[0]

    def test_chunk_text_empty(self):
        from mlsyseng_mcp.embeddings import _chunk_text

        chunks = _chunk_text("")
        assert chunks == []

    def test_chunk_text_small(self):
        from mlsyseng_mcp.embeddings import _chunk_text

        text = "hello world"
        chunks = _chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_embedding_store_init(self, tmp_path):
        from mlsyseng_mcp.embeddings import EmbeddingStore

        store = EmbeddingStore(chroma_path=str(tmp_path / "chroma"))
        assert store.chroma_path == str(tmp_path / "chroma")
        assert store.model_name == "all-MiniLM-L6-v2"


class TestExpertRegistry:
    def test_slugify(self):
        from mlsyseng_mcp.expert_registry import _slugify

        assert _slugify("ML Systems") == "ml_systems"
        assert _slugify("Deep Learning & CNNs") == "deep_learning_cnns"
        assert _slugify("  test  ") == "test"

    def test_create_expert_from_chapter(self):
        from mlsyseng_mcp.expert_registry import create_expert_from_chapter

        expert = create_expert_from_chapter(
            chapter_num=8,
            title="ML Systems",
            concepts=["Neural Network", "Gradient Descent", "Cross Validation"],
            chapter_id=1,
        )
        assert expert["slug"] == "08_ml_systems"
        assert expert["expert_name"] == "08_ML_Systems"
        assert len(expert["capabilities"]) >= 3
        assert len(expert["skills"]) > 0
        assert expert["strategy"] is not None
        assert expert["formula"]["objective"] == "minimize_validation_loss"
        assert expert["loop_config"]["epsilon"] == 0.001

    def test_create_expert_deep_learning(self):
        from mlsyseng_mcp.expert_registry import create_expert_from_chapter

        expert = create_expert_from_chapter(
            chapter_num=5,
            title="Deep Learning",
            concepts=["Deep Learning", "Neural Network", "Convolutional"],
        )
        capabilities = expert["capabilities"]
        assert any("Deep learning" in c for c in capabilities)
        assert any("Computer vision" in c for c in capabilities)

    def test_infer_skills_from_concepts(self):
        from mlsyseng_mcp.expert_registry import _infer_skills_from_concepts

        skills = _infer_skills_from_concepts(["Neural Network", "Deep Learning"])
        assert "model-trainer" in skills
        assert "deep-learning-pipeline" in skills
        assert "kaggle-preprocessor" in skills

    def test_build_formula_regression(self):
        from mlsyseng_mcp.expert_registry import _build_formula

        formula = _build_formula(["Regression", "Time Series"])
        assert "rmse" in formula["metrics"]

    def test_build_formula_classification(self):
        from mlsyseng_mcp.expert_registry import _build_formula

        formula = _build_formula(["Classification", "Precision"])
        assert "f1_score" in formula["metrics"]

    def test_register_experts_from_db(self, db):
        from mlsyseng_mcp.expert_registry import register_experts_from_db

        db.upsert_chapter(1, "Intro to ML", "/a", "content", ["Neural Network"])
        db.upsert_chapter(2, "Deep Learning", "/b", "content", ["Deep Learning"])
        results = register_experts_from_db(db)
        assert len(results) == 2
        assert results[0]["expert_name"] is not None

        experts = db.list_experts()
        assert len(experts) == 2

    def test_save_expert_json(self, tmp_path):
        from mlsyseng_mcp.expert_registry import save_expert_json

        expert = {
            "expert_name": "Test",
            "slug": "test",
            "capabilities": ["a"],
            "skills": [],
        }
        path = save_expert_json(expert, str(tmp_path / "experts"))
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["expert_name"] == "Test"

    def test_infer_experts_without_embeddings(self, db):
        from mlsyseng_mcp.expert_registry import infer_experts_for_competition

        db.upsert_expert({
            "expert_name": "ML Expert",
            "slug": "ml_expert",
            "capabilities": ["Machine learning model building"],
        })
        db.upsert_expert({
            "expert_name": "NLP Expert",
            "slug": "nlp_expert",
            "capabilities": ["Natural language processing"],
        })
        experts = infer_experts_for_competition("language model", db, top_k=1)
        assert len(experts) == 1


class TestLoopController:
    def test_l2_norm_diff(self):
        from mlsyseng_mcp.loop_controller import l2_norm_diff

        assert l2_norm_diff([0, 0], [3, 4]) == 5.0
        assert l2_norm_diff([1, 1], [1, 1]) == 0.0

    def test_l2_norm_diff_unequal_lengths(self):
        from mlsyseng_mcp.loop_controller import l2_norm_diff

        with pytest.raises(ValueError):
            l2_norm_diff([1.0], [1.0, 0.0])

    def test_convergence_loop(self):
        from mlsyseng_mcp.loop_controller import LoopController, LoopState

        controller = LoopController(
            epsilon=0.01, max_iterations=20, patience=2
        )

        def step_fn(iteration, prev_state):
            factor = 1.0 / (iteration + 1)
            base = prev_state if prev_state else [0.5, 0.5]
            state = [v + factor * 0.01 for v in base]
            return LoopState(
                iteration=iteration,
                state_vector=state,
                metrics={"loss": state[0]},
                expert_contributions=["expert1"],
            )

        result = controller.run(step_fn)
        assert result.total_iterations > 0
        assert result.exit_reason is not None

    def test_immediate_convergence(self):
        from mlsyseng_mcp.loop_controller import LoopController, LoopState

        controller = LoopController(
            epsilon=1.0, max_iterations=10, patience=1
        )

        def step_fn(iteration, prev_state):
            base = prev_state if prev_state else [1.0, 1.0]
            state = [v + 0.0001 for v in base]
            return LoopState(
                iteration=iteration,
                state_vector=state,
                metrics={"loss": state[0]},
                expert_contributions=["e1"],
            )

        result = controller.run(step_fn)
        assert result.converged

    def test_max_iterations_stop(self):
        from mlsyseng_mcp.loop_controller import LoopController, LoopState

        controller = LoopController(
            epsilon=0.0000001, max_iterations=3, patience=10
        )

        def step_fn(iteration, prev_state):
            base = prev_state if prev_state else [0.0, 0.0]
            state = [v + 1.0 for v in base]
            return LoopState(
                iteration=iteration,
                state_vector=state,
                metrics={"loss": state[0]},
                expert_contributions=["e1"],
            )

        result = controller.run(step_fn)
        assert result.total_iterations == 3
        assert "max_iterations" in result.exit_reason

    def test_from_config(self):
        from mlsyseng_mcp.loop_controller import LoopController

        config = {
            "objective": "maximize_accuracy",
            "epsilon": 0.01,
            "max_iterations": 5,
            "patience": 2,
        }
        controller = LoopController.from_config(config)
        assert controller.objective == "maximize_accuracy"
        assert controller.epsilon == 0.01
        assert controller.max_iterations == 5
        assert controller.patience == 2

    def test_to_dict(self):
        from mlsyseng_mcp.loop_controller import LoopController, LoopResult

        controller = LoopController(epsilon=0.01, max_iterations=5)
        result = LoopResult(
            converged=True,
            total_iterations=3,
            final_metrics={"loss": 0.1},
            convergence_history=[],
            final_state=[0.1, 0.9],
            exit_reason="converged_after_3_iterations",
        )
        d = controller.to_dict(result)
        assert d["converged"] is True
        assert d["total_iterations"] == 3
        assert d["config"]["epsilon"] == 0.01

    def test_build_competition_step(self):
        from mlsyseng_mcp.loop_controller import build_competition_step

        experts = [
            {"expert_name": "Expert1"},
            {"expert_name": "Expert2"},
        ]
        state = build_competition_step(experts, "titanic", 0, None)
        assert state.iteration == 0
        assert len(state.state_vector) == 3
        assert "loss" in state.metrics
        assert "accuracy" in state.metrics

        state2 = build_competition_step(experts, "titanic", 1, state.state_vector)
        assert state2.metrics["loss"] < state.metrics["loss"]


class TestSkillGenerator:
    def test_load_skills_yaml(self):
        from skill_generator import load_skills_yaml

        config = load_skills_yaml("skills.yaml")
        assert config["name"] == "mlsyseng-moe"
        assert "mcp_server" in config
        assert "platforms" in config
        assert "tools" in config
        assert "skill_content" in config

    def test_mcp_server_config(self):
        from skill_generator import _mcp_server_config

        config = {
            "mcp_server": {
                "command": "python",
                "args": ["-m", "mlsyseng_mcp"],
                "env": {"TEST_VAR": "value"},
            }
        }
        result = _mcp_server_config(config)
        assert result["command"] == "python"
        assert result["args"] == ["-m", "mlsyseng_mcp"]
