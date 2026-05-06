"""Tests for the MLSysEng MoE system."""

import json
import math
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mlsyseng_mcp.database import MLSysEngDB
from mlsyseng_mcp.docling_worker import (
    _categorize_concept,
    discover_chapter_folders,
    extract_concepts_from_text,
)
from mlsyseng_mcp.expert_registry import ExpertRegistry, _slugify
from mlsyseng_mcp.loop_controller import (
    ConvergenceResult,
    LoopController,
    StateVector,
    default_step_fn,
    l2_norm,
)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = MLSysEngDB(db_path=self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_upsert_and_get_chapter(self):
        chapter_id = self.db.upsert_chapter(
            chapter_num=1, title="Introduction", source_path="/test/01",
            markdown_content="# Intro\nTest content", page_count=10, status="extracted",
        )
        self.assertIsNotNone(chapter_id)

        chapter = self.db.get_chapter(1)
        self.assertIsNotNone(chapter)
        self.assertEqual(chapter["title"], "Introduction")
        self.assertEqual(chapter["page_count"], 10)

    def test_get_all_chapters(self):
        self.db.upsert_chapter(1, "Chapter 1", "/test/01", "content1")
        self.db.upsert_chapter(2, "Chapter 2", "/test/02", "content2")
        chapters = self.db.get_all_chapters()
        self.assertEqual(len(chapters), 2)

    def test_add_and_get_concepts(self):
        chapter_id = self.db.upsert_chapter(1, "Test", "/test", "content")
        self.db.add_concept(chapter_id, "neural network", "A computing system", "architecture", 0.9)
        self.db.add_concept(chapter_id, "gradient descent", "Optimization method", "optimization", 0.8)

        concepts = self.db.get_concepts_for_chapter(chapter_id)
        self.assertEqual(len(concepts), 2)
        self.assertEqual(concepts[0]["concept_name"], "neural network")

    def test_upsert_and_get_expert(self):
        chapter_id = self.db.upsert_chapter(1, "Test", "/test", "content")
        self.db.upsert_expert(
            expert_name="01_Test", slug="01_test", chapter_id=chapter_id,
            capabilities=["Build models"], skills=["/skills/test"],
            strategy="Baseline → Submit", formula={"objective": "minimize_loss"},
            loop_config={"epsilon": 0.001, "max_iterations": 10},
        )

        expert = self.db.get_expert("01_test")
        self.assertIsNotNone(expert)
        self.assertEqual(expert["expert_name"], "01_Test")
        self.assertIsInstance(expert["capabilities"], list)
        self.assertIsInstance(expert["formula"], dict)

    def test_convergence_states(self):
        self.db.save_convergence_state("titanic", 1, [0.1, 0.2], 0.5, False)
        self.db.save_convergence_state("titanic", 2, [0.3, 0.4], 0.1, True)

        history = self.db.get_convergence_history("titanic")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["iteration"], 1)
        self.assertFalse(history[0]["converged"])
        self.assertTrue(history[1]["converged"])

    def test_get_stats(self):
        stats = self.db.get_stats()
        self.assertIn("chapters_total", stats)
        self.assertIn("experts", stats)


class TestDoclingWorker(unittest.TestCase):
    def test_extract_concepts_from_text(self):
        text = """
        Neural network architectures use gradient descent for optimization.
        Regularization helps prevent overfitting. The transformer architecture
        uses attention mechanisms. Feature engineering is crucial for ML.
        """
        concepts = extract_concepts_from_text(text)
        self.assertGreater(len(concepts), 0)
        concept_names = [c["concept_name"] for c in concepts]
        self.assertIn("neural network", concept_names)
        self.assertIn("gradient descent", concept_names)

    def test_categorize_concept(self):
        self.assertEqual(_categorize_concept("neural network"), "architecture")
        self.assertEqual(_categorize_concept("gradient descent"), "optimization")
        self.assertEqual(_categorize_concept("regularization"), "regularization")
        self.assertEqual(_categorize_concept("f1"), "evaluation")
        self.assertEqual(_categorize_concept("unknown_thing"), "general")

    def test_discover_nonexistent_path(self):
        chapters = discover_chapter_folders("/nonexistent/path")
        self.assertEqual(chapters, [])


class TestExpertRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = MLSysEngDB(db_path=self.tmp.name)
        self.registry = ExpertRegistry(db=self.db, skills_path="/test/skills")

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_slugify(self):
        self.assertEqual(_slugify("01_ML Systems"), "01_ml_systems")
        self.assertEqual(_slugify("Deep Learning Basics"), "deep_learning_basics")
        self.assertEqual(_slugify("Test--Slug"), "test_slug")

    def test_register_expert(self):
        chapter_id = self.db.upsert_chapter(1, "ML Systems", "/test/01", "content")
        concepts = [
            {"concept_name": "neural network", "category": "architecture", "importance": 0.9},
            {"concept_name": "gradient descent", "category": "optimization", "importance": 0.8},
        ]

        expert = self.registry.register_expert_from_chapter(
            chapter_num=1, title="ML Systems", chapter_id=chapter_id, concepts=concepts,
        )

        self.assertEqual(expert["expert_name"], "01_ML Systems")
        self.assertEqual(expert["slug"], "01_ml_systems")
        self.assertIn("Build baseline models quickly", expert["capabilities"])
        self.assertGreater(len(expert["skills"]), 0)

    def test_list_experts(self):
        chapter_id = self.db.upsert_chapter(1, "Test", "/test", "content")
        self.registry.register_expert_from_chapter(1, "Test", chapter_id, [])
        experts = self.registry.list_experts()
        self.assertEqual(len(experts), 1)

    def test_export_expert_definition(self):
        chapter_id = self.db.upsert_chapter(1, "Test", "/test", "content")
        self.registry.register_expert_from_chapter(1, "Test", chapter_id, [])
        definition = self.registry.export_expert_definition("01_test")
        self.assertIsNotNone(definition)
        self.assertIn("expert_name", definition)
        self.assertIn("formula", definition)
        self.assertIn("loop_config", definition)


class TestLoopController(unittest.TestCase):
    def test_l2_norm(self):
        self.assertAlmostEqual(l2_norm([0, 0], [3, 4]), 5.0)
        self.assertAlmostEqual(l2_norm([1, 1, 1], [1, 1, 1]), 0.0)

    def test_l2_norm_different_lengths(self):
        result = l2_norm([1, 2], [1, 2, 3])
        self.assertAlmostEqual(result, 3.0)

    def test_state_vector(self):
        sv = StateVector([0.5, 0.3, 0.8])
        self.assertEqual(sv.to_list(), [0.5, 0.3, 0.8])

    def test_state_vector_from_scores(self):
        sv = StateVector.from_scores({"accuracy": 0.9, "f1": 0.85, "loss": 0.1})
        self.assertEqual(len(sv.values), 3)

    def test_state_vector_distance(self):
        a = StateVector([0, 0])
        b = StateVector([3, 4])
        self.assertAlmostEqual(a.distance(b), 5.0)

    def test_convergence_loop(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            db = MLSysEngDB(db_path=tmp.name)
            loop = LoopController(db=db, epsilon=0.01, max_iterations=20, patience=2)

            result = loop.run("test_comp", step_fn=default_step_fn)

            self.assertTrue(result.converged)
            self.assertGreater(result.total_iterations, 0)
            self.assertLess(result.final_l2_norm, 0.01)
        finally:
            os.unlink(tmp.name)

    def test_convergence_result(self):
        result = ConvergenceResult()
        self.assertFalse(result.converged)
        self.assertIsNone(result.final_state)

        d = result.to_dict()
        self.assertIn("converged", d)
        self.assertIn("total_iterations", d)

    def test_check_convergence(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            db = MLSysEngDB(db_path=tmp.name)
            loop = LoopController(db=db)

            status = loop.check_convergence("nonexistent")
            self.assertEqual(status["status"], "no_history")
        finally:
            os.unlink(tmp.name)


class TestSkillsYaml(unittest.TestCase):
    def test_skills_yaml_exists(self):
        yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
        self.assertTrue(os.path.exists(yaml_path))

    def test_skills_yaml_valid(self):
        import yaml
        yaml_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        self.assertEqual(config["name"], "mlsyseng-moe")
        self.assertIn("tools", config)
        self.assertIn("platforms", config)
        self.assertGreater(len(config["tools"]), 0)


class TestSkillGenerator(unittest.TestCase):
    def test_expand_env(self):
        sys.path.insert(0, os.path.dirname(__file__))
        from skill_generator import expand_env

        os.environ["TEST_VAR"] = "hello"
        self.assertEqual(expand_env("${TEST_VAR:-default}"), "hello")
        self.assertEqual(expand_env("${NONEXISTENT:-fallback}"), "fallback")
        del os.environ["TEST_VAR"]

    def test_get_mcp_server_config(self):
        from skill_generator import get_mcp_server_config

        config = {
            "mcp_server": {
                "command": "python",
                "args": ["-m", "mlsyseng_mcp.server"],
                "env": {"TEST": "${NONEXISTENT:-default_val}"},
            }
        }
        result = get_mcp_server_config(config)
        self.assertEqual(result["command"], "python")
        self.assertEqual(result["env"]["TEST"], "default_val")


if __name__ == "__main__":
    unittest.main()
