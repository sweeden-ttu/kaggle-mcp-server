"""
Reverse Simulation System Integration

Integrates all components:
- Test-first simulator
- Design feedback analyzer
- Hypothesis tester
- Kaggle notebook generator

Provides a unified interface for the complete workflow.
"""

from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

from .test_first_simulator import (
    TestFirstSimulator, UnitModel, TestCase, TestStatus, ModelType
)
from .design_feedback import DesignFeedbackAnalyzer, FeedbackItem, TopicAnalysis
from .hypothesis_tester import HypothesisTester, Hypothesis, HypothesisStatus
from .kaggle_notebook_generator import KaggleNotebookGenerator
from .logic_layer import LogicEngine


class ReverseSimulationSystem:
    """
    Complete reverse simulation system.
    
    Workflow:
    1. Create unit models with test cases (test-first)
    2. Analyze design and get feedback
    3. Test hypotheses with "getting warmer" loop
    4. Generate Kaggle notebooks for reverse simulation
    5. Propose next steps
    """
    
    def __init__(self, logic_engine: Optional[LogicEngine] = None):
        """Initialize the complete system."""
        self.logic_engine = logic_engine or LogicEngine()
        self.simulator = TestFirstSimulator(self.logic_engine)
        self.feedback_analyzer = DesignFeedbackAnalyzer(self.logic_engine)
        self.hypothesis_tester = HypothesisTester(
            self.logic_engine,
            self.simulator,
            self.feedback_analyzer
        )
        self.notebook_generator = KaggleNotebookGenerator()
    
    def create_model_with_tests(
        self,
        name: str,
        formula: str,
        variables: Dict[str, str],
        test_cases: List[Dict[str, Any]]
    ) -> UnitModel:
        """Create a model with test cases."""
        # Convert test case dicts to TestCase objects
        test_objects = []
        for tc_data in test_cases:
            test_case = TestCase(
                name=tc_data.get("name", "test"),
                input_data=tc_data.get("input", {}),
                expected_output=tc_data.get("expected_output", {}),
                constraints=tc_data.get("constraints", [])
            )
            test_objects.append(test_case)
        
        model = self.simulator.create_unit_model(
            name=name,
            formula=formula,
            variables=variables,
            test_cases=test_objects
        )
        
        return model
    
    def analyze_and_get_feedback(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """Analyze a model and get design feedback."""
        if model_name not in self.simulator.unit_models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.simulator.unit_models[model_name]
        
        # Get feedback
        feedback = self.feedback_analyzer.analyze_model(model, self.simulator)
        
        # Extract topics
        topics = self.feedback_analyzer.extract_topics(model)
        
        # Propose changes
        proposals = self.feedback_analyzer.propose_changes(model, feedback)
        
        return {
            "feedback": [
                {
                    "level": f.level.value,
                    "principle": f.principle.value,
                    "category": f.category,
                    "message": f.message,
                    "suggestion": f.suggestion,
                    "confidence": f.confidence
                }
                for f in feedback
            ],
            "topics": {
                "topics": topics.topics,
                "relationships": topics.relationships,
                "patterns": topics.patterns
            },
            "proposals": proposals
        }
    
    def test_hypothesis_with_feedback(
        self,
        hypothesis_description: str,
        model_name: str,
        user_feedback_callback: Optional[Callable[[str], str]] = None
    ) -> Dict[str, Any]:
        """Test a hypothesis with user feedback loop."""
        if model_name not in self.simulator.unit_models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.simulator.unit_models[model_name]
        
        # Create hypothesis
        hypothesis = self.hypothesis_tester.create_hypothesis(
            description=hypothesis_description,
            model=model
        )
        
        # Set callback
        if user_feedback_callback:
            self.hypothesis_tester.set_user_feedback_callback(user_feedback_callback)
        
        # Test hypothesis
        status, message = self.hypothesis_tester.test_hypothesis(hypothesis.id)
        
        return {
            "hypothesis_id": hypothesis.id,
            "status": status.value,
            "message": message,
            "confidence": hypothesis.confidence
        }
    
    def run_hypothesis_loop(
        self,
        hypothesis_description: str,
        model_name: str,
        max_steps: int = 10,
        user_feedback_callback: Optional[Callable[[str], str]] = None
    ) -> Dict[str, Any]:
        """Run the complete hypothesis testing loop."""
        if model_name not in self.simulator.unit_models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.simulator.unit_models[model_name]
        
        # Create initial hypothesis
        hypothesis = self.hypothesis_tester.create_hypothesis(
            description=hypothesis_description,
            model=model
        )
        
        # Set callback
        if user_feedback_callback:
            self.hypothesis_tester.set_user_feedback_callback(user_feedback_callback)
        
        # Run loop
        final_hypothesis, steps = self.hypothesis_tester.run_hypothesis_loop(
            hypothesis,
            max_steps=max_steps
        )
        
        return {
            "final_hypothesis": {
                "id": final_hypothesis.id,
                "description": final_hypothesis.description,
                "status": final_hypothesis.status.value,
                "confidence": final_hypothesis.confidence
            },
            "steps": len(steps),
            "tree": self.hypothesis_tester.visualize_hypothesis_tree()
        }
    
    def generate_reverse_simulation_notebook(
        self,
        model_name: str,
        observed_outputs: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """Generate a Kaggle notebook for reverse simulation."""
        if model_name not in self.simulator.unit_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.simulator.unit_models[model_name]
        
        notebook = self.notebook_generator.generate_reverse_simulation_notebook(
            model=model,
            observed_outputs=observed_outputs,
            simulator=self.simulator,
            hypothesis_tester=self.hypothesis_tester
        )
        
        return self.notebook_generator.save_notebook(notebook, output_path)
    
    def propose_next_steps(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Propose next steps based on current state."""
        if model_name not in self.simulator.unit_models:
            return [{"error": f"Model {model_name} not found"}]
        
        model = self.simulator.unit_models[model_name]
        
        # Get feedback
        feedback = self.feedback_analyzer.analyze_model(model, self.simulator)
        
        # Get test summary
        test_summary = self.simulator.get_test_summary(model_name)
        
        # Propose changes
        proposals = self.feedback_analyzer.propose_changes(model, feedback)
        
        # Generate next steps
        next_steps = []
        
        # Step 1: Fix errors
        errors = [f for f in feedback if f.level.value == "error"]
        if errors:
            next_steps.append({
                "priority": "high",
                "action": "fix_errors",
                "description": f"Fix {len(errors)} error(s) in the model",
                "details": [e.message for e in errors[:3]]
            })
        
        # Step 2: Improve test coverage
        if test_summary["passed"] / test_summary["total"] < 0.8:
            next_steps.append({
                "priority": "high",
                "action": "improve_tests",
                "description": "Improve test coverage",
                "details": [f"Current pass rate: {test_summary['passed']}/{test_summary['total']}"]
            })
        
        # Step 3: Apply simplifications
        simplifications = [p for p in proposals if p.get("type") == "simplification"]
        if simplifications:
            next_steps.append({
                "priority": "medium",
                "action": "simplify",
                "description": "Apply simplifications",
                "details": [s["description"] for s in simplifications[:3]]
            })
        
        # Step 4: Generate reverse simulation notebook
        next_steps.append({
            "priority": "medium",
            "action": "generate_notebook",
            "description": "Generate Kaggle notebook for reverse simulation",
            "details": ["Create notebook to guess inputs from outputs"]
        })
        
        # Step 5: Test hypotheses
        if self.hypothesis_tester.hypotheses:
            next_steps.append({
                "priority": "low",
                "action": "test_hypotheses",
                "description": "Continue hypothesis testing",
                "details": [f"{len(self.hypothesis_tester.hypotheses)} hypotheses to test"]
            })
        
        return next_steps
    
    def export_system_state(self, output_path: Path) -> Path:
        """Export complete system state."""
        state = {
            "models": {
                name: {
                    "name": model.name,
                    "formula": model.formula,
                    "variables": model.variables,
                    "test_count": len(model.test_cases)
                }
                for name, model in self.simulator.unit_models.items()
            },
            "hypotheses": {
                h.id: {
                    "description": h.description,
                    "status": h.status.value,
                    "confidence": h.confidence
                }
                for h in self.hypothesis_tester.hypotheses.values()
            },
            "test_results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "time": r.execution_time
                }
                for r in self.simulator.test_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        return output_path
