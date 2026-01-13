"""
Test-First Simulator for Unit Models and Automata Theories

This module implements a test-first approach to model generation and validation,
creating unit models and automata theories that can be tested before implementation.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time
from collections import defaultdict

from .logic_layer import LogicEngine, ValidationResult


class ModelType(Enum):
    """Types of models that can be simulated."""
    UNIT = "unit"
    AUTOMATA = "automata"
    STATE_MACHINE = "state_machine"
    TRANSITION_SYSTEM = "transition_system"


class TestStatus(Enum):
    """Status of a test case."""
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class TestCase:
    """A single test case for model validation."""
    name: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    constraints: List[str] = field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    actual_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class UnitModel:
    """A unit model with test cases."""
    name: str
    model_type: ModelType
    formula: str
    variables: Dict[str, str] = field(default_factory=dict)  # name -> type
    test_cases: List[TestCase] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)


@dataclass
class AutomataTheory:
    """An automata theory with states and transitions."""
    name: str
    states: Set[str] = field(default_factory=set)
    initial_state: Optional[str] = None
    final_states: Set[str] = field(default_factory=set)
    transitions: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, label, to)
    alphabet: Set[str] = field(default_factory=set)
    properties: List[str] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)


class TestFirstSimulator:
    """
    Test-first simulator for creating and validating unit models and automata.
    
    This simulator follows a test-driven development approach:
    1. Define test cases first
    2. Generate models that satisfy the tests
    3. Validate models against properties
    4. Refine based on feedback
    """
    
    def __init__(self, logic_engine: Optional[LogicEngine] = None):
        """Initialize the simulator."""
        self.logic_engine = logic_engine or LogicEngine()
        self.unit_models: Dict[str, UnitModel] = {}
        self.automata_theories: Dict[str, AutomataTheory] = {}
        self.test_results: List[TestCase] = []
    
    def create_unit_model(
        self,
        name: str,
        formula: str,
        variables: Optional[Dict[str, str]] = None,
        test_cases: Optional[List[TestCase]] = None
    ) -> UnitModel:
        """Create a new unit model with test cases."""
        model = UnitModel(
            name=name,
            model_type=ModelType.UNIT,
            formula=formula,
            variables=variables or {},
            test_cases=test_cases or []
        )
        self.unit_models[name] = model
        return model
    
    def create_automata_theory(
        self,
        name: str,
        states: Optional[Set[str]] = None,
        initial_state: Optional[str] = None,
        transitions: Optional[List[Tuple[str, str, str]]] = None,
        alphabet: Optional[Set[str]] = None
    ) -> AutomataTheory:
        """Create a new automata theory."""
        theory = AutomataTheory(
            name=name,
            states=states or set(),
            initial_state=initial_state,
            transitions=transitions or [],
            alphabet=alphabet or set()
        )
        self.automata_theories[name] = theory
        return theory
    
    def add_test_case(
        self,
        model_name: str,
        test_case: TestCase
    ) -> bool:
        """Add a test case to a model."""
        if model_name in self.unit_models:
            self.unit_models[model_name].test_cases.append(test_case)
            return True
        elif model_name in self.automata_theories:
            self.automata_theories[model_name].test_cases.append(test_case)
            return True
        return False
    
    def run_tests(self, model_name: str) -> List[TestCase]:
        """Run all test cases for a model."""
        results = []
        
        if model_name in self.unit_models:
            model = self.unit_models[model_name]
            for test_case in model.test_cases:
                result = self._test_unit_model(model, test_case)
                results.append(result)
        elif model_name in self.automata_theories:
            theory = self.automata_theories[model_name]
            for test_case in theory.test_cases:
                result = self._test_automata(theory, test_case)
                results.append(result)
        
        self.test_results.extend(results)
        return results
    
    def _test_unit_model(self, model: UnitModel, test_case: TestCase) -> TestCase:
        """Test a unit model against a test case."""
        start_time = time.time()
        
        try:
            # Reset logic engine
            self.logic_engine.reset()
            
            # Add model formula
            self.logic_engine.add_formula_with_tracking(model.formula)
            
            # Add constraints from test case
            for constraint in test_case.constraints:
                self.logic_engine.add_formula_with_tracking(constraint)
            
            # Add input constraints
            for var_name, var_value in test_case.input_data.items():
                if var_name in model.variables:
                    var_type = model.variables[var_name]
                    constraint = self._create_constraint(var_name, var_type, var_value)
                    if constraint:
                        self.logic_engine.add_formula_with_tracking(constraint)
            
            # Find model
            result = self.logic_engine.find_model(model.formula)
            
            # Check if satisfiable
            if result.result == ValidationResult.SATISFIABLE:
                # Extract actual output
                actual_output = {}
                if result.model and result.model.interpretation:
                    for key, value in result.model.interpretation.items():
                        actual_output[key] = str(value)
                
                test_case.actual_output = actual_output
                test_case.status = TestStatus.PASSED
                
                # Verify against expected output
                if not self._outputs_match(actual_output, test_case.expected_output):
                    test_case.status = TestStatus.FAILED
                    test_case.error_message = "Output mismatch"
            else:
                test_case.status = TestStatus.FAILED
                test_case.error_message = f"Model unsatisfiable: {result.result.value}"
        
        except Exception as e:
            test_case.status = TestStatus.ERROR
            test_case.error_message = str(e)
        
        test_case.execution_time = time.time() - start_time
        return test_case
    
    def _test_automata(self, theory: AutomataTheory, test_case: TestCase) -> TestCase:
        """Test an automata theory against a test case."""
        start_time = time.time()
        
        try:
            # Simulate automata execution
            current_state = theory.initial_state
            input_sequence = test_case.input_data.get("sequence", [])
            
            for symbol in input_sequence:
                # Find transition
                transition_found = False
                for from_state, label, to_state in theory.transitions:
                    if from_state == current_state and label == symbol:
                        current_state = to_state
                        transition_found = True
                        break
                
                if not transition_found:
                    test_case.status = TestStatus.FAILED
                    test_case.error_message = f"No transition from {current_state} on {symbol}"
                    test_case.execution_time = time.time() - start_time
                    return test_case
            
            # Check final state
            actual_output = {"final_state": current_state}
            test_case.actual_output = actual_output
            
            expected_final = test_case.expected_output.get("final_state")
            if expected_final and current_state == expected_final:
                test_case.status = TestStatus.PASSED
            else:
                test_case.status = TestStatus.FAILED
                test_case.error_message = f"Expected {expected_final}, got {current_state}"
        
        except Exception as e:
            test_case.status = TestStatus.ERROR
            test_case.error_message = str(e)
        
        test_case.execution_time = time.time() - start_time
        return test_case
    
    def _create_constraint(self, var_name: str, var_type: str, value: Any) -> Optional[str]:
        """Create a constraint formula for a variable assignment."""
        if var_type == "Bool":
            bool_val = "True" if value else "False"
            return f"{var_name} == {bool_val}"
        elif var_type == "Int":
            return f"{var_name} == {value}"
        elif var_type == "Real":
            return f"{var_name} == {value}"
        return None
    
    def _outputs_match(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check if actual output matches expected output."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if str(actual[key]) != str(expected_value):
                return False
        return True
    
    def generate_model_from_tests(self, model_name: str) -> Optional[str]:
        """Generate a model formula that satisfies all test cases."""
        if model_name not in self.unit_models:
            return None
        
        model = self.unit_models[model_name]
        if not model.test_cases:
            return None
        
        # Collect constraints from all passing test cases
        constraints = []
        for test_case in model.test_cases:
            if test_case.status == TestStatus.PASSED:
                constraints.extend(test_case.constraints)
        
        # Generate a formula that satisfies all constraints
        if constraints:
            return f"And({', '.join(constraints)})"
        
        return model.formula
    
    def get_test_summary(self, model_name: str) -> Dict[str, Any]:
        """Get a summary of test results for a model."""
        results = self.run_tests(model_name)
        
        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in results if r.status == TestStatus.FAILED),
            "errors": sum(1 for r in results if r.status == TestStatus.ERROR),
            "pending": sum(1 for r in results if r.status == TestStatus.PENDING),
            "average_time": sum(r.execution_time for r in results) / len(results) if results else 0.0,
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "error": r.error_message,
                    "time": r.execution_time
                }
                for r in results
            ]
        }
        
        return summary
    
    def export_model(self, model_name: str, path: Path) -> Path:
        """Export a model to JSON."""
        data = {}
        
        if model_name in self.unit_models:
            model = self.unit_models[model_name]
            data = {
                "type": "unit_model",
                "name": model.name,
                "formula": model.formula,
                "variables": model.variables,
                "test_cases": [
                    {
                        "name": tc.name,
                        "input_data": tc.input_data,
                        "expected_output": tc.expected_output,
                        "constraints": tc.constraints,
                        "status": tc.status.value
                    }
                    for tc in model.test_cases
                ]
            }
        elif model_name in self.automata_theories:
            theory = self.automata_theories[model_name]
            data = {
                "type": "automata",
                "name": theory.name,
                "states": list(theory.states),
                "initial_state": theory.initial_state,
                "final_states": list(theory.final_states),
                "transitions": [
                    {"from": f, "label": l, "to": t}
                    for f, l, t in theory.transitions
                ],
                "alphabet": list(theory.alphabet),
                "test_cases": [
                    {
                        "name": tc.name,
                        "input_data": tc.input_data,
                        "expected_output": tc.expected_output,
                        "status": tc.status.value
                    }
                    for tc in theory.test_cases
                ]
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return path
