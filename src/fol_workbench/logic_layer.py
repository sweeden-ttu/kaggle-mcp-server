"""Logic Layer: Z3-based FOL validation and model finding engine."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    from z3 import (
        Solver, Bool, Int, Real, String, Function, ForAll, Exists, And, Or, Not,
        Implies, Iff, sat, unsat, unknown, Model, is_true, is_false,
        IntSort, BoolSort, RealSort, StringSort, ArraySort
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Create dummy classes for type hints
    Solver = object
    Model = object


class ValidationResult(Enum):
    """Result of formula validation."""
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a found model."""
    variables: Dict[str, Any]
    interpretation: Dict[str, Any]
    is_complete: bool
    raw_model: Optional[Any] = None


@dataclass
class ValidationInfo:
    """Complete validation information."""
    result: ValidationResult
    model: Optional[ModelInfo] = None
    error_message: Optional[str] = None
    proof: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None


class LogicEngine:
    """Z3-based logic engine for FOL validation and model finding."""
    
    def __init__(self):
        """Initialize the logic engine."""
        if not Z3_AVAILABLE:
            raise ImportError(
                "z3-solver is not installed. Install it with: pip install z3-solver"
            )
        
        self.solver = Solver()
        self.variables: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
        self.constraints: List[Any] = []
    
    def reset(self):
        """Reset the solver and clear all variables and constraints."""
        self.solver = Solver()
        self.variables.clear()
        self.functions.clear()
        self.constraints.clear()
    
    def declare_variable(self, name: str, var_type: str = "Bool") -> Any:
        """
        Declare a variable.
        
        Args:
            name: Variable name
            var_type: Type ("Bool", "Int", "Real", "String")
        
        Returns:
            Z3 variable
        """
        if name in self.variables:
            return self.variables[name]
        
        if var_type == "Bool":
            var = Bool(name)
        elif var_type == "Int":
            var = Int(name)
        elif var_type == "Real":
            var = Real(name)
        elif var_type == "String":
            var = String(name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
        
        self.variables[name] = var
        return var
    
    def declare_function(self, name: str, domain: List[str], codomain: str = "Bool") -> Any:
        """
        Declare a function.
        
        Args:
            name: Function name
            domain: List of domain types
            codomain: Codomain type
        
        Returns:
            Z3 function
        """
        if name in self.functions:
            return self.functions[name]
        
        # Map type strings to Z3 sorts
        sort_map = {
            "Bool": BoolSort(),
            "Int": IntSort(),
            "Real": RealSort(),
            "String": StringSort()
        }
        
        domain_sorts = [sort_map.get(d, IntSort()) for d in domain]
        codomain_sort = sort_map.get(codomain, BoolSort())
        
        func = Function(name, *domain_sorts, codomain_sort)
        self.functions[name] = func
        return func
    
    def parse_formula(self, formula_str: str) -> Optional[Any]:
        """
        Parse a formula string into Z3 expression.
        
        This is a simplified parser. For production, consider using
        a proper parser or SMT-LIB parser.
        
        Args:
            formula_str: Formula as string
        
        Returns:
            Z3 expression or None if parsing fails
        """
        try:
            # Simple evaluation-based parser (use with caution in production)
            # In production, you'd want a proper parser
            safe_dict = {
                'And': And,
                'Or': Or,
                'Not': Not,
                'Implies': Implies,
                'Iff': Iff,
                'ForAll': ForAll,
                'Exists': Exists,
                'True': True,
                'False': False,
            }
            safe_dict.update(self.variables)
            safe_dict.update(self.functions)
            
            # Evaluate the formula
            result = eval(formula_str, {"__builtins__": {}}, safe_dict)
            return result
        except Exception as e:
            return None
    
    def add_constraint(self, constraint: Any):
        """
        Add a constraint to the solver.
        
        Args:
            constraint: Z3 expression
        """
        self.solver.add(constraint)
        self.constraints.append(constraint)
    
    def add_formula(self, formula_str: str) -> Tuple[bool, Optional[str]]:
        """
        Add a formula string as a constraint.
        
        Args:
            formula_str: Formula as string
        
        Returns:
            Tuple of (success, error_message)
        """
        formula = self.parse_formula(formula_str)
        if formula is None:
            return False, f"Failed to parse formula: {formula_str}"
        
        self.add_constraint(formula)
        return True, None
    
    def check_satisfiability(self) -> ValidationInfo:
        """
        Check if the current set of constraints is satisfiable.
        
        Returns:
            ValidationInfo with result and model if satisfiable
        """
        try:
            result = self.solver.check()
            
            if result == sat:
                model = self.solver.model()
                model_info = self._extract_model_info(model)
                return ValidationInfo(
                    result=ValidationResult.SATISFIABLE,
                    model=model_info,
                    statistics=self.solver.statistics()
                )
            elif result == unsat:
                return ValidationInfo(
                    result=ValidationResult.UNSATISFIABLE,
                    statistics=self.solver.statistics()
                )
            else:  # unknown
                return ValidationInfo(
                    result=ValidationResult.UNKNOWN,
                    statistics=self.solver.statistics()
                )
        except Exception as e:
            return ValidationInfo(
                result=ValidationResult.ERROR,
                error_message=str(e)
            )
    
    def find_model(self, formula_str: Optional[str] = None) -> ValidationInfo:
        """
        Find a model for the given formula or current constraints.
        
        Args:
            formula_str: Optional formula string to check
        
        Returns:
            ValidationInfo with model if found
        """
        if formula_str:
            self.reset()
            success, error = self.add_formula(formula_str)
            if not success:
                return ValidationInfo(
                    result=ValidationResult.ERROR,
                    error_message=error
                )
        
        return self.check_satisfiability()
    
    def prove_implication(self, premise: str, conclusion: str) -> ValidationInfo:
        """
        Prove that premise implies conclusion.
        
        This checks if (premise AND NOT conclusion) is unsatisfiable.
        
        Args:
            premise: Premise formula
            conclusion: Conclusion formula
        
        Returns:
            ValidationInfo with result
        """
        self.reset()
        
        # Add premise
        success, error = self.add_formula(premise)
        if not success:
            return ValidationInfo(
                result=ValidationResult.ERROR,
                error_message=f"Failed to parse premise: {error}"
            )
        
        # Add negated conclusion
        conclusion_expr = self.parse_formula(conclusion)
        if conclusion_expr is None:
            return ValidationInfo(
                result=ValidationResult.ERROR,
                error_message=f"Failed to parse conclusion: {conclusion}"
            )
        
        self.add_constraint(Not(conclusion_expr))
        
        # Check satisfiability
        result = self.check_satisfiability()
        
        # If unsatisfiable, the implication holds
        if result.result == ValidationResult.UNSATISFIABLE:
            return ValidationInfo(
                result=ValidationResult.SATISFIABLE,  # Implication is valid
                statistics=result.statistics
            )
        elif result.result == ValidationResult.SATISFIABLE:
            # Found counterexample
            return ValidationInfo(
                result=ValidationResult.SATISFIABLE,  # Counterexample found
                model=result.model,
                statistics=result.statistics
            )
        else:
            return result
    
    def _extract_model_info(self, model: Model) -> ModelInfo:
        """
        Extract model information from Z3 model.
        
        Args:
            model: Z3 model
        
        Returns:
            ModelInfo
        """
        variables = {}
        interpretation = {}
        
        for decl in model:
            name = decl.name()
            value = model[decl]
            
            variables[name] = str(value)
            
            # Try to get actual Python value
            try:
                if value.is_bool():
                    interpretation[name] = is_true(value)
                elif value.is_int():
                    interpretation[name] = value.as_long()
                elif value.is_real():
                    interpretation[name] = float(value.as_decimal(10))
                elif value.is_string():
                    interpretation[name] = value.as_string()
                else:
                    interpretation[name] = str(value)
            except Exception:
                interpretation[name] = str(value)
        
        return ModelInfo(
            variables=variables,
            interpretation=interpretation,
            is_complete=True,
            raw_model=model
        )
    
    def to_smt_lib(self) -> str:
        """
        Convert current constraints to SMT-LIB format.
        
        Returns:
            SMT-LIB string
        """
        return self.solver.to_smt2()
    
    def from_smt_lib(self, smt_content: str) -> Tuple[bool, Optional[str]]:
        """
        Load constraints from SMT-LIB format.
        
        Args:
            smt_content: SMT-LIB content
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Z3 can parse SMT-LIB strings
            # parse_smt2_string returns a tuple: (assertions, decls)
            from z3 import parse_smt2_string
            
            self.reset()
            # Parse SMT-LIB content
            parsed, decls = parse_smt2_string(smt_content, sorts={}, decls={})
            
            # Update functions dictionary with declarations
            for name, decl in decls.items():
                self.functions[name] = decl
            
            # Add all assertions as constraints
            if parsed:
                for assertion in parsed:
                    self.add_constraint(assertion)
            
            return True, None
        except Exception as e:
            return False, str(e)
