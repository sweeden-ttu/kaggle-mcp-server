"""
Logic Layer: Z3-based FOL validation and model finding engine.

This module provides a high-level interface to the Z3 SMT solver for:
- First-Order Logic (FOL) formula validation
- Model finding (satisfying assignments)
- Implication proving
- SMT-LIB format support

The LogicEngine class wraps Z3's solver API to provide a clean, Pythonic
interface for working with logical formulas. It handles variable and function
declarations, formula parsing, constraint management, and model extraction.

Architecture:
- ValidationResult: Enum for satisfiability results (SAT/UNSAT/UNKNOWN/ERROR)
- ModelInfo: Structured representation of Z3 models
- ValidationInfo: Complete validation results with models and statistics
- LogicEngine: Main engine class that orchestrates Z3 operations
"""

from typing import Dict, List, Optional, Tuple, Any
import ast
from dataclasses import dataclass
from enum import Enum

try:
    from z3 import (
        Solver, Bool, Int, Real, String, Function, ForAll, Exists, And, Or, Not,
        Implies, sat, unsat, unknown, Model, is_true, is_false,
        IntSort, BoolSort, RealSort, StringSort, ArraySort,
        BoolVal, IntVal, RealVal, StringVal, parse_smt2_string
    )
    # Iff (biconditional) is implemented as equality in Z3: (x == y) for Bool
    # We'll create a helper function for it
    def Iff(a, b):
        """Biconditional: a ↔ b (equivalent to a == b for Bool)"""
        return a == b
    
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Create dummy classes for type hints
    Solver = object
    Model = object
    def Iff(a, b):
        return None


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
    """
    Z3-based logic engine for FOL validation and model finding.
    
    This class provides a high-level interface to Z3 for working with
    First-Order Logic formulas. It manages:
    - Variable declarations (Bool, Int, Real, String)
    - Function declarations (predicates, functions)
    - Constraint management
    - Formula parsing and evaluation
    - Satisfiability checking
    - Model extraction and interpretation
    
    Example usage:
        engine = LogicEngine()
        engine.add_formula("And(x, Or(y, Not(z)))")
        result = engine.check_satisfiability()
        if result.result == ValidationResult.SATISFIABLE:
            print(f"Model: {result.model.interpretation}")
    """
    
    def __init__(self):
        """
        Initialize the logic engine.
        
        Creates a new Z3 solver instance and initializes empty dictionaries
        for variables, functions, and constraints. Raises ImportError if
        Z3 is not available.
        
        Raises:
            ImportError: If z3-solver package is not installed
        """
        if not Z3_AVAILABLE:
            raise ImportError(
                "z3-solver is not installed. Install it with: pip install z3-solver"
            )
        
        # Z3 solver instance - handles constraint solving
        self.solver = Solver()
        
        # Dictionary mapping variable names to Z3 variable objects
        # Example: {"x": Bool('x'), "y": Int('y')}
        self.variables: Dict[str, Any] = {}
        
        # Dictionary mapping function names to Z3 function declarations
        # Example: {"P": Function('P', IntSort(), BoolSort())}
        self.functions: Dict[str, Any] = {}
        
        # List of all constraints added to the solver
        # Used for tracking and debugging
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
    
    def define_predicate(
        self,
        name: str,
        arity: int = 0,
        symbol_type: str = "Function",
        domain_types: Optional[List[str]] = None,
        codomain: str = "Bool"
    ) -> Any:
        """
        Define a predicate or function symbol dynamically.
        
        This method allows dynamic creation of logic symbols (predicates/functions)
        with specified arity (number of arguments) and types. This is essential for
        First-Order Logic where predicates can have multiple arguments.
        
        In Z3, predicates are functions that return Bool. Constants are functions
        with arity 0.
        
        Args:
            name: Symbol name (e.g., "P", "Q", "father")
            arity: Number of arguments (0 for constants, 1+ for predicates/functions)
            symbol_type: "Constant" (arity=0) or "Function" (arity>0)
            domain_types: List of argument types (defaults to all "Int" if None)
            codomain: Return type ("Bool" for predicates, "Int"/"Real"/"String" for functions)
        
        Returns:
            Z3 function/predicate declaration
        
        Example:
            # Define a unary predicate P(x)
            engine.define_predicate("P", arity=1, domain_types=["Int"])
            
            # Define a binary predicate R(x, y)
            engine.define_predicate("R", arity=2, domain_types=["Int", "Int"])
            
            # Define a constant
            engine.define_predicate("c", arity=0, symbol_type="Constant", codomain="Int")
        """
        # Handle constants (arity 0)
        if arity == 0 or symbol_type == "Constant":
            if codomain == "Bool":
                # Boolean constant - treat as variable
                return self.declare_variable(name, "Bool")
            else:
                # Non-Boolean constant - treat as variable of that type
                return self.declare_variable(name, codomain)
        
        # Handle functions/predicates (arity > 0)
        if domain_types is None:
            # Default to all Int arguments
            domain_types = ["Int"] * arity
        elif len(domain_types) != arity:
            # Pad or truncate to match arity
            if len(domain_types) < arity:
                domain_types = domain_types + ["Int"] * (arity - len(domain_types))
            else:
                domain_types = domain_types[:arity]
        
        # Use declare_function which handles the Z3 function creation
        return self.declare_function(name, domain_types, codomain)
    
    def get_all_formulas(self) -> List[str]:
        """
        Get all formulas that have been added as constraints.
        
        This is useful for checkpoint saving - we can serialize all formulas
        and restore them later by re-applying add_formula for each.
        
        Returns:
            List of formula strings (as they were originally added)
        """
        # Note: Z3 doesn't directly provide a way to get back the original
        # formula strings. We track them in self.constraints, but those are
        # Z3 expressions, not strings. For full checkpoint support, we need
        # to maintain a separate list of formula strings.
        # This method returns empty for now - the UI layer should maintain
        # the formula strings separately.
        return []
    
    def add_formula_with_tracking(self, formula_str: str) -> Tuple[bool, Optional[str]]:
        """
        Add a formula and track the original string for checkpoint restoration.
        
        This is a wrapper around add_formula that also stores the original
        formula string for later restoration.
        
        Args:
            formula_str: Formula as string
        
        Returns:
            Tuple of (success, error_message)
        """
        # Store formula string for checkpoint restoration
        if not hasattr(self, '_formula_strings'):
            self._formula_strings = []
        
        success, error = self.add_formula(formula_str)
        if success:
            self._formula_strings.append(formula_str)
        
        return success, error
    
    def get_tracked_formulas(self) -> List[str]:
        """
        Get all tracked formula strings for checkpoint restoration.
        
        Returns:
            List of formula strings that were added via add_formula_with_tracking
        """
        if hasattr(self, '_formula_strings'):
            return self._formula_strings.copy()
        return []
    
    def restore_from_formulas(self, formulas: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Restore solver state by re-applying a list of formulas.
        
        This is used for checkpoint restoration. It resets the solver and
        re-applies all formulas from the checkpoint.
        
        Args:
            formulas: List of formula strings to restore
        
        Returns:
            Tuple of (success, error_message)
        """
        self.reset()
        if hasattr(self, '_formula_strings'):
            self._formula_strings = []
        
        for formula in formulas:
            success, error = self.add_formula_with_tracking(formula)
            if not success:
                return False, f"Failed to restore formula: {formula}. Error: {error}"
        
        return True, None
    
    def parse_formula(self, formula_str: str) -> Optional[Any]:
        """
        Parse a formula string into a Z3 expression.
        
        The parser supports two syntaxes:
        1) Python/Z3-style expressions (e.g., And(x, Or(y, Not(z))))
        2) SMT-LIB snippets (auto-detected when the string starts with '(')

        The Python-style parser uses the `ast` module to build expressions
        without relying on eval(), avoiding code execution.
        
        Supported operators:
        - And(x, y): Logical AND (conjunction)
        - Or(x, y): Logical OR (disjunction)
        - Not(x): Logical NOT (negation)
        - Implies(x, y): Implication (x → y)
        - Iff(x, y): Biconditional (x ↔ y)
        - ForAll(x, P(x)): Universal quantification (∀x P(x))
        - Exists(x, P(x)): Existential quantification (∃x P(x))
        
        Args:
            formula_str: Formula as string in Python/Z3 syntax
                        Example: "And(x, Or(y, Not(z)))"
        
        Returns:
            Z3 expression object if parsing succeeds, None otherwise
        
        Note:
            Variables are automatically declared on first use as Bool type.
            For other types, use declare_variable() first.
        """
        stripped = formula_str.strip()
        if not stripped:
            return None

        # Try SMT-LIB first if it looks like an S-expression
        if stripped.startswith("("):
            return self._parse_smt2_formula(stripped)

        return self._parse_python_expr(stripped)

    def _parse_smt2_formula(self, smt_source: str) -> Optional[Any]:
        """Parse SMT-LIB content into a Z3 expression."""
        try:
            # Wrap bare expressions so parse_smt2_string can handle them
            smt_payload = smt_source
            if "(assert" not in smt_source:
                smt_payload = f"(assert {smt_source})"

            decls: Dict[str, Any] = {}
            parsed = parse_smt2_string(smt_payload, decls=decls, sorts={})

            # Keep track of declarations for later use
            self.functions.update(decls)

            if not parsed:
                return None
            if isinstance(parsed, list):
                if len(parsed) == 1:
                    return parsed[0]
                return And(*parsed)
            return parsed
        except Exception:
            return None

    def _parse_python_expr(self, formula_str: str) -> Optional[Any]:
        """Parse Python-style logical expressions using the AST module."""
        try:
            tree = ast.parse(formula_str, mode="eval")
        except SyntaxError:
            return None

        context: Dict[str, Any] = {
            "And": And,
            "Or": Or,
            "Not": Not,
            "Implies": Implies,
            "Iff": Iff,
            "ForAll": ForAll,
            "Exists": Exists,
            "True": BoolVal(True),
            "False": BoolVal(False),
        }
        context.update(self.variables)
        context.update(self.functions)

        try:
            return self._eval_ast_node(tree.body, context)
        except ValueError:
            return None

    def _eval_ast_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Recursively evaluate an AST node into a Z3 expression."""
        if isinstance(node, ast.BoolOp):
            values = [self._eval_ast_node(v, context) for v in node.values]
            if isinstance(node.op, ast.And):
                return And(*values)
            if isinstance(node.op, ast.Or):
                return Or(*values)
            raise ValueError("Unsupported boolean operator")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return Not(self._eval_ast_node(node.operand, context))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")

            func_name = node.func.id
            if func_name not in context:
                raise ValueError(f"Unknown symbol: {func_name}")

            func = context[func_name]
            args = [self._eval_ast_node(arg, context) for arg in node.args]
            return func(*args)

        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            # Auto-declare Bool variables on first use to match previous behavior
            var = self.declare_variable(node.id, "Bool")
            context[node.id] = var
            return var

        if isinstance(node, ast.Constant):
            return self._convert_constant(node.value)

        raise ValueError(f"Unsupported syntax: {ast.dump(node)}")

    def _convert_constant(self, value: Any) -> Any:
        """Convert Python literals to Z3 values."""
        if isinstance(value, bool):
            return BoolVal(value)
        if isinstance(value, int):
            return IntVal(value)
        if isinstance(value, float):
            return RealVal(value)
        if isinstance(value, str):
            return StringVal(value)
        raise ValueError(f"Unsupported constant type: {type(value)}")
    
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
        
        This is the core method that queries Z3 to determine if the current
        constraint set (formulas added via add_formula/add_constraint) has
        a satisfying assignment (model).
        
        The method returns one of three results:
        1. SATISFIABLE: A model exists - the constraints are consistent
        2. UNSATISFIABLE: No model exists - the constraints are contradictory
        3. UNKNOWN: Z3 couldn't determine satisfiability (timeout, incomplete theory, etc.)
        
        If satisfiable, the model (variable assignments) is extracted and included
        in the ValidationInfo. Statistics from Z3 (e.g., solving time, decisions)
        are also included for performance analysis.
        
        Returns:
            ValidationInfo containing:
            - result: ValidationResult enum (SAT/UNSAT/UNKNOWN/ERROR)
            - model: ModelInfo with variable assignments (if satisfiable)
            - statistics: Z3 solver statistics (decisions, time, etc.)
            - error_message: Error description (if ERROR result)
        
        Example:
            engine.add_formula("And(x, y)")
            result = engine.check_satisfiability()
            if result.result == ValidationResult.SATISFIABLE:
                print(f"x = {result.model.interpretation['x']}")
                print(f"y = {result.model.interpretation['y']}")
        """
        try:
            # Query Z3 solver - this is where the actual solving happens
            # Z3 uses various algorithms (DPLL, CDCL, etc.) depending on theory
            result = self.solver.check()
            
            if result == sat:
                # Formula is satisfiable - extract the model
                # A model is a concrete assignment of values to variables
                # that makes all constraints true
                model = self.solver.model()
                model_info = self._extract_model_info(model)
                
                return ValidationInfo(
                    result=ValidationResult.SATISFIABLE,
                    model=model_info,  # Contains variable assignments
                    statistics=self.solver.statistics()  # Performance metrics
                )
            elif result == unsat:
                # Formula is unsatisfiable - no model exists
                # This means the constraints are contradictory
                # (e.g., "x" and "Not(x)" both asserted)
                return ValidationInfo(
                    result=ValidationResult.UNSATISFIABLE,
                    statistics=self.solver.statistics()
                )
            else:  # unknown
                # Z3 couldn't determine satisfiability
                # This can happen due to:
                # - Timeout (solver gave up)
                # - Incomplete theory (some theories are undecidable)
                # - Resource limits
                return ValidationInfo(
                    result=ValidationResult.UNKNOWN,
                    statistics=self.solver.statistics()
                )
        except Exception as e:
            # Unexpected error during solving
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
        If unsatisfiable, the implication is valid (no counterexample exists).
        If satisfiable, a counterexample model is returned.
        
        Args:
            premise: Premise formula
            conclusion: Conclusion formula
        
        Returns:
            ValidationInfo with result:
            - UNSATISFIABLE: Implication is valid (negation is unsatisfiable)
            - SATISFIABLE: Counterexample found (model shows premise true, conclusion false)
            - UNKNOWN: Could not determine (timeout, etc.)
            - ERROR: Parsing or other error occurred
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
        
        # Check satisfiability of (premise AND NOT conclusion)
        result = self.check_satisfiability()
        
        # Check for errors first
        if result.result == ValidationResult.ERROR:
            return result
        
        # If unsatisfiable, the implication is valid (no counterexample exists)
        if result.result == ValidationResult.UNSATISFIABLE:
            # The negation (premise AND NOT conclusion) is unsatisfiable,
            # which means the implication (premise → conclusion) is valid
            return ValidationInfo(
                result=ValidationResult.UNSATISFIABLE,  # Implication is valid
                statistics=result.statistics
            )
        elif result.result == ValidationResult.SATISFIABLE:
            # Found counterexample: (premise AND NOT conclusion) is satisfiable
            # The model shows values where premise is true but conclusion is false
            return ValidationInfo(
                result=ValidationResult.SATISFIABLE,  # Counterexample found
                model=result.model,
                statistics=result.statistics
            )
        else:
            # UNKNOWN case - pass through
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
            decls: Dict[str, Any] = {}
            parsed = parse_smt2_string(smt_content, sorts={}, decls=decls)

            # Update functions dictionary with declarations
            if decls:
                self.functions.update(decls)

            # Add all assertions as constraints
            if parsed:
                for assertion in parsed:
                    self.add_constraint(assertion)

            return True, None
        except Exception as e:
            return False, str(e)
