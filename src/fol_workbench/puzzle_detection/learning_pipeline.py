"""
Global Learning Pipeline Orchestrator

Orchestrates the complete learning cycle:
assign → iterate → evaluate → test → propose → infer → learn → save → optimize → eliminate → learn → plan → evaluate → experiment → design → performance improve
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    # Keep the module importable in minimal environments (e.g., logic-only usage).
    np = None  # type: ignore

try:
    from .hyperparameter_tuner import HyperparameterTuner, HyperparameterConfig, EpisodeResult
except ImportError:  # pragma: no cover
    HyperparameterTuner = None  # type: ignore
    HyperparameterConfig = Any  # type: ignore
    EpisodeResult = Any  # type: ignore

try:
    from .confidence_evaluator import ConfidenceEvaluator, ImprovementEvidence
except ImportError:  # pragma: no cover
    ConfidenceEvaluator = None  # type: ignore
    ImprovementEvidence = Any  # type: ignore
try:
    from .perceptron_units import PerceptronUnit
    from .bayesian_perceptron_assigner import BayesianPerceptronAssigner
    from .feature_extractor import GridSquareFeatures
except ImportError:  # pragma: no cover
    # These are required for the puzzle-detection learning pipeline, but not for
    # the FOL proof pipeline entrypoint.
    PerceptronUnit = Any  # type: ignore
    BayesianPerceptronAssigner = None  # type: ignore
    GridSquareFeatures = Any  # type: ignore

try:
    from ..prolog_brain import PrologBrain, PrologPerceptronAdapter
    PROLOG_BRAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    PrologBrain = None  # type: ignore
    PrologPerceptronAdapter = None  # type: ignore
    PROLOG_BRAIN_AVAILABLE = False

from ..database import Database, Perceptron
from ..logic_layer import LogicEngine, Z3_AVAILABLE


@dataclass
class PipelineState:
    """State of the learning pipeline."""
    episode_id: int = 0
    current_config: Optional[HyperparameterConfig] = None
    perceptrons: Dict[str, PerceptronUnit] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    last_cleanup: str = field(default_factory=lambda: datetime.now().isoformat())


class LearningPipeline:
    """
    Global learning pipeline orchestrator.
    
    Manages the complete learning cycle with all 16 steps:
    1. Assign, 2. Iterate, 3. Evaluate, 4. Test, 5. Propose, 6. Infer,
    7. Learn, 8. Save, 9. Optimize, 10. Eliminate, 11. Learn, 12. Plan,
    13. Evaluate, 14. Experiment, 15. Design, 16. Performance Improve
    """
    
    def __init__(
        self,
        database: Optional[Database] = None,
        hyperparameter_tuner: Optional[HyperparameterTuner] = None,
        confidence_evaluator: Optional[ConfidenceEvaluator] = None,
        bayesian_assigner: Optional[BayesianPerceptronAssigner] = None,
        prolog_brain: Optional[PrologBrain] = None,
        enable_prolog_integration: bool = True
    ):
        """
        Initialize learning pipeline.
        
        Args:
            database: Database instance for persistence
            hyperparameter_tuner: Hyperparameter tuner instance
            confidence_evaluator: Confidence evaluator instance
            bayesian_assigner: Bayesian perceptron assigner instance
            prolog_brain: PROLOG brain instance (optional)
            enable_prolog_integration: Whether to enable PROLOG brain integration
        """
        self.database = database or Database()

        if hyperparameter_tuner is not None:
            self.hyperparameter_tuner = hyperparameter_tuner
        else:
            if HyperparameterTuner is None:
                self.hyperparameter_tuner = None
            else:
                self.hyperparameter_tuner = HyperparameterTuner()

        if confidence_evaluator is not None:
            self.confidence_evaluator = confidence_evaluator
        else:
            if ConfidenceEvaluator is None:
                self.confidence_evaluator = None
            else:
                self.confidence_evaluator = ConfidenceEvaluator()

        if bayesian_assigner is not None:
            self.bayesian_assigner = bayesian_assigner
        else:
            if BayesianPerceptronAssigner is None:
                self.bayesian_assigner = None
            else:
                self.bayesian_assigner = BayesianPerceptronAssigner()
        
        # Initialize PROLOG brain if available and enabled
        self.prolog_brain = None
        self.enable_prolog_integration = enable_prolog_integration and PROLOG_BRAIN_AVAILABLE
        if self.enable_prolog_integration:
            self.prolog_brain = prolog_brain or PrologBrain()
            self.prolog_adapter = PrologPerceptronAdapter()
        
        self.state = PipelineState()
        self.running = False

    def _mean(self, values: List[float]) -> float:
        """Mean helper that works even when numpy isn't installed."""
        if not values:
            return 0.0
        if np is not None:
            return float(np.mean(values))  # type: ignore[union-attr]
        return float(sum(values) / len(values))
    
    def run_episode(
        self,
        training_data: List[Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, GridSquareFeatures]]],
        validation_data: Optional[List[Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, GridSquareFeatures]]]] = None,
        perceptrons: Optional[Dict[str, PerceptronUnit]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete learning episode.
        
        Args:
            training_data: List of (features, target) tuples for training
            validation_data: Optional validation data for testing
            
        Returns:
            Dictionary with episode results
        """
        if self.hyperparameter_tuner is None or self.confidence_evaluator is None:
            raise ImportError(
                "Puzzle-detection learning pipeline requires numpy/scipy dependencies. "
                "Install requirements to use run_episode(), or use run_fol_pipeline() for logic-only."
            )

        # Allow callers (e.g., PuzzleDetector) to provide the current in-memory perceptrons.
        if perceptrons is not None:
            self.state.perceptrons = dict(perceptrons)

        self.state.episode_id += 1
        episode_id = self.state.episode_id
        
        results = {
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        # Step 1: ASSIGN - Assign hyperparameters for episode
        config = self._step_assign(episode_id)
        results['steps']['assign'] = {'config': config.to_dict()}
        
        # Step 2: ITERATE - Run multiple iterations with static parameters
        iteration_results = self._step_iterate(training_data, config)
        results['steps']['iterate'] = iteration_results
        
        # Step 3: EVALUATE - Measure performance and confidence
        evaluation_results = self._step_evaluate()
        results['steps']['evaluate'] = evaluation_results
        
        # Step 4: TEST - Test perceptrons on validation data
        if validation_data:
            test_results = self._step_test(validation_data)
            results['steps']['test'] = test_results
        else:
            results['steps']['test'] = {'skipped': True}
        
        # Step 5: PROPOSE - Propose improvements based on results
        proposals = self._step_propose(evaluation_results)
        results['steps']['propose'] = proposals
        
        # Step 6: INFER - Use Bayesian inference for assignments
        inference_results = self._step_infer()
        results['steps']['infer'] = inference_results
        
        # Step 7: LEARN - Update perceptron weights and Bayesian priors
        learn_results = self._step_learn(training_data)
        results['steps']['learn'] = learn_results
        
        # Step 8: SAVE - Persist high-confidence perceptrons to database
        save_results = self._step_save()
        results['steps']['save'] = save_results
        
        # Step 9: OPTIMIZE - Optimize hyperparameters for next episode
        optimized_config = self._step_optimize()
        results['steps']['optimize'] = {'next_config': optimized_config.to_dict()}
        
        # Step 10: ELIMINATE - Remove low-accuracy perceptrons (after 3 days)
        elimination_results = self._step_eliminate()
        results['steps']['eliminate'] = elimination_results
        
        # Step 11: LEARN - Continue learning from remaining high-quality perceptrons
        continue_learn_results = self._step_learn_continue()
        results['steps']['learn_continue'] = continue_learn_results
        
        # Step 12: PLAN - Plan next episode configuration
        plan_results = self._step_plan()
        results['steps']['plan'] = plan_results
        
        # Step 13: EVALUATE - Evaluate overall system performance
        overall_evaluation = self._step_evaluate_overall()
        results['steps']['evaluate_overall'] = overall_evaluation
        
        # Step 14: EXPERIMENT - Try new configurations
        experiment_results = self._step_experiment()
        results['steps']['experiment'] = experiment_results
        
        # Step 15: DESIGN - Design new perceptron architectures if needed
        design_results = self._step_design()
        results['steps']['design'] = design_results
        
        # Step 16: PERFORMANCE IMPROVE - Measure and track improvements
        improvement_results = self._step_performance_improve()
        results['steps']['performance_improve'] = improvement_results
        
        # Store episode result
        episode_result = EpisodeResult(
            episode_id=episode_id,
            config=config,
            average_accuracy=evaluation_results.get('average_accuracy', 0.0),
            average_confidence=evaluation_results.get('average_confidence', 0.0),
            performance_metrics=overall_evaluation
        )
        self.hyperparameter_tuner.record_episode_result(episode_result)
        
        results['summary'] = {
            'average_accuracy': evaluation_results.get('average_accuracy', 0.0),
            'average_confidence': evaluation_results.get('average_confidence', 0.0),
            'perceptrons_saved': save_results.get('count', 0),
            'perceptrons_eliminated': elimination_results.get('count', 0),
            'improvement_evidence': improvement_results.get('evidence', {})
        }
        
        return results

    # ---------------------------------------------------------------------
    # FOL Proof / Reasoning Pipeline (requested step order)
    # ---------------------------------------------------------------------

    @dataclass
    class FOLPipelineResult:
        """
        Result of a proof-oriented FOL pipeline run.

        This intentionally mirrors the requested steps:
        - eliminate →/↔, push ¬ inward, standardize variables
        - prenex
        - skolemize
        - CNF
        - resolution/unification
        - Horn/SLD
        - Herbrand (+ optional Prolog emission)
        - render
        """

        original: str
        nnf_smt2: Optional[str] = None
        pnf_smt2: Optional[str] = None
        skolem_snf_smt2: Optional[str] = None
        cnf_smt2: Optional[str] = None
        resolution: Dict[str, Any] = field(default_factory=dict)
        horn_sld: Dict[str, Any] = field(default_factory=dict)
        herbrand: Dict[str, Any] = field(default_factory=dict)
        render: Dict[str, str] = field(default_factory=dict)
        errors: List[str] = field(default_factory=list)

    def run_fol_pipeline(
        self,
        formula_str: str,
        render_format: str = "markdown",
        max_resolution_steps: int = 200,
        herbrand_max_depth: int = 2,
        herbrand_max_terms: int = 200,
    ) -> Dict[str, Any]:
        """
        Run a proof-oriented First-Order Logic pipeline in the exact order requested.

        Input:
          - formula_str: Python/Z3-style (And/Or/Not/Implies/Iff/ForAll/Exists) or SMT-LIB.

        Output:
          - Dict with step artifacts, plus a rendered outermost visualization.

        Notes:
          - Uses Z3-backed transformations when available (`LogicEngine`).
          - Resolution/SLD/Herbrand steps are "best-effort" and intentionally bounded.
        """
        out = self.FOLPipelineResult(original=formula_str)

        if not Z3_AVAILABLE:
            out.errors.append("Z3 is not available; cannot run prenex/skolem/CNF pipeline.")
            return {"error": "Z3 not available", "result": out.__dict__}

        engine = LogicEngine()
        expr = engine.parse_formula(formula_str)
        if expr is None:
            out.errors.append("Failed to parse formula.")
            return {"error": "parse_failed", "result": out.__dict__}

        # 1) Eliminate →/↔, push ¬ inward (NNF), standardize variables uniquely.
        #    - Implication / iff elimination + De Morgan + quantifier flipping is handled by _to_nnf_equiv.
        #    - Variable standardization is handled via α-renaming during prenexing; we keep NNF here as evidence.
        simplified = engine._apply_tactics_to_single_expr(expr, ["simplify"])
        simplified = engine._normalize_quantifiers(simplified)
        nnf_expr = engine._to_nnf_equiv(simplified)
        out.nnf_smt2 = nnf_expr.sexpr()

        # 2) Prenex: move quantifiers left; keep a quantifier-free matrix.
        pnf_expr = engine._to_pnf(nnf_expr)
        out.pnf_smt2 = pnf_expr.sexpr()

        # 3) Skolemize ∃ (SNF): constants if no ∀ in scope, functions of in-scope ∀ otherwise.
        #    Z3's SNF tactic performs the standard Skolemization transformation.
        snf_expr = engine._apply_tactics_to_single_expr(expr, ["simplify", "nnf", "snf"])
        out.skolem_snf_smt2 = snf_expr.sexpr()

        # 4) CNF (when needed): distribute ∨ over ∧ (Tseitin is equisatisfiable and resolution-ready).
        cnf_expr = engine._apply_tactics_to_single_expr(snf_expr, ["simplify", "tseitin-cnf"])
        out.cnf_smt2 = cnf_expr.sexpr()

        # 5) Resolution with unification; guard occurs-check pitfalls.
        out.resolution = self._fol_resolution_best_effort(
            engine=engine,
            cnf_expr=cnf_expr,
            step_limit=max_resolution_steps,
        )

        # 6) Horn / SLD: detect Hornness and emit Prolog-ish rules (leftmost, DFS/backtracking).
        out.horn_sld = self._horn_sld_summary(engine=engine, cnf_expr=cnf_expr)

        # 7) Herbrand: build universe/base (bounded), minimal-model perspective; optional Prolog emission.
        out.herbrand = self._herbrand_summary(
            engine=engine,
            cnf_expr=cnf_expr,
            max_depth=herbrand_max_depth,
            max_terms=herbrand_max_terms,
        )

        # 8) Render outermost visualization.
        out.render = self._render_pipeline(out, render_format=render_format)

        return {"result": out.__dict__}

    # ------------------------- FOL helpers (internal) -------------------------

    def _is_literal(self, e: Any) -> bool:
        """Z3 literal = atom or Not(atom), where atom is not And/Or/Implies/quantifier."""
        from z3 import is_not, is_and, is_or, is_quantifier, is_implies  # type: ignore
        if is_quantifier(e) or is_implies(e) or is_and(e) or is_or(e):
            return False
        if is_not(e):
            a = e.arg(0)
            return not (is_and(a) or is_or(a) or is_implies(a) or is_quantifier(a))
        return True

    def _split_cnf(self, cnf_expr: Any) -> List[List[Any]]:
        """
        Convert a Z3 CNF-ish expression into a clause list:
          - [[lit1, lit2], [lit3], ...] representing ∧ of ∨ clauses.
        """
        from z3 import is_and, is_or  # type: ignore
        clauses: List[List[Any]] = []

        def as_clause(e: Any) -> List[Any]:
            if is_or(e):
                return [e.arg(i) for i in range(e.num_args())]
            return [e]

        if is_and(cnf_expr):
            for i in range(cnf_expr.num_args()):
                clauses.append(as_clause(cnf_expr.arg(i)))
        else:
            clauses.append(as_clause(cnf_expr))

        # Filter obvious non-literals defensively.
        cleaned: List[List[Any]] = []
        for c in clauses:
            lits = [l for l in c if self._is_literal(l)]
            cleaned.append(lits)
        return cleaned

    def _lit_polarity_and_atom(self, lit: Any) -> Tuple[bool, Any]:
        """Return (is_positive, atom) for a literal."""
        from z3 import is_not  # type: ignore
        if is_not(lit):
            return False, lit.arg(0)
        return True, lit

    def _collect_bound_var_names(self, engine: LogicEngine, expr: Any) -> set:
        """Collect bound variable names from leading quantifiers (best-effort)."""
        from z3 import is_quantifier  # type: ignore
        names = set()
        e = expr
        while is_quantifier(e):
            vars_open, body_open, _ = engine._open_quantifier(e)
            for v in vars_open:
                try:
                    names.add(v.decl().name())
                except Exception:
                    pass
            e = body_open
        return names

    def _occurs_in(self, var: Any, term: Any) -> bool:
        """Occurs check: does var appear anywhere inside term?"""
        if term.eq(var):
            return True
        if hasattr(term, "num_args") and term.num_args() > 0:
            for i in range(term.num_args()):
                if self._occurs_in(var, term.arg(i)):
                    return True
        return False

    def _unify_terms(self, a: Any, b: Any, subs: Dict[str, Any], var_names: set) -> Optional[Dict[str, Any]]:
        """
        Best-effort first-order unification on Z3 terms.
        Variables are identified by being 0-arity Consts whose name is in var_names.
        """
        from z3 import substitute  # type: ignore

        def is_var(t: Any) -> bool:
            try:
                return hasattr(t, "num_args") and t.num_args() == 0 and t.decl().name() in var_names
            except Exception:
                return False

        # Apply current substitutions
        if subs:
            pairs = []
            for k, v in subs.items():
                try:
                    # Recreate a Const-ish key by name using the existing term's sort
                    # (we only ever call this with real Consts, so this is safe-ish).
                    pass
                except Exception:
                    pass

        # Structural unify
        if a.eq(b):
            return subs

        if is_var(a):
            name = a.decl().name()
            if name in subs:
                return self._unify_terms(subs[name], b, subs, var_names)
            if self._occurs_in(a, b):
                return None
            subs[name] = b
            return subs

        if is_var(b):
            name = b.decl().name()
            if name in subs:
                return self._unify_terms(a, subs[name], subs, var_names)
            if self._occurs_in(b, a):
                return None
            subs[name] = a
            return subs

        # Function application unify
        try:
            if a.decl() != b.decl() or a.num_args() != b.num_args():
                return None
        except Exception:
            return None

        for i in range(a.num_args()):
            subs = self._unify_terms(a.arg(i), b.arg(i), subs, var_names)
            if subs is None:
                return None
        return subs

    def _unify_atoms(self, a: Any, b: Any, var_names: set) -> Optional[Dict[str, Any]]:
        """Unify two predicate atoms (same functor) returning substitutions, else None."""
        try:
            if a.decl() != b.decl() or a.num_args() != b.num_args():
                return None
        except Exception:
            return None
        subs: Dict[str, Any] = {}
        for i in range(a.num_args()):
            subs = self._unify_terms(a.arg(i), b.arg(i), subs, var_names)
            if subs is None:
                return None
        return subs

    def _apply_subs_to_expr(self, expr: Any, subs: Dict[str, Any]) -> Any:
        """Apply substitutions {var_name -> term} to a Z3 expr (best-effort)."""
        from z3 import substitute, Const  # type: ignore
        pairs = []
        for name, term in subs.items():
            try:
                pairs.append((Const(name, term.sort()), term))
            except Exception:
                continue
        if not pairs:
            return expr
        return substitute(expr, *pairs)

    def _fol_resolution_best_effort(self, engine: LogicEngine, cnf_expr: Any, step_limit: int) -> Dict[str, Any]:
        """
        Attempt first-order resolution on a CNF expression.
        Returns a trace and whether an empty clause was derived.
        """
        clauses = self._split_cnf(cnf_expr)
        var_names = self._collect_bound_var_names(engine, cnf_expr)

        # If there are no quantifiers left, var_names may be empty; resolution still works propositionally.
        derived: List[List[Any]] = [c[:] for c in clauses]
        trace: List[Dict[str, Any]] = []

        def clause_key(c: List[Any]) -> Tuple[str, ...]:
            return tuple(sorted([str(l) for l in c]))

        seen = {clause_key(c) for c in derived}

        for step in range(step_limit):
            progressed = False
            n = len(derived)
            for i in range(n):
                for j in range(i + 1, n):
                    c1, c2 = derived[i], derived[j]
                    for lit1 in c1:
                        p1, a1 = self._lit_polarity_and_atom(lit1)
                        for lit2 in c2:
                            p2, a2 = self._lit_polarity_and_atom(lit2)
                            if p1 == p2:
                                continue  # need complementary polarity

                            subs = self._unify_atoms(a1, a2, var_names)
                            if subs is None:
                                continue

                            # Build resolvent: (c1 \ {lit1}) ∪ (c2 \ {lit2}), then apply subs.
                            new_clause = [l for l in c1 if not l.eq(lit1)] + [l for l in c2 if not l.eq(lit2)]
                            new_clause = [self._apply_subs_to_expr(l, subs) for l in new_clause]

                            # Remove duplicate literals (string-based)
                            uniq = []
                            seen_l = set()
                            for l in new_clause:
                                s = str(l)
                                if s not in seen_l:
                                    uniq.append(l)
                                    seen_l.add(s)
                            new_clause = uniq

                            ck = clause_key(new_clause)
                            if ck in seen:
                                continue
                            seen.add(ck)
                            derived.append(new_clause)
                            progressed = True

                            trace.append({
                                "step": step,
                                "from": [i, j],
                                "resolved": {"lit1": str(lit1), "lit2": str(lit2)},
                                "subs": {k: str(v) for k, v in subs.items()},
                                "resolvent": [str(l) for l in new_clause],
                            })

                            if len(new_clause) == 0:
                                return {
                                    "status": "unsat_via_resolution",
                                    "empty_clause_derived": True,
                                    "steps": trace,
                                    "clause_count": len(derived),
                                }
                    if progressed and step % 5 == 0:
                        # modest early exit opportunities
                        pass
            if not progressed:
                break

        return {
            "status": "no_empty_clause_within_limit",
            "empty_clause_derived": False,
            "steps": trace,
            "clause_count": len(derived),
        }

    def _horn_sld_summary(self, engine: LogicEngine, cnf_expr: Any) -> Dict[str, Any]:
        """
        Determine whether the clause set is Horn and, if so, emit Prolog-ish clauses.
        """
        from z3 import is_not  # type: ignore
        clauses = self._split_cnf(cnf_expr)

        horn = True
        prolog_rules: List[str] = []
        for c in clauses:
            positives = []
            negatives = []
            for lit in c:
                if is_not(lit):
                    negatives.append(lit.arg(0))
                else:
                    positives.append(lit)
            if len(positives) > 1:
                horn = False
                break

            # Prolog form: Head :- Body.
            # CNF Horn clauses correspond to (¬b1 ∨ ... ∨ ¬bn ∨ h) == (b1 ∧ ... ∧ bn) -> h
            head = positives[0] if positives else None
            body = negatives
            if head is None:
                # Goal clause: false :- b1, ..., bn.
                if body:
                    prolog_rules.append(f"false :- {', '.join([self._to_prolog_atom(b) for b in body])}.")
                else:
                    prolog_rules.append("false.")
            else:
                if body:
                    prolog_rules.append(f"{self._to_prolog_atom(head)} :- {', '.join([self._to_prolog_atom(b) for b in body])}.")
                else:
                    prolog_rules.append(f"{self._to_prolog_atom(head)}.")

        return {
            "is_horn": horn,
            "sld_strategy": {
                "selection": "leftmost",
                "search": "depth_first",
                "backtracking": True,
            },
            "prolog_emission": prolog_rules if horn else [],
        }

    def _to_prolog_atom(self, atom: Any) -> str:
        """Best-effort rendering of a Z3 predicate atom to a Prolog-ish functor(args)."""
        try:
            name = atom.decl().name()
            if atom.num_args() == 0:
                return name.lower()
            args = []
            for i in range(atom.num_args()):
                args.append(str(atom.arg(i)))
            return f"{name.lower()}({', '.join(args)})"
        except Exception:
            return str(atom)

    def _herbrand_summary(self, engine: LogicEngine, cnf_expr: Any, max_depth: int, max_terms: int) -> Dict[str, Any]:
        """
        Build a bounded Herbrand universe and (bounded) base for the clause set.
        """
        # Extract constants (0-arity uninterpreted) and function symbols from terms in the CNF.
        constants: set = set()
        functions: Dict[str, int] = {}
        predicates: Dict[str, int] = {}

        def walk_term(t: Any):
            try:
                if hasattr(t, "num_args") and t.num_args() == 0:
                    # Could be a constant/variable; include as constant candidate.
                    constants.add(str(t))
                    return
                if hasattr(t, "decl"):
                    functions[str(t.decl().name())] = int(t.num_args())
                if hasattr(t, "num_args"):
                    for i in range(t.num_args()):
                        walk_term(t.arg(i))
            except Exception:
                return

        clauses = self._split_cnf(cnf_expr)
        for c in clauses:
            for lit in c:
                _, atom = self._lit_polarity_and_atom(lit)
                try:
                    predicates[str(atom.decl().name())] = int(atom.num_args())
                    for i in range(atom.num_args()):
                        walk_term(atom.arg(i))
                except Exception:
                    continue

        # Ensure at least one constant (Herbrand universe is non-empty).
        if not constants:
            constants.add("c0")

        # Build universe terms up to max_depth (very bounded).
        universe = sorted(list(constants))
        if functions:
            current = list(universe)
            for _d in range(max_depth):
                next_terms: List[str] = []
                for f, arity in functions.items():
                    if arity <= 0:
                        continue
                    # Cartesian product (bounded)
                    pools = [current] * arity
                    def rec_build(idx: int, acc: List[str]):
                        if len(universe) + len(next_terms) >= max_terms:
                            return
                        if idx == arity:
                            next_terms.append(f"{f}({', '.join(acc)})")
                            return
                        for v in pools[idx]:
                            rec_build(idx + 1, acc + [v])
                    rec_build(0, [])
                # Dedup + append
                for t in next_terms:
                    if t not in universe:
                        universe.append(t)
                        if len(universe) >= max_terms:
                            break
                current = list(universe)
                if len(universe) >= max_terms:
                    break

        # Build a bounded Herbrand base: predicate applied to universe terms (bounded).
        base: List[str] = []
        for p, arity in predicates.items():
            if arity == 0:
                base.append(f"{p}()")
                continue
            pools = [universe] * arity
            def rec_atoms(idx: int, acc: List[str]):
                if len(base) >= max_terms:
                    return
                if idx == arity:
                    base.append(f"{p}({', '.join(acc)})")
                    return
                for v in pools[idx]:
                    rec_atoms(idx + 1, acc + [v])
                    if len(base) >= max_terms:
                        break
            rec_atoms(0, [])
            if len(base) >= max_terms:
                break

        return {
            "herbrand_universe": universe[:max_terms],
            "herbrand_base": base[:max_terms],
            "bounds": {
                "max_depth": max_depth,
                "max_terms": max_terms,
            },
        }

    def _render_pipeline(self, res: "LearningPipeline.FOLPipelineResult", render_format: str) -> Dict[str, str]:
        """
        Render an outermost visualization of the pipeline artifacts.
        Supported: markdown | mermaid | mathjax | latex | raw_html (best-effort).
        """
        fmt = (render_format or "markdown").lower()

        if fmt == "mermaid":
            status = res.resolution.get("status", "unknown")
            empty = res.resolution.get("empty_clause_derived", False)
            horn = res.horn_sld.get("is_horn", False)
            u_sz = len(res.herbrand.get("herbrand_universe") or [])
            b_sz = len(res.herbrand.get("herbrand_base") or [])
            diagram = "\n".join([
                "flowchart TD",
                "  A[Eliminate →/↔, push ¬ inward<br/>NNF] --> B[Prenex: quantifiers left<br/>PNF]",
                "  B --> C[Skolemize ∃<br/>SNF]",
                "  C --> D[CNF (Tseitin)]",
                f"  D --> E[Resolution/Unification<br/>{status}<br/>empty_clause={empty}]",
                f"  D --> F[Horn/SLD<br/>is_horn={horn}]",
                f"  D --> G[Herbrand<br/>|U|={u_sz}, |B|={b_sz}]",
            ])
            return {"mermaid": diagram}

        if fmt == "mathjax":
            # Minimal MathJax-friendly wrapper; content is still SMT2 strings.
            return {
                "mathjax": "\n".join([
                    r"\[\textbf{FOL Pipeline Output}\]",
                    r"\[\textbf{NNF (SMT2)}:\ \texttt{" + (res.nnf_smt2 or "") + r"}\]",
                    r"\[\textbf{PNF (SMT2)}:\ \texttt{" + (res.pnf_smt2 or "") + r"}\]",
                    r"\[\textbf{SNF (SMT2)}:\ \texttt{" + (res.skolem_snf_smt2 or "") + r"}\]",
                    r"\[\textbf{CNF (SMT2)}:\ \texttt{" + (res.cnf_smt2 or "") + r"}\]",
                ])
            }

        if fmt == "latex":
            return {
                "latex": "\n".join([
                    r"\textbf{FOL Pipeline Output}\\",
                    r"\textbf{Original:} " + res.original.replace("_", r"\_") + r"\\",
                    r"\textbf{NNF (SMT2):} \texttt{" + (res.nnf_smt2 or "").replace("_", r"\_") + r"}\\",
                    r"\textbf{PNF (SMT2):} \texttt{" + (res.pnf_smt2 or "").replace("_", r"\_") + r"}\\",
                    r"\textbf{SNF (SMT2):} \texttt{" + (res.skolem_snf_smt2 or "").replace("_", r"\_") + r"}\\",
                    r"\textbf{CNF (SMT2):} \texttt{" + (res.cnf_smt2 or "").replace("_", r"\_") + r"}\\",
                ])
            }

        if fmt == "raw_html":
            return {
                "raw_html": "\n".join([
                    "<h3>FOL Pipeline Output</h3>",
                    f"<p><b>Original</b>: <code>{res.original}</code></p>",
                    f"<p><b>NNF (SMT2)</b>: <code>{res.nnf_smt2 or ''}</code></p>",
                    f"<p><b>PNF (SMT2)</b>: <code>{res.pnf_smt2 or ''}</code></p>",
                    f"<p><b>SNF (SMT2)</b>: <code>{res.skolem_snf_smt2 or ''}</code></p>",
                    f"<p><b>CNF (SMT2)</b>: <code>{res.cnf_smt2 or ''}</code></p>",
                ])
            }

        # Default markdown render (with a light “table/tree/proof” feel).
        md = []
        md.append("### FOL Learning/Proof Pipeline")
        md.append("")
        md.append("- **Step 1 (NNF)**: eliminate →/↔, push ¬ inward")
        md.append(f"  - `nnf_smt2`: `{(res.nnf_smt2 or '')[:300]}{'…' if res.nnf_smt2 and len(res.nnf_smt2) > 300 else ''}`")
        md.append("- **Step 2 (Prenex)**: move quantifiers left")
        md.append(f"  - `pnf_smt2`: `{(res.pnf_smt2 or '')[:300]}{'…' if res.pnf_smt2 and len(res.pnf_smt2) > 300 else ''}`")
        md.append("- **Step 3 (Skolem/SNF)**: skolemize ∃")
        md.append(f"  - `skolem_snf_smt2`: `{(res.skolem_snf_smt2 or '')[:300]}{'…' if res.skolem_snf_smt2 and len(res.skolem_snf_smt2) > 300 else ''}`")
        md.append("- **Step 4 (CNF)**: resolution-ready CNF (Tseitin)")
        md.append(f"  - `cnf_smt2`: `{(res.cnf_smt2 or '')[:300]}{'…' if res.cnf_smt2 and len(res.cnf_smt2) > 300 else ''}`")
        md.append("- **Step 5 (Resolution)**")
        md.append(f"  - status: `{res.resolution.get('status')}`; empty_clause: `{res.resolution.get('empty_clause_derived')}`; clauses: `{res.resolution.get('clause_count')}`")
        md.append("- **Step 6 (Horn/SLD)**")
        md.append(f"  - is_horn: `{res.horn_sld.get('is_horn')}`; prolog_rules: `{len(res.horn_sld.get('prolog_emission') or [])}`")
        md.append("- **Step 7 (Herbrand)**")
        md.append(f"  - universe_size: `{len(res.herbrand.get('herbrand_universe') or [])}`; base_size: `{len(res.herbrand.get('herbrand_base') or [])}`")
        if res.errors:
            md.append("")
            md.append("- **Warnings**:")
            for e in res.errors:
                md.append(f"  - {e}")

        return {"markdown": "\n".join(md)}

    def _unpack_example(
        self,
        example: Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, GridSquareFeatures]]
    ) -> Tuple[np.ndarray, int, Optional[GridSquareFeatures]]:
        """Unpack a training example with optional GridSquareFeatures."""
        if len(example) == 2:
            features, target = example
            return features, int(target), None
        features, target, grid_features = example
        return features, int(target), grid_features
    
    def _step_assign(self, episode_id: int) -> HyperparameterConfig:
        """Step 1: Assign hyperparameters for episode."""
        config = self.hyperparameter_tuner.assign_hyperparameters(episode_id)
        self.state.current_config = config
        return config
    
    def _step_iterate(
        self,
        training_data: List[Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, GridSquareFeatures]]],
        config: HyperparameterConfig
    ) -> Dict[str, Any]:
        """Step 2: Run multiple iterations with static parameters."""
        # Ensure we have perceptrons
        if not self.state.perceptrons:
            # Create initial perceptrons if needed
            if training_data:
                input_size = len(training_data[0][0])
                self.state.perceptrons['default'] = PerceptronUnit(
                    input_size=input_size,
                    learning_rate=config.learning_rate
                )
        
        # Train for specified iterations
        total_correct = 0
        total_samples = 0
        assignments_used = 0
        
        for iteration in range(config.iterations):
            for example in training_data[: config.batch_size * 10]:  # Limit samples per iteration
                features, target, grid_features = self._unpack_example(example)

                # Assign to a perceptron (Bayesian when grid features are available)
                perceptron = list(self.state.perceptrons.values())[0]
                assignment = None
                if grid_features is not None:
                    assignment = self.bayesian_assigner.assign_perceptron(
                        grid_features=grid_features,
                        available_perceptrons=self.state.perceptrons
                    )
                    if assignment.perceptron_id and assignment.perceptron_id in self.state.perceptrons:
                        perceptron = self.state.perceptrons[assignment.perceptron_id]
                    assignments_used += 1

                perceptron.update_learning_rate(config.learning_rate)
                
                is_correct = perceptron.train(features, target)
                if is_correct:
                    total_correct += 1
                total_samples += 1

                # Feed results back to Bayesian assigner so it can update priors.
                if assignment is not None:
                    self.bayesian_assigner.update_from_results(
                        assignment=assignment,
                        was_correct=is_correct,
                        perceptron_confidence=float(perceptron.confidence)
                    )
        
        return {
            'iterations': config.iterations,
            'accuracy': total_correct / max(1, total_samples),
            'samples_processed': total_samples,
            'assignments_used': assignments_used
        }
    
    def _step_evaluate(self) -> Dict[str, Any]:
        """Step 3: Measure performance and confidence."""
        if not self.state.perceptrons:
            return {'average_accuracy': 0.0, 'average_confidence': 0.0}
        
        accuracies = [p.accuracy for p in self.state.perceptrons.values()]
        confidences = [p.confidence for p in self.state.perceptrons.values()]
        
        avg_accuracy = self._mean([float(x) for x in accuracies]) if accuracies else 0.0
        avg_confidence = self._mean([float(x) for x in confidences]) if confidences else 0.0
        
        # Record in confidence evaluator
        self.confidence_evaluator.record_performance(avg_accuracy, avg_confidence)
        
        return {
            'average_accuracy': avg_accuracy,
            'average_confidence': avg_confidence,
            'perceptron_count': len(self.state.perceptrons)
        }
    
    def _step_test(
        self,
        validation_data: List[Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, GridSquareFeatures]]]
    ) -> Dict[str, Any]:
        """Step 4: Test perceptrons on validation data."""
        if not self.state.perceptrons or not validation_data:
            return {'skipped': True}
        
        total_correct = 0
        total_samples = len(validation_data)
        
        for example in validation_data:
            features, target, grid_features = self._unpack_example(example)
            perceptron = list(self.state.perceptrons.values())[0]
            if grid_features is not None:
                assignment = self.bayesian_assigner.assign_perceptron(grid_features, self.state.perceptrons)
                if assignment.perceptron_id and assignment.perceptron_id in self.state.perceptrons:
                    perceptron = self.state.perceptrons[assignment.perceptron_id]
            prediction, _ = perceptron.predict(features)
            predicted_value = 1 if prediction else 0
            
            if predicted_value == target:
                total_correct += 1
        
        test_accuracy = total_correct / max(1, total_samples)
        
        return {
            'test_accuracy': test_accuracy,
            'samples_tested': total_samples,
            'correct_predictions': total_correct
        }
    
    def _step_propose(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Propose improvements based on results."""
        proposals = []
        
        avg_accuracy = evaluation_results.get('average_accuracy', 0.0)
        avg_confidence = evaluation_results.get('average_confidence', 0.0)
        
        if avg_accuracy < 0.7:
            proposals.append({
                'type': 'increase_learning_rate',
                'reason': 'Low accuracy detected',
                'suggestion': 'Try higher learning rate'
            })
        
        if avg_confidence < 0.5:
            proposals.append({
                'type': 'more_training',
                'reason': 'Low confidence detected',
                'suggestion': 'Increase training iterations'
            })
        
        return {'proposals': proposals, 'count': len(proposals)}
    
    def _step_infer(self) -> Dict[str, Any]:
        """Step 6: Use Bayesian inference for assignments."""
        # This would use BayesianPerceptronAssigner to infer best assignments
        # For now, return basic info
        return {
            'inference_performed': True,
            'assignments_count': len(self.bayesian_assigner.get_assignments())
        }
    
    def _step_learn(
        self,
        training_data: List[Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, GridSquareFeatures]]]
    ) -> Dict[str, Any]:
        """Step 7: Update perceptron weights and Bayesian priors."""
        if not self.state.perceptrons or not training_data:
            return {'skipped': True}
        
        # Continue training
        total_updates = 0
        for example in training_data[:100]:  # Limit for performance
            features, target, grid_features = self._unpack_example(example)
            perceptron = list(self.state.perceptrons.values())[0]
            assignment = None
            if grid_features is not None:
                assignment = self.bayesian_assigner.assign_perceptron(grid_features, self.state.perceptrons)
                if assignment.perceptron_id and assignment.perceptron_id in self.state.perceptrons:
                    perceptron = self.state.perceptrons[assignment.perceptron_id]
            is_correct = perceptron.train(features, target)
            total_updates += 1
            if assignment is not None:
                self.bayesian_assigner.update_from_results(
                    assignment=assignment,
                    was_correct=is_correct,
                    perceptron_confidence=float(perceptron.confidence)
                )
        
        return {
            'updates_performed': total_updates,
            'perceptrons_trained': len(self.state.perceptrons)
        }
    
    def _step_save(self) -> Dict[str, Any]:
        """Step 8: Persist high-confidence perceptrons to database."""
        if not self.state.perceptrons:
            return {'count': 0}
        
        # Select high-confidence perceptrons using Bayesian assigner (posterior over quality bins).
        selected = self.bayesian_assigner.select_high_confidence_perceptrons(
            self.state.perceptrons,
            threshold=0.8
        )
        
        saved_count = 0
        for perceptron_id, perceptron in selected:
            state = perceptron.get_state()
            
            # If this perceptron already exists, update performance; else create.
            existing = self.database.get_perceptron(perceptron_id)
            if existing is None:
                self.database.create_perceptron(
                    weights=state.weights.tolist(),
                    learning_rate=state.learning_rate,
                    confidence=state.confidence,
                    accuracy=state.accuracy,
                    perceptron_id=perceptron_id,
                    metadata={
                        'training_count': state.training_count,
                        'created': state.created
                    }
                )
            else:
                self.database.update_perceptron_performance(
                    perceptron_id=perceptron_id,
                    accuracy=state.accuracy,
                    confidence=state.confidence
                )
            saved_count += 1
        
        return {'count': saved_count, 'perceptrons': [pid for pid, _ in selected]}
    
    def _step_optimize(self) -> HyperparameterConfig:
        """Step 9: Optimize hyperparameters for next episode."""
        return self.hyperparameter_tuner.optimize()
    
    def _step_eliminate(self) -> Dict[str, Any]:
        """Step 10: Remove low-accuracy perceptrons (after 3 days)."""
        # Cleanup old perceptrons from database
        deleted_count = self.database.cleanup_old_perceptrons(
            accuracy_threshold=0.6,
            age_days=3
        )
        
        # Also remove from in-memory perceptrons if they're low quality
        to_remove = []
        for pid, perceptron in self.state.perceptrons.items():
            if perceptron.accuracy < 0.6 and perceptron.confidence < 0.5:
                # Check age (simplified - would track creation time)
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.state.perceptrons[pid]
        
        return {
            'count': deleted_count + len(to_remove),
            'database_deleted': deleted_count,
            'memory_removed': len(to_remove)
        }
    
    def _step_learn_continue(self) -> Dict[str, Any]:
        """Step 11: Continue learning from remaining high-quality perceptrons."""
        # Load high-confidence perceptrons from database and ensure they exist in memory.
        high_conf_perceptrons = self.database.get_high_confidence_perceptrons(threshold=0.8)

        loaded = 0
        for p in high_conf_perceptrons:
            if p.perceptron_id in self.state.perceptrons:
                continue
            unit = PerceptronUnit(
                input_size=len(p.weights),
                learning_rate=p.learning_rate,
                initial_weights=np.array(p.weights, dtype=float),
                initial_bias=0.0
            )
            unit.accuracy = float(p.accuracy)
            unit.confidence = float(p.confidence)
            self.state.perceptrons[p.perceptron_id] = unit
            loaded += 1
        
        return {
            'high_confidence_perceptrons_available': len(high_conf_perceptrons),
            'loaded_into_memory': loaded,
            'in_memory_perceptrons': len(self.state.perceptrons),
        }
    
    def _step_plan(self) -> Dict[str, Any]:
        """Step 12: Plan next episode configuration."""
        # Analyze history and plan next episode
        next_config = self.hyperparameter_tuner.optimize()
        
        return {
            'next_episode_planned': True,
            'suggested_config': next_config.to_dict()
        }
    
    def _step_evaluate_overall(self) -> Dict[str, Any]:
        """Step 13: Evaluate overall system performance."""
        summary = self.confidence_evaluator.get_performance_summary()
        
        return {
            'overall_accuracy': summary.get('current_accuracy', 0.0),
            'overall_confidence': summary.get('current_confidence', 0.0),
            'confidence_ratio': summary.get('confidence_ratio', 0.0),
            'improvement_ratio': summary.get('improvement_ratio', 0.0),
            'sample_count': summary.get('sample_count', 0)
        }
    
    def _step_experiment(self) -> Dict[str, Any]:
        """Step 14: Try new configurations."""
        # Could try different learning rates, architectures, etc.
        return {
            'experiments_considered': True,
            'note': 'Experiment phase - could try new perceptron architectures'
        }
    
    def _step_design(self) -> Dict[str, Any]:
        """Step 15: Design new perceptron architectures if needed."""
        # Could design new perceptron types based on performance
        return {
            'design_phase': True,
            'note': 'Design phase - could create specialized perceptrons'
        }
    
    def _step_performance_improve(self) -> Dict[str, Any]:
        """Step 16: Measure and track evidence-based improvements."""
        evidence = self.confidence_evaluator.evaluate_improvement()
        
        return {
            'evidence': {
                'confidence_ratio': evidence.confidence_ratio,
                'improvement_ratio': evidence.improvement_ratio,
                'meets_criteria': evidence.meets_criteria,
                'is_significant': evidence.is_significant,
                'statistical_significance': evidence.statistical_significance
            },
            'has_improvement': evidence.meets_criteria
        }
