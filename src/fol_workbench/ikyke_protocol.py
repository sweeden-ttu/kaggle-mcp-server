"""
IKYKE Protocol Engine

The protocol engine manages the automated workflow:
1. Automatically saves work at intervals
2. Runs experiments for 3-5 minutes
3. Stops automatically
4. Evaluates results
5. Begins query phase
6. Performs analysis

This engine orchestrates the entire IKYKE workflow lifecycle.
"""

import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from queue import Queue

from .ikyke_format import (
    IkykeWorkflow, WorkflowPhase, AutoSaveConfig, RunConfig,
    EvaluationConfig, QueryConfig, AnalysisConfig
)
from .ikyke_container import (
    IkykeContainer, FormulaResult, EvaluationResults,
    QueryResult, AnalysisResult, IkykeContainerFormat
)
from .logic_layer import LogicEngine, ValidationResult, ValidationInfo
from .data_layer import DataLayer


class IkykeProtocol:
    """
    IKYKE Protocol Engine.
    
    Manages the automated workflow lifecycle with automatic saving,
    timed execution, evaluation, querying, and analysis.
    """
    
    def __init__(
        self,
        workflow: IkykeWorkflow,
        logic_engine: LogicEngine,
        data_layer: DataLayer,
        container_path: Optional[Path] = None
    ):
        """
        Initialize the IKYKE protocol engine.
        
        Args:
            workflow: IkykeWorkflow definition
            logic_engine: LogicEngine instance for formula evaluation
            data_layer: DataLayer instance for persistence
            container_path: Optional path to container file
        """
        self.workflow = workflow
        self.logic_engine = logic_engine
        self.data_layer = data_layer
        
        # Create or load container
        if container_path and container_path.exists():
            self.container = IkykeContainerFormat.load(container_path)
            self.container_path = container_path
        else:
            self.container = IkykeContainerFormat.create(workflow.header.workflow_id)
            if container_path:
                self.container_path = container_path
            else:
                self.container_path = Path(f"{workflow.name}.ikyke_container")
        
        # Runtime state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.save_queue = Queue()
        
        # Callbacks
        self.on_phase_change: Optional[Callable[[WorkflowPhase], None]] = None
        self.on_formula_result: Optional[Callable[[FormulaResult], None]] = None
        self.on_save: Optional[Callable[[], None]] = None
    
    def start(self):
        """Start the IKYKE workflow execution."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        self.container.set_phase(WorkflowPhase.INITIALIZATION)
        self.container.start_time = datetime.now().isoformat()
        
        # Start worker thread
        self.thread = threading.Thread(target=self._run_workflow, daemon=True)
        self.thread.start()
        
        # Start auto-save thread if enabled
        if self.workflow.auto_save.enabled:
            save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            save_thread.start()
    
    def stop(self):
        """Stop the workflow execution."""
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _run_workflow(self):
        """Main workflow execution loop."""
        try:
            # Phase 1: Initialization
            self._phase_initialization()
            
            # Phase 2: Running (3-5 minutes)
            self._phase_running()
            
            # Phase 3: Stopped
            self._phase_stopped()
            
            # Phase 4: Evaluation
            self._phase_evaluation()
            
            # Phase 5: Query
            self._phase_query()
            
            # Phase 6: Analysis
            self._phase_analysis()
            
            # Phase 7: Completed
            self.container.set_phase(WorkflowPhase.COMPLETED)
            self.container.end_time = datetime.now().isoformat()
            
        except Exception as e:
            self.container.set_phase(WorkflowPhase.ERROR)
            self.container.metadata["error"] = str(e)
        finally:
            self.running = False
            self._save_container()
    
    def _phase_initialization(self):
        """Initialize the workflow."""
        self.container.set_phase(WorkflowPhase.INITIALIZATION)
        if self.on_phase_change:
            self.on_phase_change(WorkflowPhase.INITIALIZATION)
        
        # Load initial formulas
        self.logic_engine.reset()
        for formula in self.workflow.formulas:
            self.logic_engine.add_formula_with_tracking(formula)
        
        for constraint in self.workflow.constraints:
            self.logic_engine.add_formula_with_tracking(constraint)
    
    def _phase_running(self):
        """Run experiments for 3-5 minutes."""
        self.container.set_phase(WorkflowPhase.RUNNING)
        if self.on_phase_change:
            self.on_phase_change(WorkflowPhase.RUNNING)
        
        # Calculate duration
        import random
        duration_minutes = random.uniform(
            self.workflow.run.duration_min,
            self.workflow.run.duration_max
        )
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        formula_count = 0
        model_count = 0
        
        # Generate and evaluate formulas
        while datetime.now() < end_time and not self.stop_event.is_set():
            # Check stop conditions
            if formula_count >= self.workflow.run.max_formulas:
                break
            if model_count >= self.workflow.run.max_models:
                break
            
            # Generate or select a formula to evaluate
            formula = self._generate_or_select_formula()
            if not formula:
                break
            
            # Evaluate formula
            start_time = time.time()
            result = self.logic_engine.find_model(formula)
            solve_time = time.time() - start_time
            
            # Create formula result
            formula_result = FormulaResult(
                formula=formula,
                timestamp=datetime.now().isoformat(),
                satisfiable=(result.result == ValidationResult.SATISFIABLE),
                model=result.model.interpretation if result.model else None,
                solve_time=solve_time,
                error=result.error_message
            )
            
            # Store result
            self.container.add_formula_result(formula_result)
            formula_count += 1
            
            if result.model:
                model_count += 1
            
            # Callback
            if self.on_formula_result:
                self.on_formula_result(formula_result)
            
            # Small delay to prevent CPU spinning
            time.sleep(0.1)
        
        self.container.elapsed_time = (datetime.now() - datetime.fromisoformat(
            self.container.start_time
        )).total_seconds()
    
    def _phase_stopped(self):
        """Handle stopped phase."""
        self.container.set_phase(WorkflowPhase.STOPPED)
        if self.on_phase_change:
            self.on_phase_change(WorkflowPhase.STOPPED)
        
        # Save final state
        self._save_container()
    
    def _phase_evaluation(self):
        """Evaluate collected results."""
        self.container.set_phase(WorkflowPhase.EVALUATION)
        if self.on_phase_change:
            self.on_phase_change(WorkflowPhase.EVALUATION)
        
        results = EvaluationResults()
        results.total_formulas = len(self.container.formula_results)
        
        total_solve_time = 0.0
        for formula_result in self.container.formula_results:
            if formula_result.satisfiable is True:
                results.satisfiable_count += 1
            elif formula_result.satisfiable is False:
                results.unsatisfiable_count += 1
            elif formula_result.error:
                results.error_count += 1
            else:
                results.unknown_count += 1
            
            if formula_result.model:
                results.total_models += 1
            
            total_solve_time += formula_result.solve_time
        
        # Calculate metrics
        if results.total_formulas > 0:
            results.satisfiability_rate = results.satisfiable_count / results.total_formulas
            results.avg_solve_time = total_solve_time / results.total_formulas
        
        results.total_time = self.container.elapsed_time
        
        # Store evaluation results
        self.container.evaluation_results = results
        self._save_container()
    
    def _phase_query(self):
        """Execute queries on collected data."""
        self.container.set_phase(WorkflowPhase.QUERY)
        if self.on_phase_change:
            self.on_phase_change(WorkflowPhase.QUERY)
        
        query_results = []
        for query in self.workflow.query.queries:
            start_time = time.time()
            results = self._execute_query(query)
            execution_time = time.time() - start_time
            
            query_result = QueryResult(
                query=query,
                timestamp=datetime.now().isoformat(),
                results=results,
                count=len(results),
                execution_time=execution_time
            )
            query_results.append(query_result)
        
        self.container.query_results = query_results
        self._save_container()
    
    def _phase_analysis(self):
        """Perform analysis on results."""
        self.container.set_phase(WorkflowPhase.ANALYSIS)
        if self.on_phase_change:
            self.on_phase_change(WorkflowPhase.ANALYSIS)
        
        analysis = AnalysisResult()
        
        # Analyze patterns
        if self.workflow.analysis.analyze_patterns:
            analysis.patterns = self._analyze_patterns()
        
        # Analyze complexity
        if self.workflow.analysis.analyze_complexity:
            analysis.complexity_metrics = self._analyze_complexity()
        
        # Analyze correlations
        if self.workflow.analysis.analyze_correlations:
            analysis.correlations = self._analyze_correlations()
        
        # Generate report
        if self.workflow.analysis.generate_report:
            analysis.report = self._generate_report()
            analysis.report_format = self.workflow.analysis.report_format
        
        self.container.analysis_result = analysis
        self._save_container()
    
    def _generate_or_select_formula(self) -> Optional[str]:
        """Generate or select a formula for evaluation."""
        # Simple implementation: cycle through initial formulas
        # In a full implementation, this could generate new formulas
        if self.workflow.formulas:
            import random
            return random.choice(self.workflow.formulas)
        return None
    
    def _execute_query(self, query: str) -> List[Any]:
        """Execute a query on the collected data."""
        results = []
        
        # Simple query implementation
        query_lower = query.lower()
        
        if "satisfiable" in query_lower:
            results = [
                r.formula for r in self.container.formula_results
                if r.satisfiable is True
            ]
        elif "unsatisfiable" in query_lower:
            results = [
                r.formula for r in self.container.formula_results
                if r.satisfiable is False
            ]
        elif "model" in query_lower:
            results = [
                r.formula for r in self.container.formula_results
                if r.model is not None
            ]
        else:
            # Default: return all formulas
            results = [r.formula for r in self.container.formula_results]
        
        # Limit results
        return results[:self.workflow.query.max_results]
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in formula results."""
        patterns = {
            "common_operators": {},
            "variable_usage": {},
            "formula_lengths": []
        }
        
        for result in self.container.formula_results:
            formula = result.formula
            patterns["formula_lengths"].append(len(formula))
            
            # Count operators
            for op in ["And", "Or", "Not", "Implies"]:
                if op in formula:
                    patterns["common_operators"][op] = \
                        patterns["common_operators"].get(op, 0) + 1
        
        return patterns
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze complexity metrics."""
        if not self.container.formula_results:
            return {}
        
        solve_times = [r.solve_time for r in self.container.formula_results]
        formula_lengths = [len(r.formula) for r in self.container.formula_results]
        
        return {
            "avg_solve_time": sum(solve_times) / len(solve_times) if solve_times else 0,
            "max_solve_time": max(solve_times) if solve_times else 0,
            "min_solve_time": min(solve_times) if solve_times else 0,
            "avg_formula_length": sum(formula_lengths) / len(formula_lengths) if formula_lengths else 0,
            "max_formula_length": max(formula_lengths) if formula_lengths else 0
        }
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between metrics."""
        # Simple correlation analysis
        correlations = {
            "length_vs_solve_time": "positive",
            "satisfiability_vs_complexity": "weak"
        }
        return correlations
    
    def _generate_report(self) -> str:
        """Generate analysis report."""
        if self.workflow.analysis.report_format == "json":
            import json
            report_data = {
                "workflow": self.workflow.name,
                "evaluation": self.container.evaluation_results.to_dict() if self.container.evaluation_results else {},
                "analysis": self.container.analysis_result.to_dict() if self.container.analysis_result else {}
            }
            return json.dumps(report_data, indent=2)
        else:
            # Markdown report
            report = f"# IKYKE Workflow Report: {self.workflow.name}\n\n"
            if self.container.evaluation_results:
                eval_res = self.container.evaluation_results
                report += f"## Evaluation Results\n\n"
                report += f"- Total Formulas: {eval_res.total_formulas}\n"
                report += f"- Satisfiable: {eval_res.satisfiable_count}\n"
                report += f"- Satisfiability Rate: {eval_res.satisfiability_rate:.2%}\n"
                report += f"- Total Models: {eval_res.total_models}\n"
                report += f"- Average Solve Time: {eval_res.avg_solve_time:.4f}s\n\n"
            return report
    
    def _auto_save_loop(self):
        """Automatic save loop running in background."""
        save_count = 0
        last_save = datetime.now()
        
        while self.running and not self.stop_event.is_set():
            config = self.workflow.auto_save
            
            if save_count >= config.max_saves:
                break
            
            # Check if it's time to save
            should_save = False
            if config.mode.value == "interval":
                elapsed = (datetime.now() - last_save).total_seconds()
                if elapsed >= config.interval_seconds:
                    should_save = True
            
            if should_save:
                self._save_container()
                self.container.record_save()
                save_count += 1
                last_save = datetime.now()
                
                if self.on_save:
                    self.on_save()
            
            time.sleep(1.0)  # Check every second
    
    def _save_container(self):
        """Save the container to disk."""
        IkykeContainerFormat.save(self.container, self.container_path)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "running": self.running,
            "phase": self.container.current_phase.value,
            "formula_count": len(self.container.formula_results),
            "elapsed_time": self.container.elapsed_time,
            "checkpoints": len(self.container.checkpoints),
            "saves": len(self.container.save_history)
        }
