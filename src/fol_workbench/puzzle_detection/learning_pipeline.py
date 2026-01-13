"""
Global Learning Pipeline Orchestrator

Orchestrates the complete learning cycle:
assign → iterate → evaluate → test → propose → infer → learn → save → optimize → eliminate → learn → plan → evaluate → experiment → design → performance improve
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

from .hyperparameter_tuner import HyperparameterTuner, HyperparameterConfig, EpisodeResult
from .confidence_evaluator import ConfidenceEvaluator, ImprovementEvidence
from .perceptron_units import PerceptronUnit
from .bayesian_perceptron_assigner import BayesianPerceptronAssigner
from ..database import Database, Perceptron


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
        bayesian_assigner: Optional[BayesianPerceptronAssigner] = None
    ):
        """
        Initialize learning pipeline.
        
        Args:
            database: Database instance for persistence
            hyperparameter_tuner: Hyperparameter tuner instance
            confidence_evaluator: Confidence evaluator instance
            bayesian_assigner: Bayesian perceptron assigner instance
        """
        self.database = database or Database()
        self.hyperparameter_tuner = hyperparameter_tuner or HyperparameterTuner()
        self.confidence_evaluator = confidence_evaluator or ConfidenceEvaluator()
        self.bayesian_assigner = bayesian_assigner or BayesianPerceptronAssigner()
        
        self.state = PipelineState()
        self.running = False
    
    def run_episode(
        self,
        training_data: List[Tuple[np.ndarray, int]],
        validation_data: Optional[List[Tuple[np.ndarray, int]]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete learning episode.
        
        Args:
            training_data: List of (features, target) tuples for training
            validation_data: Optional validation data for testing
            
        Returns:
            Dictionary with episode results
        """
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
    
    def _step_assign(self, episode_id: int) -> HyperparameterConfig:
        """Step 1: Assign hyperparameters for episode."""
        config = self.hyperparameter_tuner.assign_hyperparameters(episode_id)
        self.state.current_config = config
        return config
    
    def _step_iterate(
        self,
        training_data: List[Tuple[np.ndarray, int]],
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
        
        for iteration in range(config.iterations):
            for features, target in training_data[:config.batch_size * 10]:  # Limit samples per iteration
                # Assign to a perceptron (simplified - would use Bayesian assigner)
                perceptron = list(self.state.perceptrons.values())[0]
                perceptron.update_learning_rate(config.learning_rate)
                
                is_correct = perceptron.train(features, target)
                if is_correct:
                    total_correct += 1
                total_samples += 1
        
        return {
            'iterations': config.iterations,
            'accuracy': total_correct / max(1, total_samples),
            'samples_processed': total_samples
        }
    
    def _step_evaluate(self) -> Dict[str, Any]:
        """Step 3: Measure performance and confidence."""
        if not self.state.perceptrons:
            return {'average_accuracy': 0.0, 'average_confidence': 0.0}
        
        accuracies = [p.accuracy for p in self.state.perceptrons.values()]
        confidences = [p.confidence for p in self.state.perceptrons.values()]
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Record in confidence evaluator
        self.confidence_evaluator.record_performance(avg_accuracy, avg_confidence)
        
        return {
            'average_accuracy': avg_accuracy,
            'average_confidence': avg_confidence,
            'perceptron_count': len(self.state.perceptrons)
        }
    
    def _step_test(
        self,
        validation_data: List[Tuple[np.ndarray, int]]
    ) -> Dict[str, Any]:
        """Step 4: Test perceptrons on validation data."""
        if not self.state.perceptrons or not validation_data:
            return {'skipped': True}
        
        total_correct = 0
        total_samples = len(validation_data)
        
        for features, target in validation_data:
            perceptron = list(self.state.perceptrons.values())[0]  # Simplified
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
        training_data: List[Tuple[np.ndarray, int]]
    ) -> Dict[str, Any]:
        """Step 7: Update perceptron weights and Bayesian priors."""
        if not self.state.perceptrons or not training_data:
            return {'skipped': True}
        
        # Continue training
        total_updates = 0
        for features, target in training_data[:100]:  # Limit for performance
            perceptron = list(self.state.perceptrons.values())[0]
            perceptron.train(features, target)
            total_updates += 1
        
        return {
            'updates_performed': total_updates,
            'perceptrons_trained': len(self.state.perceptrons)
        }
    
    def _step_save(self) -> Dict[str, Any]:
        """Step 8: Persist high-confidence perceptrons to database."""
        if not self.state.perceptrons:
            return {'count': 0}
        
        # Select high-confidence perceptrons using Bayesian assigner
        selected = self.bayesian_assigner.select_high_confidence_perceptrons(
            self.state.perceptrons,
            threshold=0.8
        )
        
        saved_count = 0
        for perceptron_id, perceptron in selected:
            state = perceptron.get_state()
            
            # Save to database
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
        # Load high-confidence perceptrons from database
        high_conf_perceptrons = self.database.get_high_confidence_perceptrons(threshold=0.8)
        
        return {
            'high_confidence_perceptrons_available': len(high_conf_perceptrons),
            'in_memory_perceptrons': len(self.state.perceptrons)
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
