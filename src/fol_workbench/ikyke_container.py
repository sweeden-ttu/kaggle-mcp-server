"""
IKYKE Data Container Format

The data container stores runtime data, results, and state information
for an IKYKE workflow execution. It's separate from the workflow definition
and contains:
- Runtime state
- Collected formulas
- Found models
- Evaluation results
- Query results
- Analysis results
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .ikyke_format import WorkflowPhase


@dataclass
class FormulaResult:
    """Result for a single formula evaluation."""
    formula: str
    timestamp: str
    satisfiable: Optional[bool] = None
    model: Optional[Dict[str, Any]] = None
    solve_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormulaResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationResults:
    """Results from the evaluation phase."""
    total_formulas: int = 0
    satisfiable_count: int = 0
    unsatisfiable_count: int = 0
    unknown_count: int = 0
    error_count: int = 0
    satisfiability_rate: float = 0.0
    total_models: int = 0
    avg_solve_time: float = 0.0
    total_time: float = 0.0
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResults':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class QueryResult:
    """Result from a query execution."""
    query: str
    timestamp: str
    results: List[Any] = None
    count: int = 0
    execution_time: float = 0.0
    
    def __post_init__(self):
        """Initialize default values."""
        if self.results is None:
            self.results = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AnalysisResult:
    """Result from the analysis phase."""
    patterns: Dict[str, Any] = None
    complexity_metrics: Dict[str, Any] = None
    correlations: Dict[str, Any] = None
    report: str = ""
    report_format: str = "json"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.patterns is None:
            self.patterns = {}
        if self.complexity_metrics is None:
            self.complexity_metrics = {}
        if self.correlations is None:
            self.correlations = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class IkykeContainer:
    """
    IKYKE data container for runtime data and results.
    
    This container stores all data collected during workflow execution:
    - Current phase and state
    - Formula results
    - Evaluation results
    - Query results
    - Analysis results
    - Checkpoints and save history
    """
    container_id: str = None
    workflow_id: str = None
    created: str = None
    modified: str = None
    
    # Runtime state
    current_phase: WorkflowPhase = WorkflowPhase.INITIALIZATION
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    elapsed_time: float = 0.0
    
    # Collected data
    formula_results: List[FormulaResult] = None
    checkpoints: List[str] = None  # Checkpoint IDs
    save_history: List[str] = None  # Save timestamps
    
    # Phase results
    evaluation_results: Optional[EvaluationResults] = None
    query_results: List[QueryResult] = None
    analysis_result: Optional[AnalysisResult] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.container_id is None:
            self.container_id = str(uuid.uuid4())
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.formula_results is None:
            self.formula_results = []
        if self.checkpoints is None:
            self.checkpoints = []
        if self.save_history is None:
            self.save_history = []
        if self.query_results is None:
            self.query_results = []
        if self.metadata is None:
            self.metadata = {}
    
    def add_formula_result(self, result: FormulaResult):
        """Add a formula evaluation result."""
        self.formula_results.append(result)
        self.modified = datetime.now().isoformat()
    
    def add_checkpoint(self, checkpoint_id: str):
        """Add a checkpoint ID."""
        if checkpoint_id not in self.checkpoints:
            self.checkpoints.append(checkpoint_id)
        self.modified = datetime.now().isoformat()
    
    def record_save(self):
        """Record an automatic save event."""
        self.save_history.append(datetime.now().isoformat())
        self.modified = datetime.now().isoformat()
    
    def set_phase(self, phase: WorkflowPhase):
        """Set the current workflow phase."""
        self.current_phase = phase
        self.modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "container_id": self.container_id,
            "workflow_id": self.workflow_id,
            "created": self.created,
            "modified": self.modified,
            "current_phase": self.current_phase.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": self.elapsed_time,
            "formula_results": [r.to_dict() for r in self.formula_results],
            "checkpoints": self.checkpoints,
            "save_history": self.save_history,
            "metadata": self.metadata
        }
        
        if self.evaluation_results:
            data["evaluation_results"] = self.evaluation_results.to_dict()
        if self.query_results:
            data["query_results"] = [q.to_dict() for q in self.query_results]
        if self.analysis_result:
            data["analysis_result"] = self.analysis_result.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IkykeContainer':
        """Create from dictionary."""
        container = cls(
            container_id=data.get("container_id"),
            workflow_id=data.get("workflow_id"),
            created=data.get("created"),
            modified=data.get("modified"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            elapsed_time=data.get("elapsed_time", 0.0),
            checkpoints=data.get("checkpoints", []),
            save_history=data.get("save_history", []),
            metadata=data.get("metadata", {})
        )
        
        # Set phase
        if "current_phase" in data:
            container.current_phase = WorkflowPhase(data["current_phase"])
        
        # Load formula results
        if "formula_results" in data:
            container.formula_results = [
                FormulaResult.from_dict(r) for r in data["formula_results"]
            ]
        
        # Load evaluation results
        if "evaluation_results" in data:
            container.evaluation_results = EvaluationResults.from_dict(
                data["evaluation_results"]
            )
        
        # Load query results
        if "query_results" in data:
            container.query_results = [
                QueryResult.from_dict(q) for q in data["query_results"]
            ]
        
        # Load analysis result
        if "analysis_result" in data:
            container.analysis_result = AnalysisResult.from_dict(
                data["analysis_result"]
            )
        
        return container


class IkykeContainerFormat:
    """
    Handler for IKYKE container file format.
    
    Containers are saved with .ikyke_container extension and contain
    runtime data separate from workflow definitions.
    """
    
    @staticmethod
    def save(container: IkykeContainer, filepath: Union[str, Path]) -> Path:
        """
        Save an IKYKE container to file.
        
        Args:
            container: IkykeContainer object to save
            filepath: Path to save the file
        
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.ikyke_container')
        
        # Update modified timestamp
        container.modified = datetime.now().isoformat()
        
        # Convert to dictionary
        data = container.to_dict()
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> IkykeContainer:
        """
        Load an IKYKE container from file.
        
        Args:
            filepath: Path to .ikyke_container file
        
        Returns:
            IkykeContainer object
        
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"IKYKE container not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return IkykeContainer.from_dict(data)
    
    @staticmethod
    def create(workflow_id: str) -> IkykeContainer:
        """
        Create a new empty IKYKE container.
        
        Args:
            workflow_id: ID of the associated workflow
        
        Returns:
            New IkykeContainer
        """
        container = IkykeContainer()
        container.workflow_id = workflow_id
        return container
