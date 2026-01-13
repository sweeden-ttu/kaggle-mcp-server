"""
IKYKE File Format Specification and Parser

IKYKE (Interactive Knowledge Yielding Kernel Engine) is a protocol for
automated FOL workbench workflows that:
1. Automatically saves work at intervals
2. Runs experiments for 3-5 minutes
3. Stops automatically
4. Evaluates results
5. Begins query phase
6. Performs analysis

File Format:
- Extension: .ikyke
- Format: JSON-based with metadata and workflow definitions
- Structure: Header + Workflow + Data Container
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum


class WorkflowPhase(Enum):
    """Phases of the IKYKE workflow."""
    INITIALIZATION = "initialization"
    RUNNING = "running"
    STOPPED = "stopped"
    EVALUATION = "evaluation"
    QUERY = "query"
    ANALYSIS = "analysis"
    COMPLETED = "completed"
    ERROR = "error"


class SaveMode(Enum):
    """Automatic save modes."""
    INTERVAL = "interval"  # Save at fixed intervals
    EVENT = "event"  # Save on specific events
    CONTINUOUS = "continuous"  # Save continuously
    MANUAL = "manual"  # Manual saves only


@dataclass
class IkykeHeader:
    """
    IKYKE file header containing metadata and format version.
    
    This header identifies the file format and provides metadata
    about the workflow definition.
    """
    format_version: str = "1.0"
    format_name: str = "IKYKE"
    created: str = None
    modified: str = None
    workflow_id: str = None
    description: str = ""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.workflow_id is None:
            self.workflow_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IkykeHeader':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AutoSaveConfig:
    """
    Configuration for automatic saving.
    
    Defines when and how the workflow should automatically save
    its state and results.
    """
    enabled: bool = True
    mode: SaveMode = SaveMode.INTERVAL
    interval_seconds: int = 30  # For INTERVAL mode
    events: List[str] = None  # For EVENT mode: ["formula_added", "model_found", etc.]
    max_saves: int = 100  # Maximum number of auto-saves
    
    def __post_init__(self):
        """Initialize default values."""
        if self.events is None:
            self.events = ["formula_added", "model_found", "checkpoint"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['mode'] = self.mode.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoSaveConfig':
        """Create from dictionary."""
        if isinstance(data.get('mode'), str):
            data['mode'] = SaveMode(data['mode'])
        return cls(**data)


@dataclass
class RunConfig:
    """
    Configuration for the running phase.
    
    Defines how long to run experiments and when to stop.
    """
    duration_min: int = 3  # Minimum duration in minutes
    duration_max: int = 5  # Maximum duration in minutes
    stop_conditions: List[str] = None  # Conditions that trigger early stop
    max_formulas: int = 1000  # Maximum formulas to process
    max_models: int = 100  # Maximum models to find
    
    def __post_init__(self):
        """Initialize default values."""
        if self.stop_conditions is None:
            self.stop_conditions = ["time_limit", "max_formulas", "max_models"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation phase.
    
    Defines what to evaluate and how to measure results.
    """
    evaluate_satisfiability: bool = True
    evaluate_models: bool = True
    evaluate_implications: bool = True
    metrics: List[str] = None  # Metrics to compute
    threshold_satisfiable: float = 0.5  # Threshold for satisfiability rate
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metrics is None:
            self.metrics = ["satisfiability_rate", "model_count", "avg_solve_time"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class QueryConfig:
    """
    Configuration for the query phase.
    
    Defines queries to run on the collected data.
    """
    queries: List[str] = None  # Query strings or patterns
    query_language: str = "fol"  # Query language: "fol", "smt", "json"
    max_results: int = 100
    
    def __post_init__(self):
        """Initialize default values."""
        if self.queries is None:
            self.queries = [
                "Find all satisfiable formulas",
                "Find formulas with models",
                "Find unsatisfiable formulas"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AnalysisConfig:
    """
    Configuration for the analysis phase.
    
    Defines what analysis to perform on results.
    """
    analyze_patterns: bool = True
    analyze_complexity: bool = True
    analyze_correlations: bool = True
    generate_report: bool = True
    report_format: str = "json"  # "json", "markdown", "html"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class IkykeWorkflow:
    """
    Complete IKYKE workflow definition.
    
    Contains all configuration for the automated workflow phases.
    """
    name: str
    header: IkykeHeader
    auto_save: AutoSaveConfig
    run: RunConfig
    evaluation: EvaluationConfig
    query: QueryConfig
    analysis: AnalysisConfig
    formulas: List[str] = None  # Initial formulas to process
    constraints: List[str] = None  # Initial constraints
    
    def __post_init__(self):
        """Initialize default values."""
        if self.formulas is None:
            self.formulas = []
        if self.constraints is None:
            self.constraints = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "header": self.header.to_dict(),
            "auto_save": self.auto_save.to_dict(),
            "run": self.run.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "query": self.query.to_dict(),
            "analysis": self.analysis.to_dict(),
            "formulas": self.formulas,
            "constraints": self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IkykeWorkflow':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            header=IkykeHeader.from_dict(data["header"]),
            auto_save=AutoSaveConfig.from_dict(data["auto_save"]),
            run=RunConfig.from_dict(data["run"]),
            evaluation=EvaluationConfig.from_dict(data["evaluation"]),
            query=QueryConfig.from_dict(data["query"]),
            analysis=AnalysisConfig.from_dict(data["analysis"]),
            formulas=data.get("formulas", []),
            constraints=data.get("constraints", [])
        )


class IkykeFileFormat:
    """
    IKYKE file format handler.
    
    Provides methods to read and write .ikyke files following
    the IKYKE protocol specification.
    """
    
    @staticmethod
    def save(workflow: IkykeWorkflow, filepath: Union[str, Path]) -> Path:
        """
        Save an IKYKE workflow to file.
        
        Args:
            workflow: IkykeWorkflow object to save
            filepath: Path to save the file
        
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.ikyke')
        
        # Update modified timestamp
        workflow.header.modified = datetime.now().isoformat()
        
        # Convert to dictionary
        data = workflow.to_dict()
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> IkykeWorkflow:
        """
        Load an IKYKE workflow from file.
        
        Args:
            filepath: Path to .ikyke file
        
        Returns:
            IkykeWorkflow object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"IKYKE file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate format
        if "header" not in data:
            raise ValueError("Invalid IKYKE file: missing header")
        
        if data["header"].get("format_name") != "IKYKE":
            raise ValueError("Invalid IKYKE file: incorrect format name")
        
        return IkykeWorkflow.from_dict(data)
    
    @staticmethod
    def create_default(name: str = "Default Workflow") -> IkykeWorkflow:
        """
        Create a default IKYKE workflow with standard settings.
        
        Args:
            name: Workflow name
        
        Returns:
            IkykeWorkflow with default configuration
        """
        return IkykeWorkflow(
            name=name,
            header=IkykeHeader(description=f"Default IKYKE workflow: {name}"),
            auto_save=AutoSaveConfig(),
            run=RunConfig(),
            evaluation=EvaluationConfig(),
            query=QueryConfig(),
            analysis=AnalysisConfig()
        )
