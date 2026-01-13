"""Data Layer: Handles persistence, checkpoints, and file I/O for JSON and SMT-LIB formats."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class FileFormat(Enum):
    """Supported file formats."""
    JSON = "json"
    SMT_LIB = "smt2"
    PROJECT = "folproj"


@dataclass
class Checkpoint:
    """Represents a checkpoint in the workbench."""
    id: str
    timestamp: str
    formula: str
    constraints: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create checkpoint from dictionary."""
        return cls(**data)


@dataclass
class ProjectMetadata:
    """Project metadata structure."""
    name: str
    description: str
    created: str
    modified: str
    checkpoints: List[str]  # List of checkpoint IDs
    datasets: List[str]  # List of dataset names
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectMetadata':
        """Create from dictionary."""
        return cls(**data)


class DataLayer:
    """Data persistence layer for FOL workbench."""
    
    def __init__(self, project_dir: Optional[Path] = None):
        """
        Initialize data layer.
        
        Args:
            project_dir: Base directory for projects. Defaults to ~/.fol_workbench
        """
        if project_dir is None:
            project_dir = Path.home() / ".fol_workbench"
        
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.project_dir / "checkpoints"
        self.datasets_dir = self.project_dir / "datasets"
        self.projects_dir = self.project_dir / "projects"
        
        # Create subdirectories
        for dir_path in [self.checkpoints_dir, self.datasets_dir, self.projects_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        formula: str,
        constraints: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None
    ) -> Checkpoint:
        """
        Save a checkpoint.
        
        Args:
            formula: The FOL formula
            constraints: List of constraint formulas
            metadata: Optional metadata dictionary
            checkpoint_id: Optional checkpoint ID (generated if not provided)
        
        Returns:
            The saved checkpoint
        """
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        checkpoint = Checkpoint(
            id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            formula=formula,
            constraints=constraints,
            metadata=metadata or {}
        )
        
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: The checkpoint ID
        
        Returns:
            The checkpoint or None if not found
        """
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Checkpoint.from_dict(data)
    
    def list_checkpoints(self) -> List[Checkpoint]:
        """List all checkpoints."""
        checkpoints = []
        for checkpoint_file in self.checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                checkpoints.append(Checkpoint.from_dict(data))
            except Exception:
                continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: The checkpoint ID
        
        Returns:
            True if deleted, False if not found
        """
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False
    
    def save_project(
        self,
        name: str,
        description: str = "",
        checkpoints: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None
    ) -> ProjectMetadata:
        """
        Save or update a project.
        
        Args:
            name: Project name
            description: Project description
            checkpoints: List of checkpoint IDs
            datasets: List of dataset names
        
        Returns:
            The project metadata
        """
        project_file = self.projects_dir / f"{name}.json"
        
        if project_file.exists():
            with open(project_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            project = ProjectMetadata.from_dict(data)
            project.description = description
            project.modified = datetime.now().isoformat()
            if checkpoints is not None:
                project.checkpoints = checkpoints
            if datasets is not None:
                project.datasets = datasets
        else:
            project = ProjectMetadata(
                name=name,
                description=description,
                created=datetime.now().isoformat(),
                modified=datetime.now().isoformat(),
                checkpoints=checkpoints or [],
                datasets=datasets or []
            )
        
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project.to_dict(), f, indent=2, ensure_ascii=False)
        
        return project
    
    def load_project(self, name: str) -> Optional[ProjectMetadata]:
        """
        Load a project by name.
        
        Args:
            name: Project name
        
        Returns:
            The project metadata or None if not found
        """
        project_file = self.projects_dir / f"{name}.json"
        if not project_file.exists():
            return None
        
        with open(project_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ProjectMetadata.from_dict(data)
    
    def list_projects(self) -> List[ProjectMetadata]:
        """List all projects."""
        projects = []
        for project_file in self.projects_dir.glob("*.json"):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                projects.append(ProjectMetadata.from_dict(data))
            except Exception:
                continue
        
        # Sort by modified date (newest first)
        projects.sort(key=lambda p: p.modified, reverse=True)
        return projects
    
    def save_smt_lib(self, content: str, filename: str) -> Path:
        """
        Save content as SMT-LIB format.
        
        Args:
            content: SMT-LIB formatted content
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        if not filename.endswith('.smt2'):
            filename += '.smt2'
        
        output_file = self.datasets_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_file
    
    def load_smt_lib(self, filename: str) -> Optional[str]:
        """
        Load SMT-LIB file.
        
        Args:
            filename: Input filename
        
        Returns:
            File content or None if not found
        """
        if not filename.endswith('.smt2'):
            filename += '.smt2'
        
        input_file = self.datasets_dir / filename
        if not input_file.exists():
            return None
        
        with open(input_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> Path:
        """
        Export data to JSON file.
        
        Args:
            data: Data dictionary
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'
        
        output_file = self.datasets_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def import_from_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Import data from JSON file.
        
        Args:
            filename: Input filename
        
        Returns:
            Data dictionary or None if not found
        """
        if not filename.endswith('.json'):
            filename += '.json'
        
        input_file = self.datasets_dir / filename
        if not input_file.exists():
            return None
        
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
