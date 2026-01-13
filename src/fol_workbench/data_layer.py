"""Data Layer: Handles persistence, checkpoints, and file I/O for JSON and SMT-LIB formats."""

import json
import os
import uuid
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
    
    def generate_dataset_name(self, prefix: str = "model") -> str:
        """
        Generate a unique dataset name using timestamp and UUID.
        
        Format: {prefix}_{YYYYMMDD}_v{version}_{short_uuid}
        Example: model_20231027_v1_a3f2b1
        
        Args:
            prefix: Name prefix (default: "model")
        
        Returns:
            Unique dataset name
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_v1_{short_uuid}"
    
    def create_dataset(self, name: Optional[str] = None) -> Path:
        """
        Create a new dataset folder.
        
        A dataset is a folder containing multiple .json or .smt2 files.
        If name is not provided, generates a unique name using timestamp and UUID.
        
        Args:
            name: Dataset name (optional, auto-generated if None)
        
        Returns:
            Path to the created dataset directory
        """
        if name is None:
            name = self.generate_dataset_name()
        
        dataset_path = self.datasets_dir / name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            "name": name,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "files": []
        }
        
        metadata_file = dataset_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return dataset_path
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets (folders in datasets directory).
        
        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []
        for dataset_path in self.datasets_dir.iterdir():
            if dataset_path.is_dir():
                metadata_file = dataset_path / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        datasets.append(metadata)
                    except Exception:
                        continue
                else:
                    # Dataset without metadata - create basic entry
                    datasets.append({
                        "name": dataset_path.name,
                        "created": "unknown",
                        "modified": "unknown",
                        "files": []
                    })
        
        return sorted(datasets, key=lambda d: d.get("modified", ""), reverse=True)
    
    def get_dataset_path(self, name: str) -> Optional[Path]:
        """
        Get the path to a dataset folder.
        
        Args:
            name: Dataset name
        
        Returns:
            Path to dataset directory or None if not found
        """
        dataset_path = self.datasets_dir / name
        if dataset_path.exists() and dataset_path.is_dir():
            return dataset_path
        return None
    
    def add_file_to_dataset(self, dataset_name: str, file_path: Path, file_type: str = "json") -> bool:
        """
        Add a file reference to a dataset's metadata.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to the file (relative to dataset or absolute)
            file_type: Type of file ("json", "smt2", etc.)
        
        Returns:
            True if successful, False otherwise
        """
        dataset_path = self.get_dataset_path(dataset_name)
        if not dataset_path:
            return False
        
        metadata_file = dataset_path / "metadata.json"
        if not metadata_file.exists():
            return False
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            file_info = {
                "name": file_path.name,
                "path": str(file_path),
                "type": file_type,
                "added": datetime.now().isoformat()
            }
            
            if "files" not in metadata:
                metadata["files"] = []
            
            metadata["files"].append(file_info)
            metadata["modified"] = datetime.now().isoformat()
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False
    
    def import_dataset_folder(self, folder_path: Path, dataset_name: Optional[str] = None) -> Optional[Path]:
        """
        Import a folder as a dataset.
        
        Uses QFileDialog to browse folders. A dataset is treated as a folder
        containing multiple .json or .smt2 files.
        
        Args:
            folder_path: Path to the folder to import
            dataset_name: Optional name for the dataset (auto-generated if None)
        
        Returns:
            Path to the created dataset directory, or None if failed
        """
        if not folder_path.exists() or not folder_path.is_dir():
            return None
        
        # Create dataset
        if dataset_name is None:
            dataset_name = self.generate_dataset_name()
        
        dataset_path = self.create_dataset(dataset_name)
        
        # Copy files from source folder
        imported_count = 0
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                if file_path.suffix in ['.json', '.smt2']:
                    # Copy file to dataset
                    dest_file = dataset_path / file_path.name
                    import shutil
                    shutil.copy2(file_path, dest_file)
                    
                    # Add to metadata
                    self.add_file_to_dataset(
                        dataset_name,
                        dest_file,
                        file_path.suffix[1:]  # Remove leading dot
                    )
                    imported_count += 1
        
        return dataset_path
    
    def save_checkpoint(
        self,
        formula: str,
        constraints: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        all_formulas: Optional[List[str]] = None
    ) -> Checkpoint:
        """
        Save a checkpoint with complete formula tracking.
        
        This method saves a checkpoint that includes:
        - Main formula
        - Constraint formulas
        - All formulas (for complete restoration)
        - Metadata
        
        To restore a checkpoint, load it and re-apply all formulas using
        logic_engine.restore_from_formulas(all_formulas).
        
        Args:
            formula: The main FOL formula
            constraints: List of constraint formulas
            metadata: Optional metadata dictionary
            checkpoint_id: Optional checkpoint ID (generated if not provided)
            all_formulas: Complete list of all formulas for restoration
        
        Returns:
            The saved checkpoint
        """
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Combine all formulas for restoration
        if all_formulas is None:
            all_formulas = []
            if formula:
                all_formulas.append(formula)
            all_formulas.extend(constraints)
        
        checkpoint_data = {
            "id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "formula": formula,
            "constraints": constraints,
            "all_formulas": all_formulas,  # Complete list for restoration
            "metadata": metadata or {}
        }
        
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Create Checkpoint object for return
        checkpoint = Checkpoint(
            id=checkpoint_id,
            timestamp=checkpoint_data["timestamp"],
            formula=formula,
            constraints=constraints,
            metadata=checkpoint_data["metadata"]
        )
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint by ID with full formula tracking.
        
        Args:
            checkpoint_id: The checkpoint ID
        
        Returns:
            Dictionary with checkpoint data including 'all_formulas' for restoration,
            or None if not found
        """
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure all_formulas exists (for backward compatibility)
        if "all_formulas" not in data:
            all_formulas = []
            if data.get("formula"):
                all_formulas.append(data["formula"])
            all_formulas.extend(data.get("constraints", []))
            data["all_formulas"] = all_formulas
        
        return data
    
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
