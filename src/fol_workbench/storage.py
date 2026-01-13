"""
Storage Abstraction Layer

Supports multiple storage backends:
- Local storage (filesystem)
- Network storage (SMB/NFS)
- Cloud storage (S3, Azure, GCS)
- GitHub storage (Git repositories)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO
import json
import os


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save(self, path: str, data: bytes) -> bool:
        """Save data to storage."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> Optional[bytes]:
        """Load data from storage."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete path from storage."""
        pass
    
    @abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        """List paths with given prefix."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage."""
    
    def __init__(self, base_path: str = "./storage/local"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path."""
        return self.base_path / path.lstrip('/')
    
    def save(self, path: str, data: bytes) -> bool:
        try:
            full_path = self._get_full_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            return False
    
    def load(self, path: str) -> Optional[bytes]:
        try:
            full_path = self._get_full_path(path)
            if not full_path.exists():
                return None
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def exists(self, path: str) -> bool:
        return self._get_full_path(path).exists()
    
    def delete(self, path: str) -> bool:
        try:
            full_path = self._get_full_path(path)
            if full_path.exists():
                full_path.unlink()
            return True
        except Exception:
            return False
    
    def list(self, prefix: str = "") -> List[str]:
        try:
            prefix_path = self._get_full_path(prefix)
            if not prefix_path.exists():
                return []
            return [str(p.relative_to(self.base_path)) for p in prefix_path.rglob('*') if p.is_file()]
        except Exception:
            return []


class NetworkStorage(StorageBackend):
    """Network storage (SMB/NFS) - placeholder implementation."""
    
    def __init__(self, network_path: str, username: Optional[str] = None, password: Optional[str] = None):
        self.network_path = network_path
        self.username = username
        self.password = password
        # In production, use smbprotocol or similar
    
    def save(self, path: str, data: bytes) -> bool:
        # Placeholder - would use SMB/NFS client
        try:
            full_path = Path(self.network_path) / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            return False
    
    def load(self, path: str) -> Optional[bytes]:
        try:
            full_path = Path(self.network_path) / path
            if not full_path.exists():
                return None
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def exists(self, path: str) -> bool:
        return (Path(self.network_path) / path).exists()
    
    def delete(self, path: str) -> bool:
        try:
            (Path(self.network_path) / path).unlink()
            return True
        except Exception:
            return False
    
    def list(self, prefix: str = "") -> List[str]:
        try:
            prefix_path = Path(self.network_path) / prefix
            return [str(p.relative_to(Path(self.network_path))) for p in prefix_path.rglob('*') if p.is_file()]
        except Exception:
            return []


class CloudStorage(StorageBackend):
    """Cloud storage abstraction (S3, Azure, GCS)."""
    
    def __init__(self, provider: str = "s3", **config):
        self.provider = provider
        self.config = config
        # In production, use boto3, azure-storage-blob, google-cloud-storage
    
    def save(self, path: str, data: bytes) -> bool:
        # Placeholder - would use cloud SDK
        return False
    
    def load(self, path: str) -> Optional[bytes]:
        # Placeholder
        return None
    
    def exists(self, path: str) -> bool:
        return False
    
    def delete(self, path: str) -> bool:
        return False
    
    def list(self, prefix: str = "") -> List[str]:
        return []


class GitHubStorage(StorageBackend):
    """GitHub storage using Git repositories."""
    
    def __init__(self, repo_url: str, token: Optional[str] = None, branch: str = "main"):
        self.repo_url = repo_url
        self.token = token
        self.branch = branch
        self.local_path = Path(f"./storage/github/{Path(repo_url).stem}")
        # In production, use PyGithub or gitpython
    
    def save(self, path: str, data: bytes) -> bool:
        try:
            full_path = self.local_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(data)
            # In production, commit and push to GitHub
            return True
        except Exception:
            return False
    
    def load(self, path: str) -> Optional[bytes]:
        try:
            full_path = self.local_path / path
            if not full_path.exists():
                return None
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def exists(self, path: str) -> bool:
        return (self.local_path / path).exists()
    
    def delete(self, path: str) -> bool:
        try:
            (self.local_path / path).unlink()
            return True
        except Exception:
            return False
    
    def list(self, prefix: str = "") -> List[str]:
        try:
            prefix_path = self.local_path / prefix
            return [str(p.relative_to(self.local_path)) for p in prefix_path.rglob('*') if p.is_file()]
        except Exception:
            return []


class StorageManager:
    """Manages multiple storage backends."""
    
    def __init__(self):
        self.backends: Dict[str, StorageBackend] = {}
        self.default_backend: Optional[str] = None
    
    def register_backend(self, name: str, backend: StorageBackend, default: bool = False):
        """Register a storage backend."""
        self.backends[name] = backend
        if default or self.default_backend is None:
            self.default_backend = name
    
    def get_backend(self, name: Optional[str] = None) -> Optional[StorageBackend]:
        """Get a storage backend by name."""
        if name is None:
            name = self.default_backend
        return self.backends.get(name)
    
    def save(self, path: str, data: bytes, backend: Optional[str] = None) -> bool:
        """Save to specified backend or default."""
        storage = self.get_backend(backend)
        if storage:
            return storage.save(path, data)
        return False
    
    def load(self, path: str, backend: Optional[str] = None) -> Optional[bytes]:
        """Load from specified backend or default."""
        storage = self.get_backend(backend)
        if storage:
            return storage.load(path)
        return None
    
    def save_json(self, path: str, data: Any, backend: Optional[str] = None) -> bool:
        """Save JSON data."""
        json_data = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        return self.save(path, json_data, backend)
    
    def load_json(self, path: str, backend: Optional[str] = None) -> Optional[Any]:
        """Load JSON data."""
        data = self.load(path, backend)
        if data:
            return json.loads(data.decode('utf-8'))
        return None
