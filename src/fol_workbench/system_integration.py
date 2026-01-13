"""
System Integration

Integrates all components:
- Database (users, axioms, theses)
- Task engine and planner
- Calendar and schedule
- Storage backends
- Document generator
- Web scraper and pourage
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from .database import Database, User, Axiom, Thesis, Task, ScheduleEvent
from .task_engine import TaskEngine, Calendar, TaskPlanner
from .storage import StorageManager, LocalStorage, NetworkStorage, CloudStorage, GitHubStorage
from .document_generator import DocumentGenerator
from .web_scraper import WebScraper, Pourage
from .logic_layer import LogicEngine


class FOLWorkbenchSystem:
    """Complete FOL Workbench system integration."""
    
    def __init__(self, db_path: str = "fol_workbench.db"):
        # Database
        self.db = Database(db_path)
        
        # Task management
        self.task_engine = TaskEngine(self.db)
        self.calendar = Calendar(self.db)
        self.task_planner = TaskPlanner(self.db)
        
        # Storage
        self.storage = StorageManager()
        self.storage.register_backend("local", LocalStorage(), default=True)
        
        # Document generation
        self.document_generator = DocumentGenerator()
        
        # Web scraping
        try:
            self.web_scraper = WebScraper()
            self.pourage = Pourage()
        except ImportError:
            self.web_scraper = None
            self.pourage = None
        
        # Logic engine
        self.logic_engine = LogicEngine()
    
    def setup_storage_backends(
        self,
        network_path: Optional[str] = None,
        cloud_config: Optional[Dict[str, Any]] = None,
        github_repo: Optional[str] = None
    ):
        """Set up additional storage backends."""
        if network_path:
            self.storage.register_backend("network", NetworkStorage(network_path))
        
        if cloud_config:
            provider = cloud_config.get("provider", "s3")
            self.storage.register_backend("cloud", CloudStorage(provider, **cloud_config))
        
        if github_repo:
            token = cloud_config.get("github_token") if cloud_config else None
            self.storage.register_backend("github", GitHubStorage(github_repo, token))
    
    def create_user(self, username: str, email: str) -> User:
        """Create a new user."""
        return self.db.create_user(username, email)
    
    def create_axiom(self, user_id: str, name: str, formula: str, **kwargs) -> Axiom:
        """Create an axiom."""
        return self.db.create_axiom(user_id, name, formula, **kwargs)
    
    def create_thesis(self, user_id: str, title: str, statement: str, **kwargs) -> Thesis:
        """Create a thesis."""
        return self.db.create_thesis(user_id, title, statement, **kwargs)
    
    def create_task(self, user_id: str, title: str, **kwargs) -> Task:
        """Create a task."""
        return self.task_engine.create_task(user_id, title, **kwargs)
    
    def create_calendar_event(self, user_id: str, title: str, start_time: str, end_time: str, **kwargs) -> ScheduleEvent:
        """Create a calendar event."""
        return self.calendar.create_event(user_id, title, start_time, end_time, **kwargs)
    
    def generate_thesis_document(self, thesis_id: str, output_path: str) -> Path:
        """Generate beautiful thesis document."""
        thesis = self.db.theses.get(thesis_id)
        if not thesis:
            raise ValueError(f"Thesis {thesis_id} not found")
        
        document = self.document_generator.generate_thesis_document(thesis.to_dict())
        return self.document_generator.save_document(document, output_path)
    
    def generate_task_report(self, user_id: str, output_path: str) -> Path:
        """Generate task report."""
        tasks = self.task_engine.get_tasks(user_id)
        document = self.document_generator.generate_task_report([t.to_dict() for t in tasks])
        return self.document_generator.save_document(document, output_path)
    
    def scrape_and_extract_formulas(self, url: str) -> List[str]:
        """Scrape web page and extract formulas."""
        if not self.web_scraper:
            raise ImportError("BeautifulSoup4 is required for web scraping")
        return self.web_scraper.extract_formulas(url)
    
    def aggregate_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate content from multiple sources using Pourage."""
        if not self.pourage:
            raise ImportError("BeautifulSoup4 is required for Pourage")
        
        for source in sources:
            if source['type'] == 'web':
                self.pourage.add_web_source(source['url'])
            elif source['type'] == 'file':
                self.pourage.add_file_source(source['path'])
        
        return self.pourage.aggregate()
    
    def plan_week(self, user_id: str) -> Dict[str, Any]:
        """Plan tasks and schedule for the week."""
        return self.task_planner.plan_week(user_id)
    
    def save_to_storage(self, path: str, data: Any, backend: str = "local") -> bool:
        """Save data to storage backend."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            import json
            data = json.dumps(data, indent=2).encode('utf-8')
        return self.storage.save(path, data, backend)
    
    def load_from_storage(self, path: str, backend: str = "local") -> Optional[bytes]:
        """Load data from storage backend."""
        return self.storage.load(path, backend)
