"""
Database Models and Schema

Defines database tables for:
- Users
- Axioms
- Theses
- Tasks
- Schedules
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User table model."""
    user_id: str
    username: str
    email: str
    created: str = None
    modified: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(**data)


@dataclass
class Axiom:
    """Axiom table model for storing logical axioms."""
    axiom_id: str
    user_id: str
    name: str
    formula: str
    description: str = ""
    created: str = None
    modified: str = None
    tags: List[str] = None
    verified: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Axiom':
        return cls(**data)


@dataclass
class Thesis:
    """Thesis table model for storing logical theses/propositions."""
    thesis_id: str
    user_id: str
    title: str
    statement: str
    proof: str = ""
    status: str = "draft"
    created: str = None
    modified: str = None
    axioms_used: List[str] = None  # List of axiom IDs
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.axioms_used is None:
            self.axioms_used = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thesis':
        return cls(**data)


@dataclass
class Task:
    """Task model for task planner."""
    task_id: str
    user_id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[str] = None
    created: str = None
    modified: str = None
    completed: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.priority, str):
            self.priority = TaskPriority(self.priority)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        if isinstance(data.get('status'), str):
            data['status'] = TaskStatus(data['status'])
        if isinstance(data.get('priority'), str):
            data['priority'] = TaskPriority(data['priority'])
        return cls(**data)


@dataclass
class ScheduleEvent:
    """Schedule event model."""
    event_id: str
    user_id: str
    title: str
    start_time: str
    end_time: str
    description: str = ""
    created: str = None
    modified: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.modified is None:
            self.modified = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleEvent':
        return cls(**data)


@dataclass
class Perceptron:
    """Perceptron model for puzzle detection."""
    perceptron_id: str
    weights: List[float]  # Serialized weight vector
    learning_rate: float
    confidence: float  # Current confidence level (0.0-1.0)
    accuracy: float  # Measured accuracy over time
    created: str = None
    last_used: str = None
    performance_history: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now().isoformat()
        if self.last_used is None:
            self.last_used = datetime.now().isoformat()
        if self.performance_history is None:
            self.performance_history = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Perceptron':
        return cls(**data)


class Database:
    """
    Simple JSON-based database for development.
    In production, this would use SQLite, PostgreSQL, etc.
    """
    
    def __init__(self, db_path: str = "fol_workbench.db"):
        self.db_path = db_path
        self.users: Dict[str, User] = {}
        self.axioms: Dict[str, Axiom] = {}
        self.theses: Dict[str, Thesis] = {}
        self.tasks: Dict[str, Task] = {}
        self.schedule_events: Dict[str, ScheduleEvent] = {}
        self.perceptrons: Dict[str, Perceptron] = {}
        self._load()
    
    def _load(self):
        """Load database from file."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.users = {uid: User.from_dict(u) for uid, u in data.get('users', {}).items()}
            self.axioms = {aid: Axiom.from_dict(a) for aid, a in data.get('axioms', {}).items()}
            self.theses = {tid: Thesis.from_dict(t) for tid, t in data.get('theses', {}).items()}
            self.tasks = {tid: Task.from_dict(t) for tid, t in data.get('tasks', {}).items()}
            self.schedule_events = {
                eid: ScheduleEvent.from_dict(e)
                for eid, e in data.get('schedule_events', {}).items()
            }
            self.perceptrons = {
                pid: Perceptron.from_dict(p)
                for pid, p in data.get('perceptrons', {}).items()
            }
        except FileNotFoundError:
            pass  # Start with empty database
    
    def _save(self):
        """Save database to file."""
        data = {
            'users': {uid: u.to_dict() for uid, u in self.users.items()},
            'axioms': {aid: a.to_dict() for aid, a in self.axioms.items()},
            'theses': {tid: t.to_dict() for tid, t in self.theses.items()},
            'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()},
            'schedule_events': {
                eid: e.to_dict() for eid, e in self.schedule_events.items()
            },
            'perceptrons': {
                pid: p.to_dict() for pid, p in self.perceptrons.items()
            }
        }
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # User methods
    def create_user(self, username: str, email: str, user_id: Optional[str] = None) -> User:
        import uuid
        if user_id is None:
            user_id = str(uuid.uuid4())
        user = User(user_id=user_id, username=username, email=email)
        self.users[user_id] = user
        self._save()
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)
    
    # Axiom methods
    def create_axiom(self, user_id: str, name: str, formula: str, **kwargs) -> Axiom:
        import uuid
        axiom_id = str(uuid.uuid4())
        axiom = Axiom(axiom_id=axiom_id, user_id=user_id, name=name, formula=formula, **kwargs)
        self.axioms[axiom_id] = axiom
        self._save()
        return axiom
    
    def get_axioms(self, user_id: Optional[str] = None) -> List[Axiom]:
        if user_id:
            return [a for a in self.axioms.values() if a.user_id == user_id]
        return list(self.axioms.values())
    
    # Thesis methods
    def create_thesis(self, user_id: str, title: str, statement: str, **kwargs) -> Thesis:
        import uuid
        thesis_id = str(uuid.uuid4())
        thesis = Thesis(thesis_id=thesis_id, user_id=user_id, title=title, statement=statement, **kwargs)
        self.theses[thesis_id] = thesis
        self._save()
        return thesis
    
    def get_theses(self, user_id: Optional[str] = None) -> List[Thesis]:
        if user_id:
            return [t for t in self.theses.values() if t.user_id == user_id]
        return list(self.theses.values())
    
    # Task methods
    def create_task(self, user_id: str, title: str, **kwargs) -> Task:
        import uuid
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, user_id=user_id, title=title, **kwargs)
        self.tasks[task_id] = task
        self._save()
        return task
    
    def get_tasks(self, user_id: Optional[str] = None, status: Optional[TaskStatus] = None) -> List[Task]:
        tasks = list(self.tasks.values())
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks
    
    # Schedule methods
    def create_schedule_event(self, user_id: str, title: str, start_time: str, end_time: str, **kwargs) -> ScheduleEvent:
        import uuid
        event_id = str(uuid.uuid4())
        event = ScheduleEvent(event_id=event_id, user_id=user_id, title=title, start_time=start_time, end_time=end_time, **kwargs)
        self.schedule_events[event_id] = event
        self._save()
        return event
    
    def get_schedule_events(self, user_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[ScheduleEvent]:
        events = list(self.schedule_events.values())
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if start_date:
            events = [e for e in events if e.start_time >= start_date]
        if end_date:
            events = [e for e in events if e.start_time <= end_date]
        return events
