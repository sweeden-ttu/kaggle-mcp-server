"""
Task Engine and Planner

Manages tasks, schedules, and calendar events with automatic planning.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

from .database import Task, TaskStatus, TaskPriority, ScheduleEvent, Database


class TaskEngine:
    """Task engine for managing and planning tasks."""
    
    def __init__(self, database: Database):
        self.db = database
    
    def create_task(
        self,
        user_id: str,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: Optional[str] = None
    ) -> Task:
        """Create a new task."""
        return self.db.create_task(
            user_id=user_id,
            title=title,
            description=description,
            priority=priority,
            due_date=due_date
        )
    
    def get_tasks(
        self,
        user_id: str,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None
    ) -> List[Task]:
        """Get tasks for a user with optional filters."""
        tasks = self.db.get_tasks(user_id=user_id, status=status)
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        return sorted(tasks, key=lambda t: (
            t.priority.value if t.priority else 'medium',
            t.due_date or '9999-12-31'
        ))
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        task = self.db.tasks.get(task_id)
        if task:
            task.status = status
            task.modified = datetime.now().isoformat()
            if status == TaskStatus.COMPLETED:
                task.completed = datetime.now().isoformat()
            self.db._save()
            return True
        return False
    
    def schedule_task(self, task_id: str, start_time: str, end_time: str) -> Optional[ScheduleEvent]:
        """Schedule a task as a calendar event."""
        task = self.db.tasks.get(task_id)
        if not task:
            return None
        
        event = self.db.create_schedule_event(
            user_id=task.user_id,
            title=task.title,
            start_time=start_time,
            end_time=end_time,
            description=task.description,
            metadata={"task_id": task_id}
        )
        return event
    
    def get_upcoming_tasks(self, user_id: str, days: int = 7) -> List[Task]:
        """Get tasks due in the next N days."""
        tasks = self.get_tasks(user_id, status=TaskStatus.PENDING)
        cutoff = (datetime.now() + timedelta(days=days)).isoformat()
        return [t for t in tasks if t.due_date and t.due_date <= cutoff]
    
    def get_overdue_tasks(self, user_id: str) -> List[Task]:
        """Get overdue tasks."""
        tasks = self.get_tasks(user_id, status=TaskStatus.PENDING)
        now = datetime.now().isoformat()
        return [t for t in tasks if t.due_date and t.due_date < now]
    
    def auto_schedule_tasks(self, user_id: str) -> List[ScheduleEvent]:
        """Automatically schedule tasks based on priority and due dates."""
        tasks = self.get_upcoming_tasks(user_id, days=30)
        events = []
        current_time = datetime.now()
        
        for task in sorted(tasks, key=lambda t: (
            t.priority.value if t.priority else 'medium',
            t.due_date or '9999-12-31'
        )):
            # Estimate duration (1 hour default, 2 hours for high priority)
            duration_hours = 2 if task.priority == TaskPriority.HIGH else 1
            
            # Schedule at least 1 day before due date
            if task.due_date:
                due = datetime.fromisoformat(task.due_date)
                start = due - timedelta(days=1, hours=duration_hours)
                if start < current_time:
                    start = current_time
            else:
                start = current_time
            
            end = start + timedelta(hours=duration_hours)
            
            event = self.schedule_task(
                task.task_id,
                start.isoformat(),
                end.isoformat()
            )
            if event:
                events.append(event)
                current_time = end  # Next task starts after this one
        
        return events


class Calendar:
    """Calendar manager for schedule events."""
    
    def __init__(self, database: Database):
        self.db = database
    
    def create_event(
        self,
        user_id: str,
        title: str,
        start_time: str,
        end_time: str,
        description: str = ""
    ) -> ScheduleEvent:
        """Create a calendar event."""
        return self.db.create_schedule_event(
            user_id=user_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            description=description
        )
    
    def get_events(
        self,
        user_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[ScheduleEvent]:
        """Get calendar events for a date range."""
        return self.db.get_schedule_events(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_today_events(self, user_id: str) -> List[ScheduleEvent]:
        """Get events for today."""
        today = datetime.now().date().isoformat()
        tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
        return self.get_events(user_id, start_date=today, end_date=tomorrow)
    
    def get_week_events(self, user_id: str) -> List[ScheduleEvent]:
        """Get events for the current week."""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=7)
        return self.get_events(
            user_id,
            start_date=week_start.isoformat(),
            end_date=week_end.isoformat()
        )


class TaskPlanner:
    """High-level task planning interface."""
    
    def __init__(self, database: Database):
        self.db = database
        self.task_engine = TaskEngine(database)
        self.calendar = Calendar(database)
    
    def plan_week(self, user_id: str) -> Dict[str, Any]:
        """Plan tasks for the week."""
        tasks = self.task_engine.get_upcoming_tasks(user_id, days=7)
        events = self.calendar.get_week_events(user_id)
        
        # Auto-schedule high-priority tasks
        auto_scheduled = self.task_engine.auto_schedule_tasks(user_id)
        
        return {
            "tasks": [t.to_dict() for t in tasks],
            "events": [e.to_dict() for e in events],
            "auto_scheduled": [e.to_dict() for e in auto_scheduled],
            "overdue": [t.to_dict() for t in self.task_engine.get_overdue_tasks(user_id)]
        }
