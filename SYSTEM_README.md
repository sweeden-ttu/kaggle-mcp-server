# FOL Workbench - Complete System Documentation

## Overview

The FOL Workbench is now a complete, integrated system with:

- **Database Models**: Users, Axioms, Theses, Tasks, Schedules
- **Task Engine**: Automated task planning and scheduling
- **Calendar System**: Event management and scheduling
- **Storage Backends**: Local, Network, Cloud, GitHub
- **Document Generator**: Beautiful HTML/Markdown documents
- **Web Scraper (Soup)**: BeautifulSoup-based web scraping
- **Pourage**: Content aggregation from multiple sources
- **Docker Support**: Containerized deployment
- **IKYKE Protocol**: Automated workflow execution

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FOL Workbench System                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Database   │  │ Task Engine  │  │   Calendar   │  │
│  │  (Users,     │  │   & Planner  │  │   Manager    │  │
│  │  Axioms,     │  │              │  │              │  │
│  │  Theses)     │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Storage    │  │  Document    │  │ Web Scraper  │  │
│  │   Manager    │  │  Generator   │  │  & Pourage   │  │
│  │              │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Logic Engine │  │ IKYKE Protocol│  │   PyQt6 UI   │  │
│  │    (Z3)      │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Database Tables

### Users Table
- `user_id`: Unique identifier
- `username`: User name
- `email`: Email address
- `created`, `modified`: Timestamps
- `metadata`: Additional data

### Axioms Table
- `axiom_id`: Unique identifier
- `user_id`: Owner
- `name`: Axiom name
- `formula`: FOL formula
- `description`: Description
- `tags`: Tags for organization
- `verified`: Verification status

### Theses Table
- `thesis_id`: Unique identifier
- `user_id`: Owner
- `title`: Thesis title
- `statement`: Logical statement
- `proof`: Proof text
- `status`: Draft/Published/etc.
- `axioms_used`: List of axiom IDs
- `tags`: Tags

### Tasks Table
- `task_id`: Unique identifier
- `user_id`: Owner
- `title`: Task title
- `description`: Description
- `status`: Pending/In Progress/Completed/etc.
- `priority`: Low/Medium/High/Critical
- `due_date`: Due date
- `tags`: Tags

### Schedule Events Table
- `event_id`: Unique identifier
- `user_id`: Owner
- `title`: Event title
- `start_time`, `end_time`: Time range
- `description`: Description

## Storage Backends

### Local Storage
```python
from fol_workbench.storage import LocalStorage

storage = LocalStorage(base_path="./storage/local")
storage.save("file.txt", b"content")
data = storage.load("file.txt")
```

### Network Storage
```python
from fol_workbench.storage import NetworkStorage

storage = NetworkStorage(
    network_path="//server/share",
    username="user",
    password="pass"
)
```

### Cloud Storage
```python
from fol_workbench.storage import CloudStorage

storage = CloudStorage(
    provider="s3",
    access_key="...",
    secret_key="...",
    bucket="my-bucket"
)
```

### GitHub Storage
```python
from fol_workbench.storage import GitHubStorage

storage = GitHubStorage(
    repo_url="https://github.com/user/repo",
    token="ghp_...",
    branch="main"
)
```

## Task Engine

### Create Task
```python
from fol_workbench.task_engine import TaskEngine
from fol_workbench.database import TaskPriority

task = task_engine.create_task(
    user_id="user123",
    title="Prove theorem X",
    priority=TaskPriority.HIGH,
    due_date="2024-12-31"
)
```

### Auto-Schedule Tasks
```python
events = task_engine.auto_schedule_tasks(user_id="user123")
# Automatically schedules tasks based on priority and due dates
```

### Get Upcoming Tasks
```python
tasks = task_engine.get_upcoming_tasks(user_id="user123", days=7)
```

## Calendar

### Create Event
```python
from fol_workbench.task_engine import Calendar

event = calendar.create_event(
    user_id="user123",
    title="Logic Workshop",
    start_time="2024-01-15T10:00:00",
    end_time="2024-01-15T12:00:00"
)
```

### Get Week Events
```python
events = calendar.get_week_events(user_id="user123")
```

## Document Generator

### Generate Thesis Document
```python
from fol_workbench.document_generator import DocumentGenerator

generator = DocumentGenerator()
document = generator.generate_thesis_document(thesis_data)
generator.save_document(document, "thesis.html")
```

### Generate Task Report
```python
document = generator.generate_task_report(tasks)
generator.save_document(document, "tasks.html")
```

## Web Scraper (Soup)

### Extract Formulas from Web
```python
from fol_workbench.web_scraper import WebScraper

scraper = WebScraper()
formulas = scraper.extract_formulas("https://example.com/logic")
```

### Extract Text and Links
```python
text = scraper.extract_text(url)
links = scraper.extract_links(url)
metadata = scraper.extract_metadata(url)
```

## Pourage (Content Aggregation)

### Aggregate Multiple Sources
```python
from fol_workbench.web_scraper import Pourage

pourage = Pourage()
pourage.add_web_source("https://example.com/page1")
pourage.add_web_source("https://example.com/page2")
pourage.add_file_source("./local_file.txt")

aggregated = pourage.aggregate()
unique_formulas = pourage.extract_unique_formulas()
```

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t fol-workbench .

# Run with docker-compose
docker-compose up -d

# Or run directly
docker run -v ./data:/app/data fol-workbench
```

### Docker Compose Services
- `fol-workbench`: Main application
- `postgres`: PostgreSQL database (optional)
- `redis`: Redis cache (optional)

### Environment Variables
- `DB_PATH`: Database file path
- `STORAGE_LOCAL_PATH`: Local storage path
- `STORAGE_NETWORK_PATH`: Network storage path

## System Integration

### Complete System Usage
```python
from fol_workbench.system_integration import FOLWorkbenchSystem

# Initialize system
system = FOLWorkbenchSystem()

# Create user
user = system.create_user("alice", "alice@example.com")

# Create axiom
axiom = system.create_axiom(
    user.user_id,
    "Modus Ponens",
    "Implies(And(p, Implies(p, q)), q)"
)

# Create thesis
thesis = system.create_thesis(
    user.user_id,
    "Theorem 1",
    "ForAll(x, P(x))"
)

# Create task
task = system.create_task(
    user.user_id,
    "Prove Theorem 1",
    priority=TaskPriority.HIGH
)

# Plan week
plan = system.plan_week(user.user_id)

# Generate documents
system.generate_thesis_document(thesis.thesis_id, "thesis.html")
system.generate_task_report(user.user_id, "tasks.html")

# Web scraping
formulas = system.scrape_and_extract_formulas("https://example.com")

# Aggregate sources
sources = [
    {"type": "web", "url": "https://example.com/page1"},
    {"type": "file", "path": "./local.txt"}
]
aggregated = system.aggregate_sources(sources)
```

## File Structure

```
fol-workbench/
├── src/fol_workbench/
│   ├── database.py              # Database models
│   ├── storage.py                # Storage backends
│   ├── task_engine.py            # Task engine & calendar
│   ├── document_generator.py     # Document generation
│   ├── web_scraper.py            # Web scraping & Pourage
│   ├── system_integration.py     # System integration
│   ├── logic_layer.py            # Z3 logic engine
│   ├── data_layer.py             # Data persistence
│   ├── ui_layer.py               # PyQt6 UI
│   ├── ikyke_format.py           # IKYKE protocol
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Features Summary

✅ **Database**: Users, Axioms, Theses, Tasks, Schedules  
✅ **Task Engine**: Automated planning and scheduling  
✅ **Calendar**: Event management  
✅ **Storage**: Local, Network, Cloud, GitHub  
✅ **Documents**: Beautiful HTML/Markdown generation  
✅ **Web Scraping**: BeautifulSoup integration  
✅ **Pourage**: Multi-source content aggregation  
✅ **Docker**: Containerized deployment  
✅ **IKYKE**: Automated workflow protocol  
✅ **Z3 Integration**: Logic validation and model finding  

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with Docker
docker-compose up -d
```

## Usage

```bash
# Run desktop application
python fol_workbench.py

# Or with Docker
docker-compose up
```

The system is now fully integrated and ready for production use!
