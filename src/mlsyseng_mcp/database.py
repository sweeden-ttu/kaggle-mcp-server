"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


class MLSysEngDB:
    """SQLite database for storing extracted knowledge, experts, and state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_num INTEGER UNIQUE,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    extracted_at TEXT,
                    page_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER REFERENCES chapters(id),
                    concept_name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    importance REAL DEFAULT 0.5,
                    UNIQUE(chapter_id, concept_name)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,
                    skills TEXT,
                    strategy TEXT,
                    formula TEXT,
                    loop_config TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS competition_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    expert_id INTEGER REFERENCES experts(id),
                    state_vector TEXT,
                    iteration INTEGER DEFAULT 0,
                    score REAL,
                    notebook_path TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS convergence_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT NOT NULL,
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    timestamp TEXT,
                    UNIQUE(competition, iteration)
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_chapter ON experts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_entries_competition ON competition_entries(competition);
                CREATE INDEX IF NOT EXISTS idx_convergence_competition ON convergence_states(competition);
            """)

    def upsert_chapter(
        self,
        chapter_num: int,
        title: str,
        source_path: str,
        markdown_content: str = "",
        page_count: int = 0,
        status: str = "extracted",
    ) -> int:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO chapters (chapter_num, title, source_path, markdown_content, extracted_at, page_count, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_num) DO UPDATE SET
                     title=excluded.title,
                     source_path=excluded.source_path,
                     markdown_content=excluded.markdown_content,
                     extracted_at=excluded.extracted_at,
                     page_count=excluded.page_count,
                     status=excluded.status""",
                (chapter_num, title, source_path, markdown_content, datetime.now().isoformat(), page_count, status),
            )
            row = conn.execute("SELECT id FROM chapters WHERE chapter_num=?", (chapter_num,)).fetchone()
            return row["id"]

    def get_chapter(self, chapter_num: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM chapters WHERE chapter_num=?", (chapter_num,)).fetchone()
            return dict(row) if row else None

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_num").fetchall()
            return [dict(r) for r in rows]

    def add_concept(
        self, chapter_id: int, concept_name: str, description: str = "", category: str = "general", importance: float = 0.5
    ) -> int:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO concepts (chapter_id, concept_name, description, category, importance)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_id, concept_name) DO UPDATE SET
                     description=excluded.description,
                     category=excluded.category,
                     importance=excluded.importance""",
                (chapter_id, concept_name, description, category, importance),
            )
            row = conn.execute(
                "SELECT id FROM concepts WHERE chapter_id=? AND concept_name=?", (chapter_id, concept_name)
            ).fetchone()
            return row["id"]

    def get_concepts_for_chapter(self, chapter_id: int) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM concepts WHERE chapter_id=? ORDER BY importance DESC", (chapter_id,)).fetchall()
            return [dict(r) for r in rows]

    def upsert_expert(
        self,
        expert_name: str,
        slug: str,
        chapter_id: int,
        capabilities: List[str],
        skills: List[str],
        strategy: str,
        formula: Dict[str, Any],
        loop_config: Dict[str, Any],
    ) -> int:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(slug) DO UPDATE SET
                     expert_name=excluded.expert_name,
                     chapter_id=excluded.chapter_id,
                     capabilities=excluded.capabilities,
                     skills=excluded.skills,
                     strategy=excluded.strategy,
                     formula=excluded.formula,
                     loop_config=excluded.loop_config""",
                (
                    expert_name, slug, chapter_id,
                    json.dumps(capabilities), json.dumps(skills),
                    strategy, json.dumps(formula), json.dumps(loop_config),
                    datetime.now().isoformat(),
                ),
            )
            row = conn.execute("SELECT id FROM experts WHERE slug=?", (slug,)).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM experts WHERE slug=?", (slug,)).fetchone()
            if row:
                d = dict(row)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(field):
                        d[field] = json.loads(d[field])
                return d
            return None

    def get_all_experts(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(field):
                        d[field] = json.loads(d[field])
                result.append(d)
            return result

    def add_competition_entry(
        self, competition: str, expert_id: int, state_vector: List[float], iteration: int, score: float, notebook_path: str
    ) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO competition_entries (competition, expert_id, state_vector, iteration, score, notebook_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (competition, expert_id, json.dumps(state_vector), iteration, score, notebook_path, datetime.now().isoformat()),
            )
            return cursor.lastrowid

    def save_convergence_state(
        self, competition: str, iteration: int, state_vector: List[float], l2_norm: float, converged: bool
    ):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO convergence_states (competition, iteration, state_vector, l2_norm, converged, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(competition, iteration) DO UPDATE SET
                     state_vector=excluded.state_vector,
                     l2_norm=excluded.l2_norm,
                     converged=excluded.converged,
                     timestamp=excluded.timestamp""",
                (competition, iteration, json.dumps(state_vector), l2_norm, int(converged), datetime.now().isoformat()),
            )

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM convergence_states WHERE competition=? ORDER BY iteration", (competition,)
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d.get("state_vector"):
                    d["state_vector"] = json.loads(d["state_vector"])
                result.append(d)
            return result

    def get_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            extracted = conn.execute("SELECT COUNT(*) as c FROM chapters WHERE status='extracted'").fetchone()["c"]
            concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
            experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            entries = conn.execute("SELECT COUNT(*) as c FROM competition_entries").fetchone()["c"]
            return {
                "chapters_total": chapters,
                "chapters_extracted": extracted,
                "concepts": concepts,
                "experts": experts,
                "competition_entries": entries,
            }
