"""SQLite database operations for MLSysEng MoE system.

Stores extracted chapter content, expert definitions, extraction status,
and state convergence history.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
    )


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    folder_path TEXT,
                    pdf_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array
                    extracted_at TEXT,
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,  -- JSON array
                    skills TEXT,        -- JSON array
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    message TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS convergence_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT,   -- JSON array of floats
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    metadata TEXT,       -- JSON object
                    recorded_at TEXT
                );

                CREATE TABLE IF NOT EXISTS competition_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    expert_slug TEXT NOT NULL,
                    notebook_path TEXT,
                    skills_used TEXT,    -- JSON array
                    score REAL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    updated_at TEXT
                );
            """)

    # -- Chapter operations --

    def upsert_chapter(
        self,
        chapter_name: str,
        folder_path: str = "",
        pdf_path: str = "",
        markdown_content: str = "",
        concepts: Optional[List[str]] = None,
        status: str = "extracted",
    ) -> int:
        now = datetime.utcnow().isoformat()
        concepts_json = json.dumps(concepts or [])
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO chapters
                       (chapter_name, folder_path, pdf_path, markdown_content,
                        concepts, extracted_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_name) DO UPDATE SET
                       folder_path=excluded.folder_path,
                       pdf_path=excluded.pdf_path,
                       markdown_content=excluded.markdown_content,
                       concepts=excluded.concepts,
                       extracted_at=excluded.extracted_at,
                       status=excluded.status
                """,
                (chapter_name, folder_path, pdf_path, markdown_content,
                 concepts_json, now, status),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_name=?",
                (chapter_name,),
            ).fetchone()
            return row["id"]

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_name=?",
                (chapter_name,),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["concepts"] = json.loads(d.get("concepts") or "[]")
            return d

    def list_chapters(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_name"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["concepts"] = json.loads(d.get("concepts") or "[]")
                result.append(d)
            return result

    # -- Expert operations --

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO experts
                       (expert_name, slug, chapter_id, capabilities, skills,
                        strategy, formula, loop_config, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(slug) DO UPDATE SET
                       expert_name=excluded.expert_name,
                       chapter_id=excluded.chapter_id,
                       capabilities=excluded.capabilities,
                       skills=excluded.skills,
                       strategy=excluded.strategy,
                       formula=excluded.formula,
                       loop_config=excluded.loop_config,
                       updated_at=excluded.updated_at
                """,
                (
                    expert["expert_name"],
                    expert["slug"],
                    expert.get("chapter_id"),
                    json.dumps(expert.get("capabilities", [])),
                    json.dumps(expert.get("skills", [])),
                    expert.get("strategy", ""),
                    json.dumps(expert.get("formula", {})),
                    json.dumps(expert.get("loop_config", {})),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE slug=?",
                (expert["slug"],),
            ).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug=?", (slug,)
            ).fetchone()
            if row is None:
                return None
            return self._deserialize_expert(row)

    def list_experts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            return [self._deserialize_expert(r) for r in rows]

    def _deserialize_expert(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("capabilities", "skills"):
            d[field] = json.loads(d.get(field) or "[]")
        for field in ("formula", "loop_config"):
            d[field] = json.loads(d.get(field) or "{}")
        return d

    # -- Extraction status --

    def log_extraction(
        self,
        chapter_name: str,
        stage: str,
        status: str = "started",
        message: str = "",
    ):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO extraction_status
                       (chapter_name, stage, status, message, started_at)
                   VALUES (?, ?, ?, ?, ?)
                """,
                (chapter_name, stage, status, message, now),
            )

    def complete_extraction(self, chapter_name: str, stage: str, message: str = ""):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE extraction_status
                   SET status='completed', message=?, completed_at=?
                   WHERE chapter_name=? AND stage=? AND status='started'
                """,
                (message, now, chapter_name, stage),
            )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM extraction_status ORDER BY started_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Convergence history --

    def record_convergence(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
        metadata: Optional[Dict] = None,
    ):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO convergence_history
                       (competition, iteration, state_vector, l2_norm,
                        converged, metadata, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    int(converged),
                    json.dumps(metadata or {}),
                    now,
                ),
            )

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM convergence_history
                   WHERE competition=? ORDER BY iteration""",
                (competition,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["state_vector"] = json.loads(d.get("state_vector") or "[]")
                d["metadata"] = json.loads(d.get("metadata") or "{}")
                result.append(d)
            return result

    # -- Competition entries --

    def upsert_entry(self, entry: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO competition_entries
                       (competition, expert_slug, notebook_path,
                        skills_used, score, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING
                """,
                (
                    entry["competition"],
                    entry["expert_slug"],
                    entry.get("notebook_path", ""),
                    json.dumps(entry.get("skills_used", [])),
                    entry.get("score"),
                    entry.get("status", "pending"),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """SELECT id FROM competition_entries
                   WHERE competition=? AND expert_slug=?
                   ORDER BY id DESC LIMIT 1""",
                (entry["competition"], entry["expert_slug"]),
            ).fetchone()
            return row["id"] if row else 0

    def list_entries(self, competition: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM competition_entries
                   WHERE competition=? ORDER BY created_at""",
                (competition,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["skills_used"] = json.loads(d.get("skills_used") or "[]")
                result.append(d)
            return result

    # -- Stats --

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            entries = conn.execute(
                "SELECT COUNT(*) as c FROM competition_entries"
            ).fetchone()["c"]
            extractions = conn.execute(
                "SELECT COUNT(*) as c FROM extraction_status WHERE status='completed'"
            ).fetchone()["c"]
            return {
                "total_chapters": chapters,
                "total_experts": experts,
                "total_entries": entries,
                "completed_extractions": extractions,
                "db_path": self.db_path,
            }
