"""SQLite database operations for MLSysEng MoE system.

Stores extracted chapter content, expert definitions, extraction status,
and competition entry state.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db")


class Database:
    """Thread-safe SQLite database for MLSysEng MoE."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_schema(self):
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_markdown TEXT,
                    concepts TEXT,
                    extracted_at TIMESTAMP,
                    embedding_id TEXT,
                    UNIQUE(chapter_number)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT NOT NULL,
                    skills TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    formula TEXT NOT NULL,
                    loop_config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS extraction_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    pages_extracted INTEGER DEFAULT 0,
                    UNIQUE(chapter_number)
                );

                CREATE TABLE IF NOT EXISTS competition_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT NOT NULL,
                    metric_values TEXT NOT NULL,
                    converged INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(competition, iteration)
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(chapter_number);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_competition_state ON competition_state(competition, iteration);
            """)

    def upsert_chapter(
        self,
        chapter_number: str,
        title: str,
        source_path: str,
        content_markdown: Optional[str] = None,
        concepts: Optional[List[str]] = None,
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                """INSERT INTO chapters (chapter_number, title, source_path, content_markdown, concepts, extracted_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_number) DO UPDATE SET
                       title=excluded.title,
                       source_path=excluded.source_path,
                       content_markdown=COALESCE(excluded.content_markdown, chapters.content_markdown),
                       concepts=COALESCE(excluded.concepts, chapters.concepts),
                       extracted_at=COALESCE(excluded.extracted_at, chapters.extracted_at)
                """,
                (
                    chapter_number,
                    title,
                    source_path,
                    content_markdown,
                    json.dumps(concepts) if concepts else None,
                    datetime.utcnow().isoformat() if content_markdown else None,
                ),
            )
            return cursor.lastrowid

    def get_chapter(self, chapter_number: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM chapters WHERE chapter_number = ?", (chapter_number,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("concepts"):
            result["concepts"] = json.loads(result["concepts"])
        return result

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_number"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("concepts"):
                d["concepts"] = json.loads(d["concepts"])
            results.append(d)
        return results

    def update_chapter_embedding(self, chapter_number: str, embedding_id: str):
        with self._conn:
            self._conn.execute(
                "UPDATE chapters SET embedding_id = ? WHERE chapter_number = ?",
                (embedding_id, chapter_number),
            )

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        slug = expert_data["slug"]
        with self._conn:
            cursor = self._conn.execute(
                """INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    expert_data["expert_name"],
                    slug,
                    expert_data.get("chapter_id"),
                    json.dumps(expert_data["capabilities"]),
                    json.dumps(expert_data["skills"]),
                    expert_data["strategy"],
                    json.dumps(expert_data["formula"]),
                    json.dumps(expert_data["loop_config"]),
                    datetime.utcnow().isoformat(),
                ),
            )
            return cursor.lastrowid

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return self._deserialize_expert(dict(row))

    def get_all_experts(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        return [self._deserialize_expert(dict(r)) for r in rows]

    def _deserialize_expert(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field) and isinstance(d[field], str):
                d[field] = json.loads(d[field])
        return d

    def update_extraction_status(
        self,
        chapter_number: str,
        status: str,
        error_message: Optional[str] = None,
        pages_extracted: int = 0,
    ):
        now = datetime.utcnow().isoformat()
        with self._conn:
            self._conn.execute(
                """INSERT INTO extraction_status (chapter_number, status, started_at, completed_at, error_message, pages_extracted)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_number) DO UPDATE SET
                       status=excluded.status,
                       started_at=CASE WHEN excluded.status='extracting' THEN ? ELSE extraction_status.started_at END,
                       completed_at=CASE WHEN excluded.status IN ('completed','failed') THEN ? ELSE NULL END,
                       error_message=excluded.error_message,
                       pages_extracted=excluded.pages_extracted
                """,
                (chapter_number, status, now if status == "extracting" else None,
                 now if status in ("completed", "failed") else None,
                 error_message, pages_extracted, now, now),
            )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM extraction_status ORDER BY chapter_number"
        ).fetchall()
        return [dict(r) for r in rows]

    def save_competition_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        metric_values: Dict[str, float],
        converged: bool = False,
    ):
        with self._conn:
            self._conn.execute(
                """INSERT INTO competition_state (competition, iteration, state_vector, metric_values, converged)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(competition, iteration) DO UPDATE SET
                       state_vector=excluded.state_vector,
                       metric_values=excluded.metric_values,
                       converged=excluded.converged
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    json.dumps(metric_values),
                    int(converged),
                ),
            )

    def get_competition_states(self, competition: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM competition_state WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["state_vector"] = json.loads(d["state_vector"])
            d["metric_values"] = json.loads(d["metric_values"])
            d["converged"] = bool(d["converged"])
            results.append(d)
        return results

    def get_stats(self) -> Dict[str, Any]:
        chapters = self._conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
        extracted = self._conn.execute(
            "SELECT COUNT(*) as c FROM chapters WHERE content_markdown IS NOT NULL"
        ).fetchone()["c"]
        experts = self._conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
        competitions = self._conn.execute(
            "SELECT COUNT(DISTINCT competition) as c FROM competition_state"
        ).fetchone()["c"]
        return {
            "total_chapters": chapters,
            "extracted_chapters": extracted,
            "total_experts": experts,
            "active_competitions": competitions,
        }

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
