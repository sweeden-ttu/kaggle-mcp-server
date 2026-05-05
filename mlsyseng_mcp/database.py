"""SQLite database for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
)


class MoEDatabase:
    """SQLite-backed storage for extracted chapters, experts, and loop state."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id   TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    source_path  TEXT,
                    markdown     TEXT,
                    concepts     TEXT,  -- JSON list of concept strings
                    extracted_at TEXT,
                    metadata     TEXT   -- JSON blob
                );

                CREATE TABLE IF NOT EXISTS experts (
                    expert_name  TEXT PRIMARY KEY,
                    slug         TEXT UNIQUE NOT NULL,
                    chapter_id   TEXT,
                    capabilities TEXT,  -- JSON list
                    skills       TEXT,  -- JSON list of skill paths
                    strategy     TEXT,
                    formula      TEXT,  -- JSON object
                    loop_config  TEXT,  -- JSON object
                    created_at   TEXT,
                    metadata     TEXT   -- JSON blob
                );

                CREATE TABLE IF NOT EXISTS loop_states (
                    state_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition  TEXT NOT NULL,
                    iteration    INTEGER NOT NULL,
                    state_vector TEXT,   -- JSON list of floats
                    metrics      TEXT,   -- JSON object
                    converged    INTEGER DEFAULT 0,
                    created_at   TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    job_id       TEXT PRIMARY KEY,
                    status       TEXT DEFAULT 'pending',
                    total_files  INTEGER DEFAULT 0,
                    processed    INTEGER DEFAULT 0,
                    errors       TEXT,  -- JSON list
                    started_at   TEXT,
                    finished_at  TEXT
                );
                """
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Chapter CRUD ──────────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_id: str,
        title: str,
        source_path: str,
        markdown: str,
        concepts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chapters (chapter_id, title, source_path, markdown, concepts, extracted_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_id) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown=excluded.markdown,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    metadata=excluded.metadata
                """,
                (
                    chapter_id,
                    title,
                    source_path,
                    markdown,
                    json.dumps(concepts),
                    datetime.now().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )

    def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_id"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Expert CRUD ───────────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experts
                    (expert_name, slug, chapter_id, capabilities, skills,
                     strategy, formula, loop_config, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name) DO UPDATE SET
                    slug=excluded.slug,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config,
                    metadata=excluded.metadata
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
                    datetime.now().isoformat(),
                    json.dumps(expert.get("metadata", {})),
                ),
            )

    def get_expert(self, expert_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Loop state ────────────────────────────────────────────────────

    def save_loop_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        metrics: Dict[str, Any],
        converged: bool = False,
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO loop_states (competition, iteration, state_vector, metrics, converged, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    json.dumps(metrics),
                    int(converged),
                    datetime.now().isoformat(),
                ),
            )
            return cursor.lastrowid

    def get_loop_states(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM loop_states WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_latest_loop_state(self, competition: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM loop_states WHERE competition = ? ORDER BY iteration DESC LIMIT 1",
            (competition,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # ── Extraction jobs ───────────────────────────────────────────────

    def create_extraction_job(self, job_id: str, total_files: int) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO extraction_jobs (job_id, status, total_files, processed, errors, started_at)
                VALUES (?, 'running', ?, 0, '[]', ?)
                """,
                (job_id, total_files, datetime.now().isoformat()),
            )

    def update_extraction_job(
        self,
        job_id: str,
        processed: Optional[int] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self.conn:
            if processed is not None:
                self.conn.execute(
                    "UPDATE extraction_jobs SET processed = ? WHERE job_id = ?",
                    (processed, job_id),
                )
            if status is not None:
                self.conn.execute(
                    "UPDATE extraction_jobs SET status = ? WHERE job_id = ?",
                    (status, job_id),
                )
                if status in ("completed", "failed"):
                    self.conn.execute(
                        "UPDATE extraction_jobs SET finished_at = ? WHERE job_id = ?",
                        (datetime.now().isoformat(), job_id),
                    )
            if error is not None:
                row = self.conn.execute(
                    "SELECT errors FROM extraction_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                errors = json.loads(row["errors"]) if row else []
                errors.append(error)
                self.conn.execute(
                    "UPDATE extraction_jobs SET errors = ? WHERE job_id = ?",
                    (json.dumps(errors), job_id),
                )

    def get_extraction_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM extraction_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        chapters = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        experts = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        loop_states = self.conn.execute("SELECT COUNT(*) FROM loop_states").fetchone()[0]
        return {
            "chapters": chapters,
            "experts": experts,
            "loop_states": loop_states,
            "db_path": self.db_path,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("concepts", "capabilities", "skills", "errors"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        for key in ("formula", "loop_config", "metrics", "metadata"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        if "state_vector" in d and isinstance(d["state_vector"], str):
            try:
                d["state_vector"] = json.loads(d["state_vector"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
