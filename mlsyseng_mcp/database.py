"""SQLite database operations for MLSysEng MoE knowledge storage.

Stores extracted chapter content, expert definitions, and extraction metadata.
"""

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


class MLSysEngDatabase:
    """SQLite-backed storage for extracted ML knowledge and expert definitions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_schema(self):
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id   TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    folder_path  TEXT,
                    pdf_path     TEXT,
                    content_md   TEXT,
                    concepts     TEXT,   -- JSON list of extracted concepts
                    extracted_at TEXT,
                    metadata     TEXT    -- JSON blob
                );

                CREATE TABLE IF NOT EXISTS experts (
                    expert_name  TEXT PRIMARY KEY,
                    slug         TEXT UNIQUE NOT NULL,
                    chapter_id   TEXT REFERENCES chapters(chapter_id),
                    capabilities TEXT,   -- JSON list
                    skills       TEXT,   -- JSON list of skill paths
                    strategy     TEXT,
                    formula      TEXT,   -- JSON object
                    loop_config  TEXT,   -- JSON object
                    created_at   TEXT,
                    metadata     TEXT    -- JSON blob
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id   TEXT,
                    event        TEXT,
                    detail       TEXT,
                    timestamp    TEXT
                );

                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition  TEXT NOT NULL,
                    iteration    INTEGER NOT NULL,
                    state_vector TEXT,   -- JSON list of floats
                    metrics      TEXT,   -- JSON object
                    timestamp    TEXT
                );
                """
            )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Chapter operations ---

    def upsert_chapter(
        self,
        chapter_id: str,
        title: str,
        folder_path: str = "",
        pdf_path: str = "",
        content_md: str = "",
        concepts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chapters
                    (chapter_id, title, folder_path, pdf_path, content_md,
                     concepts, extracted_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_id) DO UPDATE SET
                    title=excluded.title,
                    folder_path=excluded.folder_path,
                    pdf_path=excluded.pdf_path,
                    content_md=excluded.content_md,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    metadata=excluded.metadata
                """,
                (
                    chapter_id,
                    title,
                    folder_path,
                    pdf_path,
                    content_md,
                    json.dumps(concepts or []),
                    now,
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

    def chapter_exists(self, chapter_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        return row is not None

    # --- Expert operations ---

    def upsert_expert(
        self,
        expert_name: str,
        slug: str,
        chapter_id: str = "",
        capabilities: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        strategy: str = "",
        formula: Optional[Dict[str, Any]] = None,
        loop_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        now = datetime.utcnow().isoformat()
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
                    expert_name,
                    slug,
                    chapter_id,
                    json.dumps(capabilities or []),
                    json.dumps(skills or []),
                    strategy,
                    json.dumps(formula or {}),
                    json.dumps(loop_config or {}),
                    now,
                    json.dumps(metadata or {}),
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

    # --- Extraction log ---

    def log_extraction_event(
        self, chapter_id: str, event: str, detail: str = ""
    ):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO extraction_log (chapter_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (chapter_id, event, detail, now),
            )

    def get_extraction_log(
        self, chapter_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if chapter_id:
            rows = self.conn.execute(
                "SELECT * FROM extraction_log WHERE chapter_id = ? ORDER BY timestamp",
                (chapter_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM extraction_log ORDER BY timestamp"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # --- State snapshots ---

    def save_state_snapshot(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO state_snapshots
                    (competition, iteration, state_vector, metrics, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    json.dumps(metrics or {}),
                    now,
                ),
            )

    def get_state_snapshots(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM state_snapshots WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        chapter_count = self.conn.execute(
            "SELECT COUNT(*) FROM chapters"
        ).fetchone()[0]
        expert_count = self.conn.execute(
            "SELECT COUNT(*) FROM experts"
        ).fetchone()[0]
        log_count = self.conn.execute(
            "SELECT COUNT(*) FROM extraction_log"
        ).fetchone()[0]
        snapshot_count = self.conn.execute(
            "SELECT COUNT(*) FROM state_snapshots"
        ).fetchone()[0]
        return {
            "chapters": chapter_count,
            "experts": expert_count,
            "extraction_log_entries": log_count,
            "state_snapshots": snapshot_count,
            "db_path": self.db_path,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("concepts", "capabilities", "skills", "formula",
                     "loop_config", "metadata", "state_vector", "metrics"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
