"""SQLite database operations for MLSysEng MoE system."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class Database:
    """SQLite store for extracted chapters, experts, and extraction status."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = _ensure_parent(db_path or DEFAULT_DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chapters (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_num INTEGER NOT NULL,
                title       TEXT NOT NULL,
                source_path TEXT NOT NULL,
                content_md  TEXT,
                concepts    TEXT,          -- JSON list
                extracted_at TEXT,
                UNIQUE(chapter_num)
            );

            CREATE TABLE IF NOT EXISTS experts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                slug         TEXT NOT NULL UNIQUE,
                expert_name  TEXT NOT NULL,
                chapter_num  INTEGER,
                capabilities TEXT,         -- JSON list
                skills       TEXT,         -- JSON list
                strategy     TEXT,
                formula      TEXT,         -- JSON object
                loop_config  TEXT,         -- JSON object
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                FOREIGN KEY (chapter_num) REFERENCES chapters(chapter_num)
            );

            CREATE TABLE IF NOT EXISTS extraction_jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_num INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                started_at  TEXT,
                finished_at TEXT,
                error       TEXT,
                UNIQUE(chapter_num)
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Chapters
    # ------------------------------------------------------------------

    def upsert_chapter(
        self,
        chapter_num: int,
        title: str,
        source_path: str,
        content_md: Optional[str] = None,
        concepts: Optional[List[str]] = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO chapters (chapter_num, title, source_path, content_md, concepts, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_num) DO UPDATE SET
                title = excluded.title,
                source_path = excluded.source_path,
                content_md = COALESCE(excluded.content_md, content_md),
                concepts = COALESCE(excluded.concepts, concepts),
                extracted_at = excluded.extracted_at
            """,
            (chapter_num, title, source_path, content_md, json.dumps(concepts or []), now),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_chapter(self, chapter_num: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_num = ?", (chapter_num,)
        ).fetchone()
        return dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_num"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Experts
    # ------------------------------------------------------------------

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO experts
                (slug, expert_name, chapter_num, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                expert_name  = excluded.expert_name,
                chapter_num  = excluded.chapter_num,
                capabilities = excluded.capabilities,
                skills       = excluded.skills,
                strategy     = excluded.strategy,
                formula      = excluded.formula,
                loop_config  = excluded.loop_config,
                updated_at   = excluded.updated_at
            """,
            (
                expert["slug"],
                expert["expert_name"],
                expert.get("chapter_num"),
                json.dumps(expert.get("capabilities", [])),
                json.dumps(expert.get("skills", [])),
                expert.get("strategy", ""),
                json.dumps(expert.get("formula", {})),
                json.dumps(expert.get("loop_config", {})),
                now,
                now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experts ORDER BY slug").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(field):
                    d[field] = json.loads(d[field])
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Extraction jobs
    # ------------------------------------------------------------------

    def set_extraction_status(
        self,
        chapter_num: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        started = now if status == "running" else None
        finished = now if status in ("done", "error") else None
        self.conn.execute(
            """
            INSERT INTO extraction_jobs (chapter_num, status, started_at, finished_at, error)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chapter_num) DO UPDATE SET
                status      = excluded.status,
                started_at  = COALESCE(excluded.started_at, started_at),
                finished_at = excluded.finished_at,
                error       = excluded.error
            """,
            (chapter_num, status, started, finished, error),
        )
        self.conn.commit()

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM extraction_jobs ORDER BY chapter_num"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
