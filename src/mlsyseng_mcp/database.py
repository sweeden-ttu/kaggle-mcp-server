"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
)


class Database:
    """SQLite wrapper for chapter content, experts, and extraction state."""

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
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_name TEXT UNIQUE NOT NULL,
                    title       TEXT NOT NULL,
                    content_md  TEXT,
                    concepts    TEXT,  -- JSON array
                    pdf_path    TEXT,
                    extracted_at TEXT,
                    updated_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug        TEXT UNIQUE NOT NULL,
                    expert_name TEXT NOT NULL,
                    chapter_id  INTEGER REFERENCES chapters(id),
                    capabilities TEXT,  -- JSON array
                    skills      TEXT,  -- JSON array
                    strategy    TEXT,
                    formula     TEXT,  -- JSON object
                    loop_config TEXT,  -- JSON object
                    created_at  TEXT,
                    updated_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id  INTEGER REFERENCES chapters(id),
                    status      TEXT NOT NULL DEFAULT 'pending',
                    message     TEXT,
                    started_at  TEXT,
                    finished_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_folder ON chapters(folder_name);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                """
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Chapter operations ──

    def upsert_chapter(
        self,
        folder_name: str,
        title: str,
        content_md: str,
        concepts: List[str],
        pdf_path: str,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chapters (folder_name, title, content_md, concepts, pdf_path, extracted_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(folder_name) DO UPDATE SET
                    title=excluded.title,
                    content_md=excluded.content_md,
                    concepts=excluded.concepts,
                    pdf_path=excluded.pdf_path,
                    updated_at=excluded.updated_at
                """,
                (folder_name, title, content_md, json.dumps(concepts), pdf_path, now, now),
            )
        row = self.conn.execute(
            "SELECT id FROM chapters WHERE folder_name=?", (folder_name,)
        ).fetchone()
        return row["id"]

    def get_chapter(self, folder_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE folder_name=?", (folder_name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_chapter_by_id(self, chapter_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE id=?", (chapter_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM chapters ORDER BY folder_name").fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Expert operations ──

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        slug = expert_data["slug"]
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experts (slug, expert_name, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
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
                    slug,
                    expert_data["expert_name"],
                    expert_data.get("chapter_id"),
                    json.dumps(expert_data.get("capabilities", [])),
                    json.dumps(expert_data.get("skills", [])),
                    expert_data.get("strategy", ""),
                    json.dumps(expert_data.get("formula", {})),
                    json.dumps(expert_data.get("loop_config", {})),
                    now,
                    now,
                ),
            )
        row = self.conn.execute(
            "SELECT id FROM experts WHERE slug=?", (slug,)
        ).fetchone()
        return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug=?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return self._expert_row_to_dict(row)

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experts ORDER BY slug").fetchall()
        return [self._expert_row_to_dict(r) for r in rows]

    # ── Extraction log ──

    def log_extraction(self, chapter_id: int, status: str, message: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO extraction_log (chapter_id, status, message, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chapter_id,
                    status,
                    message,
                    now if status == "started" else None,
                    now if status in ("completed", "failed") else None,
                ),
            )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT c.folder_name, c.title, el.status, el.message, el.started_at, el.finished_at
            FROM extraction_log el
            JOIN chapters c ON c.id = el.chapter_id
            ORDER BY el.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        chapters = self.conn.execute("SELECT COUNT(*) AS cnt FROM chapters").fetchone()["cnt"]
        experts = self.conn.execute("SELECT COUNT(*) AS cnt FROM experts").fetchone()["cnt"]
        extractions = self.conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM extraction_log GROUP BY status"
        ).fetchall()
        return {
            "total_chapters": chapters,
            "total_experts": experts,
            "extraction_counts": {r["status"]: r["cnt"] for r in extractions},
        }

    # ── Helpers ──

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("concepts"):
            d["concepts"] = json.loads(d["concepts"])
        return d

    @staticmethod
    def _expert_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d
