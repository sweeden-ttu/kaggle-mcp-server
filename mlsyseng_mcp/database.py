"""SQLite database operations for MLSysEng MoE system.

Stores extracted chapter content, concepts, and expert metadata.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


class Database:
    """SQLite database for chapter content and expert metadata."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_num INTEGER UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array
                    extracted_at TEXT,
                    page_count INTEGER DEFAULT 0
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
                    chapter_num INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_num ON chapters(chapter_num);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_extraction_status ON extraction_status(chapter_num, status);
            """)
            conn.commit()
        finally:
            conn.close()

    def upsert_chapter(
        self,
        chapter_num: int,
        title: str,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
        page_count: int = 0,
    ) -> int:
        """Insert or update a chapter record. Returns the chapter id."""
        conn = self._connect()
        try:
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO chapters (chapter_num, title, source_path, markdown_content, concepts, extracted_at, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_num) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    page_count=excluded.page_count
                """,
                (chapter_num, title, source_path, markdown_content, json.dumps(concepts), now, page_count),
            )
            conn.commit()
            row = conn.execute("SELECT id FROM chapters WHERE chapter_num = ?", (chapter_num,)).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_chapter(self, chapter_num: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM chapters WHERE chapter_num = ?", (chapter_num,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            return d
        finally:
            conn.close()

    def list_chapters(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_num").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
                result.append(d)
            return result
        finally:
            conn.close()

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        """Insert or update an expert record. Returns expert id."""
        conn = self._connect()
        try:
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
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
                    expert_data["expert_name"],
                    expert_data["slug"],
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
            conn.commit()
            row = conn.execute("SELECT id FROM experts WHERE slug = ?", (expert_data["slug"],)).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
            if row is None:
                return None
            return self._parse_expert_row(row)
        finally:
            conn.close()

    def list_experts(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
            return [self._parse_expert_row(r) for r in rows]
        finally:
            conn.close()

    def _parse_expert_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(key):
                d[key] = json.loads(d[key])
            else:
                d[key] = [] if key in ("capabilities", "skills") else {}
        return d

    def set_extraction_status(
        self, chapter_num: int, status: str, error_message: Optional[str] = None
    ) -> None:
        conn = self._connect()
        try:
            now = datetime.utcnow().isoformat()
            if status == "started":
                conn.execute(
                    "INSERT INTO extraction_status (chapter_num, status, started_at) VALUES (?, ?, ?)",
                    (chapter_num, status, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE extraction_status SET status = ?, error_message = ?, completed_at = ?
                    WHERE chapter_num = ? AND status = 'started'
                    """,
                    (status, error_message, now, chapter_num),
                )
            conn.commit()
        finally:
            conn.close()

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT es.*, c.title as chapter_title
                FROM extraction_status es
                LEFT JOIN chapters c ON c.chapter_num = es.chapter_num
                ORDER BY es.chapter_num
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            total_pages = conn.execute("SELECT COALESCE(SUM(page_count), 0) FROM chapters").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM extraction_status WHERE status = 'started'").fetchone()[0]
            completed = conn.execute("SELECT COUNT(*) FROM extraction_status WHERE status = 'completed'").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM extraction_status WHERE status = 'failed'").fetchone()[0]
            return {
                "chapters_indexed": chapter_count,
                "experts_registered": expert_count,
                "total_pages_extracted": total_pages,
                "extractions_pending": pending,
                "extractions_completed": completed,
                "extractions_failed": failed,
            }
        finally:
            conn.close()

    def get_all_chapter_content(self) -> List[Dict[str, Any]]:
        """Get all chapter content for embedding generation."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT chapter_num, title, markdown_content, concepts FROM chapters ORDER BY chapter_num"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
                result.append(d)
            return result
        finally:
            conn.close()
