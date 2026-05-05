"""SQLite database operations for MLSysEng MoE system."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
)


class Database:
    """SQLite database for storing extracted chapter content and expert definitions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    chapter_number INTEGER,
                    source_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array of extracted concepts
                    extracted_at TEXT,
                    word_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,   -- JSON array
                    skills TEXT,         -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,        -- JSON object
                    loop_config TEXT,    -- JSON object
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT,
                    status TEXT,        -- pending, running, completed, failed
                    error_message TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_name ON chapters(chapter_name);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_extraction_status ON extraction_log(status);
                """
            )

    def upsert_chapter(
        self,
        chapter_name: str,
        chapter_number: int,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
    ) -> int:
        now = datetime.utcnow().isoformat()
        word_count = len(markdown_content.split())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter_name, chapter_number, source_path,
                                      markdown_content, concepts, extracted_at, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_name) DO UPDATE SET
                    source_path = excluded.source_path,
                    markdown_content = excluded.markdown_content,
                    concepts = excluded.concepts,
                    extracted_at = excluded.extracted_at,
                    word_count = excluded.word_count
                """,
                (
                    chapter_name,
                    chapter_number,
                    source_path,
                    markdown_content,
                    json.dumps(concepts),
                    now,
                    word_count,
                ),
            )
            cursor = conn.execute(
                "SELECT id FROM chapters WHERE chapter_name = ?", (chapter_name,)
            )
            return cursor.fetchone()["id"]

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_name = ?", (chapter_name,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["concepts"] = json.loads(result["concepts"]) if result["concepts"] else []
            return result

    def list_chapters(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, chapter_name, chapter_number, word_count, extracted_at FROM chapters ORDER BY chapter_number"
            ).fetchall()
            return [dict(r) for r in rows]

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                     skills, strategy, formula, loop_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    expert_name = excluded.expert_name,
                    chapter_id = excluded.chapter_id,
                    capabilities = excluded.capabilities,
                    skills = excluded.skills,
                    strategy = excluded.strategy,
                    formula = excluded.formula,
                    loop_config = excluded.loop_config,
                    updated_at = excluded.updated_at
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
            cursor = conn.execute(
                "SELECT id FROM experts WHERE slug = ?", (expert["slug"],)
            )
            return cursor.fetchone()["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if result[field]:
                    result[field] = json.loads(result[field])
            return result

    def list_experts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, expert_name, slug, strategy, created_at FROM experts ORDER BY expert_name"
            ).fetchall()
            return [dict(r) for r in rows]

    def log_extraction(self, chapter_name: str, status: str, error: Optional[str] = None):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            if status in ("pending", "running"):
                conn.execute(
                    "INSERT INTO extraction_log (chapter_name, status, started_at) VALUES (?, ?, ?)",
                    (chapter_name, status, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE extraction_log
                    SET status = ?, error_message = ?, completed_at = ?
                    WHERE chapter_name = ? AND status = 'running'
                    """,
                    (status, error, now, chapter_name),
                )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chapter_name, status, error_message, started_at, completed_at "
                "FROM extraction_log ORDER BY started_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            total_words = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) FROM chapters"
            ).fetchone()[0]
            return {
                "chapters": chapter_count,
                "experts": expert_count,
                "total_words": total_words,
            }
