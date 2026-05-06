"""SQLite database operations for the MLSysEng MoE system.

Stores extracted chapter content, expert definitions, and extraction metadata.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.mlsyseng/mlsyseng.db")


class Database:
    """Thread-safe SQLite database for MLSysEng knowledge storage."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _init_schema(self):
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_num INTEGER UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_count INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept TEXT NOT NULL,
                    description TEXT,
                    category TEXT DEFAULT 'general',
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT UNIQUE NOT NULL,
                    expert_name TEXT NOT NULL,
                    chapter_id INTEGER,
                    capabilities TEXT,  -- JSON array
                    skills TEXT,        -- JSON array
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_num INTEGER,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            """)

    def upsert_chapter(self, chapter_num: int, title: str, source_path: str,
                       markdown_content: str, page_count: int = 0) -> int:
        word_count = len(markdown_content.split()) if markdown_content else 0
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO chapters (chapter_num, title, source_path, markdown_content,
                                      page_count, word_count, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_num) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    page_count=excluded.page_count,
                    word_count=excluded.word_count,
                    extracted_at=excluded.extracted_at
            """, (chapter_num, title, source_path, markdown_content,
                  page_count, word_count, datetime.now(timezone.utc).isoformat()))
            cur.execute("SELECT id FROM chapters WHERE chapter_num=?", (chapter_num,))
            return cur.fetchone()["id"]

    def add_concepts(self, chapter_id: int, concepts: List[Dict[str, str]]):
        with self._cursor() as cur:
            cur.execute("DELETE FROM concepts WHERE chapter_id=?", (chapter_id,))
            for c in concepts:
                cur.execute(
                    "INSERT INTO concepts (chapter_id, concept, description, category) VALUES (?,?,?,?)",
                    (chapter_id, c["concept"], c.get("description", ""),
                     c.get("category", "general")))

    def upsert_expert(self, slug: str, expert_name: str, chapter_id: Optional[int],
                      capabilities: List[str], skills: List[str], strategy: str,
                      formula: Dict[str, Any], loop_config: Dict[str, Any]) -> int:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO experts (slug, expert_name, chapter_id, capabilities, skills,
                                     strategy, formula, loop_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    expert_name=excluded.expert_name,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config
            """, (slug, expert_name, chapter_id,
                  json.dumps(capabilities), json.dumps(skills),
                  strategy, json.dumps(formula), json.dumps(loop_config)))
            cur.execute("SELECT id FROM experts WHERE slug=?", (slug,))
            return cur.fetchone()["id"]

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM chapters ORDER BY chapter_num")
            return [dict(r) for r in cur.fetchall()]

    def get_chapter(self, chapter_num: int) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM chapters WHERE chapter_num=?", (chapter_num,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_chapter_content(self, chapter_id: int) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute("SELECT markdown_content FROM chapters WHERE id=?", (chapter_id,))
            row = cur.fetchone()
            return row["markdown_content"] if row else None

    def get_concepts(self, chapter_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            if chapter_id:
                cur.execute("SELECT * FROM concepts WHERE chapter_id=?", (chapter_id,))
            else:
                cur.execute("SELECT * FROM concepts")
            return [dict(r) for r in cur.fetchall()]

    def get_all_experts(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM experts ORDER BY expert_name")
            rows = []
            for r in cur.fetchall():
                d = dict(r)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                rows.append(d)
            return rows

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM experts WHERE slug=?", (slug,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return d

    def log_extraction(self, chapter_num: int, status: str,
                       error_message: Optional[str] = None):
        with self._cursor() as cur:
            now = datetime.now(timezone.utc).isoformat()
            if status == "started":
                cur.execute("""
                    INSERT INTO extraction_log (chapter_num, status, started_at)
                    VALUES (?, ?, ?)
                """, (chapter_num, status, now))
            elif status in ("completed", "failed"):
                cur.execute("""
                    UPDATE extraction_log SET status=?, completed_at=?, error_message=?
                    WHERE id = (
                        SELECT id FROM extraction_log
                        WHERE chapter_num=? AND status='started'
                        ORDER BY id DESC LIMIT 1
                    )
                """, (status, now, error_message, chapter_num))

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("""
                SELECT chapter_num, status, started_at, completed_at, error_message
                FROM extraction_log ORDER BY id DESC
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as n FROM chapters")
            chapters = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) as n FROM concepts")
            concepts = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) as n FROM experts")
            experts = cur.fetchone()["n"]
            cur.execute("SELECT COALESCE(SUM(word_count), 0) as n FROM chapters")
            words = cur.fetchone()["n"]
            return {
                "chapters_indexed": chapters,
                "concepts_extracted": concepts,
                "experts_registered": experts,
                "total_words": words,
                "db_path": self.db_path,
            }
