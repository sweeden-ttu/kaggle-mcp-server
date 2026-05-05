"""SQLite operations for MLSysEng MoE system.

Manages chapters, experts, concepts, and extraction metadata.
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


class MLSysEngDB:
    """SQLite database for storing extracted knowledge and expert definitions."""

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
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number INTEGER UNIQUE,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    content_md TEXT,
                    page_count INTEGER DEFAULT 0,
                    extracted_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id),
                    UNIQUE(chapter_id, concept)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER,
                    capabilities TEXT,  -- JSON array
                    skills TEXT,        -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER,
                    status TEXT NOT NULL,  -- pending, running, completed, failed
                    message TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            """)
            conn.commit()
        finally:
            conn.close()

    def upsert_chapter(
        self,
        chapter_number: int,
        title: str,
        source_path: str,
        content_md: str,
        page_count: int = 0,
    ) -> int:
        """Insert or update a chapter. Returns chapter id."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        try:
            cur = conn.execute(
                """INSERT INTO chapters (chapter_number, title, source_path, content_md, page_count, extracted_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_number) DO UPDATE SET
                     title=excluded.title,
                     source_path=excluded.source_path,
                     content_md=excluded.content_md,
                     page_count=excluded.page_count,
                     updated_at=excluded.updated_at""",
                (chapter_number, title, source_path, content_md, page_count, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_number=?", (chapter_number,)
            ).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_chapter(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_number=?", (chapter_number,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_number"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def add_concepts(self, chapter_id: int, concepts: List[Dict[str, str]]):
        """Add concepts for a chapter (upsert)."""
        conn = self._get_conn()
        try:
            for c in concepts:
                conn.execute(
                    """INSERT INTO concepts (chapter_id, concept, description, category)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(chapter_id, concept) DO UPDATE SET
                         description=excluded.description,
                         category=excluded.category""",
                    (chapter_id, c["concept"], c.get("description", ""), c.get("category", "general")),
                )
            conn.commit()
        finally:
            conn.close()

    def get_concepts(self, chapter_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            if chapter_id:
                rows = conn.execute(
                    "SELECT * FROM concepts WHERE chapter_id=?", (chapter_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM concepts").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        """Insert or update an expert definition. Returns expert id."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        try:
            capabilities = json.dumps(expert.get("capabilities", []))
            skills = json.dumps(expert.get("skills", []))
            formula = json.dumps(expert.get("formula", {}))
            loop_config = json.dumps(expert.get("loop_config", {}))

            conn.execute(
                """INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(slug) DO UPDATE SET
                     expert_name=excluded.expert_name,
                     chapter_id=excluded.chapter_id,
                     capabilities=excluded.capabilities,
                     skills=excluded.skills,
                     strategy=excluded.strategy,
                     formula=excluded.formula,
                     loop_config=excluded.loop_config,
                     updated_at=excluded.updated_at""",
                (
                    expert["expert_name"],
                    expert["slug"],
                    expert.get("chapter_id"),
                    capabilities,
                    skills,
                    expert.get("strategy", ""),
                    formula,
                    loop_config,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM experts WHERE slug=?", (expert["slug"],)
            ).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM experts WHERE slug=?", (slug,)).fetchone()
            if not row:
                return None
            expert = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if expert.get(field):
                    expert[field] = json.loads(expert[field])
            return expert
        finally:
            conn.close()

    def get_all_experts(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
            experts = []
            for row in rows:
                expert = dict(row)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if expert.get(field):
                        expert[field] = json.loads(expert[field])
                experts.append(expert)
            return experts
        finally:
            conn.close()

    def log_extraction(
        self, chapter_id: int, status: str, message: str = ""
    ) -> int:
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        try:
            cur = conn.execute(
                """INSERT INTO extraction_log (chapter_id, status, message, started_at)
                   VALUES (?, ?, ?, ?)""",
                (chapter_id, status, message, now),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def update_extraction_log(self, log_id: int, status: str, message: str = ""):
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()
        try:
            conn.execute(
                "UPDATE extraction_log SET status=?, message=?, completed_at=? WHERE id=?",
                (status, message, now, log_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT el.*, c.title as chapter_title, c.chapter_number
                   FROM extraction_log el
                   LEFT JOIN chapters c ON el.chapter_id = c.id
                   ORDER BY el.id DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            chapters = conn.execute("SELECT COUNT(*) as cnt FROM chapters").fetchone()["cnt"]
            concepts = conn.execute("SELECT COUNT(*) as cnt FROM concepts").fetchone()["cnt"]
            experts = conn.execute("SELECT COUNT(*) as cnt FROM experts").fetchone()["cnt"]
            extractions = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM extraction_log GROUP BY status"
            ).fetchall()
            return {
                "chapters": chapters,
                "concepts": concepts,
                "experts": experts,
                "extractions": {r["status"]: r["cnt"] for r in extractions},
            }
        finally:
            conn.close()
