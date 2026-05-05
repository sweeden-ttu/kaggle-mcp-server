"""SQLite operations for MLSysEng MoE system."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> Path:
    return Path(os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db")
    ))


class Database:
    """SQLite database for storing extracted chapters, experts, and concepts."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number INTEGER UNIQUE,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array
                    extracted_at TEXT DEFAULT (datetime('now')),
                    word_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER,
                    capabilities TEXT,  -- JSON array
                    skills TEXT,  -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,  -- JSON object
                    loop_config TEXT,  -- JSON object
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept_name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER,
                    status TEXT NOT NULL,
                    message TEXT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            """)
            conn.commit()
        finally:
            conn.close()

    def insert_chapter(
        self,
        chapter_number: int,
        title: str,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
        word_count: int = 0
    ) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO chapters
                   (chapter_number, title, source_path, markdown_content, concepts, word_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (chapter_number, title, source_path, markdown_content, json.dumps(concepts), word_count)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_chapter(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_number = ?",
                (chapter_number,)
            ).fetchone()
            if row:
                d = dict(row)
                d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
                return d
            return None
        finally:
            conn.close()

    def list_chapters(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, chapter_number, title, word_count, extracted_at FROM chapters ORDER BY chapter_number"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def insert_expert(self, expert_data: Dict[str, Any]) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO experts
                   (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    expert_data["expert_name"],
                    expert_data["slug"],
                    expert_data.get("chapter_id"),
                    json.dumps(expert_data.get("capabilities", [])),
                    json.dumps(expert_data.get("skills", [])),
                    expert_data.get("strategy", ""),
                    json.dumps(expert_data.get("formula", {})),
                    json.dumps(expert_data.get("loop_config", {})),
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                d = dict(row)
                d["capabilities"] = json.loads(d["capabilities"]) if d["capabilities"] else []
                d["skills"] = json.loads(d["skills"]) if d["skills"] else []
                d["formula"] = json.loads(d["formula"]) if d["formula"] else {}
                d["loop_config"] = json.loads(d["loop_config"]) if d["loop_config"] else {}
                return d
            return None
        finally:
            conn.close()

    def list_experts(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, expert_name, slug, strategy, created_at FROM experts ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def insert_concepts(self, chapter_id: int, concepts: List[Dict[str, str]]):
        conn = self._get_conn()
        try:
            conn.executemany(
                "INSERT INTO concepts (chapter_id, concept_name, description, category) VALUES (?, ?, ?, ?)",
                [(chapter_id, c["name"], c.get("description", ""), c.get("category", "")) for c in concepts]
            )
            conn.commit()
        finally:
            conn.close()

    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT c.*, ch.title as chapter_title, ch.chapter_number
                   FROM concepts c
                   JOIN chapters ch ON c.chapter_id = ch.id
                   WHERE c.concept_name LIKE ? OR c.description LIKE ?""",
                (f"%{query}%", f"%{query}%")
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def log_extraction(self, chapter_id: Optional[int], status: str, message: str):
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO extraction_log (chapter_id, status, message) VALUES (?, ?, ?)",
                (chapter_id, status, message)
            )
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            chapters = conn.execute("SELECT COUNT(*) as count FROM chapters").fetchone()["count"]
            experts = conn.execute("SELECT COUNT(*) as count FROM experts").fetchone()["count"]
            concepts = conn.execute("SELECT COUNT(*) as count FROM concepts").fetchone()["count"]
            total_words = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) as total FROM chapters"
            ).fetchone()["total"]
            return {
                "chapters_indexed": chapters,
                "experts_registered": experts,
                "concepts_extracted": concepts,
                "total_words": total_words,
            }
        finally:
            conn.close()
