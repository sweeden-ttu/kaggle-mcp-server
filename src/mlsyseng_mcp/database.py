"""SQLite database operations for MLSysEng MoE system.

Stores extracted chapter content, expert definitions, concepts, and embeddings metadata.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
    )


class Database:
    """SQLite backend for chapter content, experts, and concept storage."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_num INTEGER UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    page_count INTEGER DEFAULT 0,
                    extracted_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept_name TEXT NOT NULL,
                    description TEXT,
                    category TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE,
                    UNIQUE(chapter_id, concept_name)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER,
                    capabilities TEXT,
                    skills TEXT,
                    strategy TEXT,
                    formula TEXT,
                    loop_config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(concept_name);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            """)
            conn.commit()
        finally:
            conn.close()

    # ── Chapter operations ────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_num: int,
        title: str,
        source_path: str,
        markdown_content: str,
        page_count: int = 0,
    ) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO chapters (chapter_num, title, source_path, markdown_content, page_count, extracted_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_num) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    page_count=excluded.page_count,
                    extracted_at=excluded.extracted_at,
                    updated_at=excluded.updated_at
                """,
                (chapter_num, title, source_path, markdown_content, page_count,
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_num = ?", (chapter_num,)
            ).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_chapter(self, chapter_num: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_num = ?", (chapter_num,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_chapters(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_num"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Concept operations ────────────────────────────────────────────

    def add_concepts(self, chapter_id: int, concepts: List[Dict[str, Any]]):
        conn = self._connect()
        try:
            for c in concepts:
                conn.execute(
                    """
                    INSERT INTO concepts (chapter_id, concept_name, description, category, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chapter_id, concept_name) DO UPDATE SET
                        description=excluded.description,
                        category=excluded.category,
                        confidence=excluded.confidence
                    """,
                    (chapter_id, c["name"], c.get("description", ""),
                     c.get("category", "general"), c.get("confidence", 1.0)),
                )
            conn.commit()
        finally:
            conn.close()

    def get_concepts_for_chapter(self, chapter_id: int) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM concepts WHERE chapter_id = ? ORDER BY concept_name",
                (chapter_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT c.*, ch.title as chapter_title, ch.chapter_num
                FROM concepts c
                JOIN chapters ch ON c.chapter_id = ch.id
                WHERE c.concept_name LIKE ? OR c.description LIKE ?
                ORDER BY c.confidence DESC
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Expert operations ─────────────────────────────────────────────

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, updated_at)
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
                    expert_data["slug"],
                    expert_data.get("chapter_id"),
                    json.dumps(expert_data.get("capabilities", [])),
                    json.dumps(expert_data.get("skills", [])),
                    expert_data.get("strategy", ""),
                    json.dumps(expert_data.get("formula", {})),
                    json.dumps(expert_data.get("loop_config", {})),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM experts WHERE slug = ?", (expert_data["slug"],)
            ).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return result
        finally:
            conn.close()

    def list_experts(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                results.append(result)
            return results
        finally:
            conn.close()

    # ── Extraction log ────────────────────────────────────────────────

    def log_extraction(self, chapter_id: int, status: str, message: str = ""):
        conn = self._connect()
        try:
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO extraction_log (chapter_id, status, message, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chapter_id, status, message, now, now if status in ("completed", "error") else None),
            )
            conn.commit()
        finally:
            conn.close()

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT el.*, ch.title as chapter_title, ch.chapter_num
                FROM extraction_log el
                LEFT JOIN chapters ch ON el.chapter_id = ch.id
                ORDER BY el.id DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        conn = self._connect()
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
                "db_path": self.db_path,
            }
        finally:
            conn.close()
