"""SQLite database operations for MLSysEng MoE system.

Stores extracted chapter content, expert definitions, concepts, and extraction status.
"""

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Chapter:
    id: Optional[int] = None
    chapter_number: int = 0
    title: str = ""
    slug: str = ""
    source_path: str = ""
    markdown_content: str = ""
    concepts: str = "[]"
    extracted_at: float = 0.0
    status: str = "pending"


@dataclass
class Expert:
    id: Optional[int] = None
    expert_name: str = ""
    slug: str = ""
    chapter_id: Optional[int] = None
    capabilities: str = "[]"
    skills: str = "[]"
    strategy: str = ""
    formula: str = "{}"
    loop_config: str = "{}"
    created_at: float = 0.0


@dataclass
class ConceptEntry:
    id: Optional[int] = None
    chapter_id: Optional[int] = None
    concept: str = ""
    description: str = ""
    category: str = ""


DEFAULT_DB_PATH = os.path.expanduser("~/.mlsyseng/mlsyseng.db")


class MLSysEngDB:
    """SQLite-backed storage for the MLSysEng MoE system."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL,
                    markdown_content TEXT DEFAULT '',
                    concepts TEXT DEFAULT '[]',
                    extracted_at REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT DEFAULT '[]',
                    skills TEXT DEFAULT '[]',
                    strategy TEXT DEFAULT '',
                    formula TEXT DEFAULT '{}',
                    loop_config TEXT DEFAULT '{}',
                    created_at REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER REFERENCES chapters(id),
                    concept TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    category TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_slug ON chapters(slug);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
            """)

    def upsert_chapter(self, chapter: Chapter) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM chapters WHERE slug = ?", (chapter.slug,)
            ).fetchone()
            if row:
                conn.execute(
                    """UPDATE chapters SET chapter_number=?, title=?, source_path=?,
                       markdown_content=?, concepts=?, extracted_at=?, status=?
                       WHERE slug=?""",
                    (
                        chapter.chapter_number, chapter.title, chapter.source_path,
                        chapter.markdown_content, chapter.concepts,
                        chapter.extracted_at, chapter.status, chapter.slug,
                    ),
                )
                return row["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO chapters
                       (chapter_number, title, slug, source_path, markdown_content,
                        concepts, extracted_at, status)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        chapter.chapter_number, chapter.title, chapter.slug,
                        chapter.source_path, chapter.markdown_content,
                        chapter.concepts, chapter.extracted_at, chapter.status,
                    ),
                )
                return cur.lastrowid

    def get_chapter(self, slug: str) -> Optional[Chapter]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                return Chapter(**dict(row))
        return None

    def list_chapters(self) -> list[Chapter]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_number"
            ).fetchall()
            return [Chapter(**dict(r)) for r in rows]

    def upsert_expert(self, expert: Expert) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM experts WHERE slug = ?", (expert.slug,)
            ).fetchone()
            if row:
                conn.execute(
                    """UPDATE experts SET expert_name=?, chapter_id=?, capabilities=?,
                       skills=?, strategy=?, formula=?, loop_config=?, created_at=?
                       WHERE slug=?""",
                    (
                        expert.expert_name, expert.chapter_id,
                        expert.capabilities, expert.skills, expert.strategy,
                        expert.formula, expert.loop_config, expert.created_at,
                        expert.slug,
                    ),
                )
                return row["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO experts
                       (expert_name, slug, chapter_id, capabilities, skills,
                        strategy, formula, loop_config, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        expert.expert_name, expert.slug, expert.chapter_id,
                        expert.capabilities, expert.skills, expert.strategy,
                        expert.formula, expert.loop_config, expert.created_at,
                    ),
                )
                return cur.lastrowid

    def get_expert(self, slug: str) -> Optional[Expert]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                return Expert(**dict(row))
        return None

    def list_experts(self) -> list[Expert]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            return [Expert(**dict(r)) for r in rows]

    def add_concept(self, entry: ConceptEntry) -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO concepts (chapter_id, concept, description, category)
                   VALUES (?,?,?,?)""",
                (entry.chapter_id, entry.concept, entry.description, entry.category),
            )
            return cur.lastrowid

    def search_concepts(self, query: str) -> list[ConceptEntry]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM concepts
                   WHERE concept LIKE ? OR description LIKE ?
                   ORDER BY concept""",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [ConceptEntry(**dict(r)) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        with self._get_conn() as conn:
            chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            extracted = conn.execute(
                "SELECT COUNT(*) as c FROM chapters WHERE status='extracted'"
            ).fetchone()["c"]
            experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
            return {
                "total_chapters": chapters,
                "extracted_chapters": extracted,
                "total_experts": experts,
                "total_concepts": concepts,
                "db_path": self.db_path,
            }
