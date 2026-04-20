"""SQLite database operations for MLSysEng MoE.

Manages chapters, experts, concepts, and extraction state.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.mlsyseng/mlsyseng.db")


@dataclass
class Chapter:
    id: Optional[int] = None
    folder_name: str = ""
    title: str = ""
    pdf_path: str = ""
    content_md: str = ""
    concepts: str = "[]"
    extracted_at: Optional[str] = None
    status: str = "pending"

    @property
    def concept_list(self) -> list[str]:
        return json.loads(self.concepts) if self.concepts else []


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
    created_at: Optional[str] = None

    @property
    def capabilities_list(self) -> list[str]:
        return json.loads(self.capabilities) if self.capabilities else []

    @property
    def skills_list(self) -> list[str]:
        return json.loads(self.skills) if self.skills else []

    @property
    def formula_dict(self) -> dict:
        return json.loads(self.formula) if self.formula else {}

    @property
    def loop_config_dict(self) -> dict:
        return json.loads(self.loop_config) if self.loop_config else {}

    def to_dict(self) -> dict:
        return {
            "expert_name": self.expert_name,
            "slug": self.slug,
            "capabilities": self.capabilities_list,
            "skills": self.skills_list,
            "strategy": self.strategy,
            "formula": self.formula_dict,
            "loop_config": self.loop_config_dict,
        }


@dataclass
class ConceptEntry:
    id: Optional[int] = None
    chapter_id: Optional[int] = None
    concept: str = ""
    description: str = ""
    embedding_id: Optional[str] = None


class Database:
    """SQLite database for MLSysEng MoE system."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_name TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    pdf_path TEXT,
                    content_md TEXT DEFAULT '',
                    concepts TEXT DEFAULT '[]',
                    extracted_at TEXT,
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT DEFAULT '[]',
                    skills TEXT DEFAULT '[]',
                    strategy TEXT DEFAULT '',
                    formula TEXT DEFAULT '{}',
                    loop_config TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER REFERENCES chapters(id),
                    concept TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    embedding_id TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER REFERENCES chapters(id),
                    event TEXT NOT NULL,
                    details TEXT DEFAULT '',
                    timestamp TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter
                    ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_slug
                    ON experts(slug);
            """)

    def upsert_chapter(self, chapter: Chapter) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM chapters WHERE folder_name = ?",
                (chapter.folder_name,),
            ).fetchone()
            if row:
                conn.execute(
                    """UPDATE chapters SET title=?, pdf_path=?, content_md=?,
                       concepts=?, extracted_at=?, status=? WHERE id=?""",
                    (
                        chapter.title, chapter.pdf_path, chapter.content_md,
                        chapter.concepts, chapter.extracted_at, chapter.status,
                        row["id"],
                    ),
                )
                return row["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO chapters
                       (folder_name, title, pdf_path, content_md, concepts, extracted_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chapter.folder_name, chapter.title, chapter.pdf_path,
                        chapter.content_md, chapter.concepts, chapter.extracted_at,
                        chapter.status,
                    ),
                )
                return cur.lastrowid

    def get_chapter(self, chapter_id: int) -> Optional[Chapter]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chapters WHERE id = ?", (chapter_id,)).fetchone()
            if row:
                return Chapter(**dict(row))
            return None

    def get_chapter_by_folder(self, folder_name: str) -> Optional[Chapter]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE folder_name = ?", (folder_name,)
            ).fetchone()
            if row:
                return Chapter(**dict(row))
            return None

    def list_chapters(self, status: Optional[str] = None) -> list[Chapter]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM chapters WHERE status = ? ORDER BY folder_name",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chapters ORDER BY folder_name"
                ).fetchall()
            return [Chapter(**dict(r)) for r in rows]

    def upsert_expert(self, expert: Expert) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM experts WHERE slug = ?", (expert.slug,)
            ).fetchone()
            if row:
                conn.execute(
                    """UPDATE experts SET expert_name=?, chapter_id=?, capabilities=?,
                       skills=?, strategy=?, formula=?, loop_config=? WHERE id=?""",
                    (
                        expert.expert_name, expert.chapter_id, expert.capabilities,
                        expert.skills, expert.strategy, expert.formula,
                        expert.loop_config, row["id"],
                    ),
                )
                return row["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO experts
                       (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        expert.expert_name, expert.slug, expert.chapter_id,
                        expert.capabilities, expert.skills, expert.strategy,
                        expert.formula, expert.loop_config,
                    ),
                )
                return cur.lastrowid

    def get_expert(self, slug: str) -> Optional[Expert]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
            if row:
                return Expert(**dict(row))
            return None

    def list_experts(self) -> list[Expert]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
            return [Expert(**dict(r)) for r in rows]

    def add_concept(self, concept: ConceptEntry) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO concepts (chapter_id, concept, description, embedding_id)
                   VALUES (?, ?, ?, ?)""",
                (concept.chapter_id, concept.concept, concept.description, concept.embedding_id),
            )
            return cur.lastrowid

    def get_concepts(self, chapter_id: Optional[int] = None) -> list[ConceptEntry]:
        with self._conn() as conn:
            if chapter_id is not None:
                rows = conn.execute(
                    "SELECT * FROM concepts WHERE chapter_id = ?", (chapter_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM concepts").fetchall()
            return [ConceptEntry(**dict(r)) for r in rows]

    def log_event(self, chapter_id: int, event: str, details: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO extraction_log (chapter_id, event, details) VALUES (?, ?, ?)",
                (chapter_id, event, details),
            )

    def get_stats(self) -> dict:
        with self._conn() as conn:
            chapters_total = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            chapters_done = conn.execute(
                "SELECT COUNT(*) FROM chapters WHERE status = 'extracted'"
            ).fetchone()[0]
            experts_total = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            concepts_total = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
            return {
                "chapters_total": chapters_total,
                "chapters_extracted": chapters_done,
                "chapters_pending": chapters_total - chapters_done,
                "experts_total": experts_total,
                "concepts_total": concepts_total,
            }
