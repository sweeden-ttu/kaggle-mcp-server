"""SQLite database operations for MLSysEng MoE system.

Manages storage of extracted chapter content, expert definitions,
concepts, and extraction metadata.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class Database:
    def __init__(self, db_path: str | None = None):
        self.db_path = _ensure_parent(db_path or DEFAULT_DB_PATH)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number INTEGER UNIQUE,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    extracted_at TEXT,
                    page_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER,
                    capabilities TEXT,  -- JSON array
                    skills TEXT,        -- JSON array
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER,
                    status TEXT NOT NULL,  -- pending, running, done, error
                    message TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter
                    ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_slug
                    ON experts(slug);
                """
            )

    # ── chapter CRUD ──────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_number: int,
        title: str,
        source_path: str,
        markdown_content: str,
        page_count: int = 0,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chapters
                    (chapter_number, title, source_path, markdown_content,
                     extracted_at, page_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_number) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    extracted_at=excluded.extracted_at,
                    page_count=excluded.page_count
                """,
                (
                    chapter_number,
                    title,
                    source_path,
                    markdown_content,
                    datetime.utcnow().isoformat(),
                    page_count,
                ),
            )
            return cur.lastrowid or self.get_chapter_id(chapter_number)

    def get_chapter_id(self, chapter_number: int) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_number=?", (chapter_number,)
            ).fetchone()
            return row["id"] if row else None

    def list_chapters(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, chapter_number, title, extracted_at, page_count "
                "FROM chapters ORDER BY chapter_number"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chapter_content(self, chapter_number: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT markdown_content FROM chapters WHERE chapter_number=?",
                (chapter_number,),
            ).fetchone()
            return row["markdown_content"] if row else None

    # ── concepts ──────────────────────────────────────────────────

    def add_concepts(
        self, chapter_id: int, concepts: list[dict[str, str]]
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM concepts WHERE chapter_id=?", (chapter_id,))
            conn.executemany(
                "INSERT INTO concepts (chapter_id, concept, description, category) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        chapter_id,
                        c["concept"],
                        c.get("description", ""),
                        c.get("category", "general"),
                    )
                    for c in concepts
                ],
            )

    def get_concepts(self, chapter_id: int | None = None) -> list[dict]:
        with self._connect() as conn:
            if chapter_id is not None:
                rows = conn.execute(
                    "SELECT * FROM concepts WHERE chapter_id=?", (chapter_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM concepts").fetchall()
            return [dict(r) for r in rows]

    # ── experts ───────────────────────────────────────────────────

    def upsert_expert(self, expert: dict[str, Any]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO experts
                    (expert_name, slug, chapter_id, capabilities, skills,
                     strategy, formula, loop_config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    expert_name=excluded.expert_name,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config,
                    created_at=excluded.created_at
                """,
                (
                    expert["expert_name"],
                    expert["slug"],
                    expert.get("chapter_id"),
                    json.dumps(expert.get("capabilities", [])),
                    json.dumps(expert.get("skills", [])),
                    expert.get(
                        "strategy",
                        "Baseline -> Feature Eng -> Model Selection -> Submit",
                    ),
                    json.dumps(expert.get("formula", {})),
                    json.dumps(expert.get("loop_config", {})),
                    datetime.utcnow().isoformat(),
                ),
            )
            return cur.lastrowid or 0

    def list_experts(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, expert_name, slug, chapter_id, capabilities, "
                "skills, strategy, formula, loop_config, created_at "
                "FROM experts ORDER BY expert_name"
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if d[field]:
                        d[field] = json.loads(d[field])
                results.append(d)
            return results

    def get_expert_by_slug(self, slug: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug=?", (slug,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d[field]:
                    d[field] = json.loads(d[field])
            return d

    # ── extraction log ────────────────────────────────────────────

    def log_extraction(
        self, chapter_id: int, status: str, message: str = ""
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO extraction_log "
                "(chapter_id, status, message, started_at, finished_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    chapter_id,
                    status,
                    message,
                    now if status == "running" else None,
                    now if status in ("done", "error") else None,
                ),
            )

    def get_extraction_status(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT el.*, c.chapter_number, c.title
                FROM extraction_log el
                LEFT JOIN chapters c ON el.chapter_id = c.id
                ORDER BY el.id DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    # ── stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._connect() as conn:
            chapters = conn.execute("SELECT COUNT(*) as n FROM chapters").fetchone()
            concepts = conn.execute("SELECT COUNT(*) as n FROM concepts").fetchone()
            experts = conn.execute("SELECT COUNT(*) as n FROM experts").fetchone()
            return {
                "chapters": chapters["n"],
                "concepts": concepts["n"],
                "experts": experts["n"],
            }
