"""SQLite database operations for MLSysEng MoE system.

Manages chapters, experts, concepts, and extraction state.
"""

import json
import os
import sqlite3
from typing import Any, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    _ensure_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_number TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            content_md TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            page_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL REFERENCES chapters(id),
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            UNIQUE(chapter_id, name)
        );

        CREATE TABLE IF NOT EXISTS experts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL UNIQUE,
            expert_name TEXT NOT NULL,
            chapter_id INTEGER REFERENCES chapters(id),
            capabilities TEXT DEFAULT '[]',
            skills TEXT DEFAULT '[]',
            strategy TEXT,
            formula TEXT DEFAULT '{}',
            loop_config TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS extraction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL REFERENCES chapters(id),
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
        """
    )
    conn.commit()


# ── Chapter CRUD ──────────────────────────────────────────────────────────────


def upsert_chapter(
    conn: sqlite3.Connection,
    chapter_number: str,
    title: str,
    source_path: str,
    content_md: str | None = None,
    page_count: int = 0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO chapters (chapter_number, title, source_path, content_md, page_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chapter_number) DO UPDATE SET
            title = excluded.title,
            source_path = excluded.source_path,
            content_md = COALESCE(excluded.content_md, chapters.content_md),
            page_count = excluded.page_count,
            extracted_at = CURRENT_TIMESTAMP
        """,
        (chapter_number, title, source_path, content_md, page_count),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM chapters WHERE chapter_number = ?", (chapter_number,)
    ).fetchone()
    return row["id"]


def get_chapter(conn: sqlite3.Connection, chapter_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM chapters WHERE id = ?", (chapter_id,)).fetchone()
    return dict(row) if row else None


def list_chapters(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_number").fetchall()
    return [dict(r) for r in rows]


# ── Concept CRUD ──────────────────────────────────────────────────────────────


def upsert_concept(
    conn: sqlite3.Connection,
    chapter_id: int,
    name: str,
    description: str | None = None,
    category: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO concepts (chapter_id, name, description, category)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(chapter_id, name) DO UPDATE SET
            description = COALESCE(excluded.description, concepts.description),
            category = COALESCE(excluded.category, concepts.category)
        """,
        (chapter_id, name, description, category),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM concepts WHERE chapter_id = ? AND name = ?",
        (chapter_id, name),
    ).fetchone()
    return row["id"]


def list_concepts(
    conn: sqlite3.Connection, chapter_id: int | None = None
) -> list[dict]:
    if chapter_id is not None:
        rows = conn.execute(
            "SELECT * FROM concepts WHERE chapter_id = ? ORDER BY name",
            (chapter_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM concepts ORDER BY name").fetchall()
    return [dict(r) for r in rows]


# ── Expert CRUD ───────────────────────────────────────────────────────────────


def upsert_expert(conn: sqlite3.Connection, expert_data: dict[str, Any]) -> int:
    slug = expert_data["slug"]
    conn.execute(
        """
        INSERT INTO experts (slug, expert_name, chapter_id, capabilities, skills,
                             strategy, formula, loop_config)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            expert_name = excluded.expert_name,
            chapter_id = excluded.chapter_id,
            capabilities = excluded.capabilities,
            skills = excluded.skills,
            strategy = excluded.strategy,
            formula = excluded.formula,
            loop_config = excluded.loop_config
        """,
        (
            slug,
            expert_data.get("expert_name", slug),
            expert_data.get("chapter_id"),
            json.dumps(expert_data.get("capabilities", [])),
            json.dumps(expert_data.get("skills", [])),
            expert_data.get("strategy", ""),
            json.dumps(expert_data.get("formula", {})),
            json.dumps(expert_data.get("loop_config", {})),
        ),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM experts WHERE slug = ?", (slug,)).fetchone()
    return row["id"]


def get_expert(conn: sqlite3.Connection, slug: str) -> dict | None:
    row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
    if not row:
        return None
    data = dict(row)
    for field in ("capabilities", "skills", "formula", "loop_config"):
        if data.get(field):
            try:
                data[field] = json.loads(data[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return data


def list_experts(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM experts ORDER BY slug").fetchall()
    results = []
    for row in rows:
        data = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(data)
    return results


# ── Extraction log ────────────────────────────────────────────────────────────


def log_extraction(
    conn: sqlite3.Connection,
    chapter_id: int,
    status: str,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO extraction_log (chapter_id, status, error_message,
            started_at, completed_at)
        VALUES (?, ?, ?,
            CASE WHEN ? = 'running' THEN CURRENT_TIMESTAMP ELSE NULL END,
            CASE WHEN ? IN ('done', 'error') THEN CURRENT_TIMESTAMP ELSE NULL END
        )
        """,
        (chapter_id, status, error_message, status, status),
    )
    conn.commit()


def get_extraction_status(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.chapter_number, c.title, el.status, el.error_message,
               el.started_at, el.completed_at
        FROM extraction_log el
        JOIN chapters c ON c.id = el.chapter_id
        ORDER BY el.id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict:
    chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
    concept_count = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
    extracted = conn.execute(
        "SELECT COUNT(*) FROM chapters WHERE content_md IS NOT NULL"
    ).fetchone()[0]
    return {
        "total_chapters": chapter_count,
        "extracted_chapters": extracted,
        "total_concepts": concept_count,
        "total_experts": expert_count,
    }
