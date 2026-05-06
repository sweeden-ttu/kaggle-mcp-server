"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
)


def get_db_path() -> Path:
    path = Path(DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or str(get_db_path())
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema."""
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER NOT NULL,
                title TEXT NOT NULL,
                folder_path TEXT NOT NULL,
                pdf_path TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chapter_number)
            );

            CREATE TABLE IF NOT EXISTS content_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                block_type TEXT NOT NULL DEFAULT 'text',
                content TEXT NOT NULL,
                page_number INTEGER,
                position INTEGER,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                concept_name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_name TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                chapter_id INTEGER,
                capabilities TEXT NOT NULL DEFAULT '[]',
                skills TEXT NOT NULL DEFAULT '[]',
                strategy TEXT,
                formula TEXT NOT NULL DEFAULT '{}',
                loop_config TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE INDEX IF NOT EXISTS idx_content_chapter ON content_blocks(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
        """)
        conn.commit()
    finally:
        conn.close()


def insert_chapter(
    chapter_number: int,
    title: str,
    folder_path: str,
    pdf_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insert or update a chapter record. Returns chapter ID."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO chapters (chapter_number, title, folder_path, pdf_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chapter_number) DO UPDATE SET
                title=excluded.title,
                folder_path=excluded.folder_path,
                pdf_path=excluded.pdf_path,
                extracted_at=CURRENT_TIMESTAMP""",
            (chapter_number, title, folder_path, pdf_path),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM chapters WHERE chapter_number=?", (chapter_number,)
        ).fetchone()
        return row["id"]
    finally:
        conn.close()


def insert_content_block(
    chapter_id: int,
    content: str,
    block_type: str = "text",
    page_number: Optional[int] = None,
    position: Optional[int] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insert a content block for a chapter."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO content_blocks (chapter_id, block_type, content, page_number, position)
            VALUES (?, ?, ?, ?, ?)""",
            (chapter_id, block_type, content, page_number, position),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_concept(
    chapter_id: int,
    concept_name: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insert a concept extracted from a chapter."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO concepts (chapter_id, concept_name, description, category)
            VALUES (?, ?, ?, ?)""",
            (chapter_id, concept_name, description, category),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_expert(
    expert_name: str,
    slug: str,
    chapter_id: Optional[int] = None,
    capabilities: Optional[list] = None,
    skills: Optional[list] = None,
    strategy: Optional[str] = None,
    formula: Optional[dict] = None,
    loop_config: Optional[dict] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insert or update an expert definition."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                expert_name=excluded.expert_name,
                chapter_id=excluded.chapter_id,
                capabilities=excluded.capabilities,
                skills=excluded.skills,
                strategy=excluded.strategy,
                formula=excluded.formula,
                loop_config=excluded.loop_config""",
            (
                expert_name,
                slug,
                chapter_id,
                json.dumps(capabilities or []),
                json.dumps(skills or []),
                strategy,
                json.dumps(formula or {}),
                json.dumps(loop_config or {}),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM experts WHERE slug=?", (slug,)
        ).fetchone()
        return row["id"]
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Get all registered experts."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["capabilities"] = json.loads(d["capabilities"])
            d["skills"] = json.loads(d["skills"])
            d["formula"] = json.loads(d["formula"])
            d["loop_config"] = json.loads(d["loop_config"])
            results.append(d)
        return results
    finally:
        conn.close()


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a single expert by slug."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM experts WHERE slug=?", (slug,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["capabilities"] = json.loads(d["capabilities"])
        d["skills"] = json.loads(d["skills"])
        d["formula"] = json.loads(d["formula"])
        d["loop_config"] = json.loads(d["loop_config"])
        return d
    finally:
        conn.close()


def get_chapter_content(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    """Get all content blocks for a chapter."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM content_blocks WHERE chapter_id=? ORDER BY position",
            (chapter_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_chapter_concepts(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    """Get all concepts for a chapter."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM concepts WHERE chapter_id=? ORDER BY concept_name",
            (chapter_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    """Get all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_number"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_extraction(
    chapter_id: int,
    status: str,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Log extraction status for a chapter."""
    conn = get_connection(db_path)
    try:
        if status in ("completed", "failed"):
            conn.execute(
                """UPDATE extraction_log SET status=?, error_message=?, completed_at=CURRENT_TIMESTAMP
                WHERE chapter_id=? AND status='in_progress'""",
                (status, error_message, chapter_id),
            )
        else:
            conn.execute(
                """INSERT INTO extraction_log (chapter_id, status) VALUES (?, ?)""",
                (chapter_id, status),
            )
        conn.commit()
    finally:
        conn.close()


def get_extraction_status(db_path: Optional[str] = None) -> list[dict]:
    """Get extraction status for all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT c.chapter_number, c.title, el.status, el.started_at, el.completed_at, el.error_message
            FROM chapters c
            LEFT JOIN extraction_log el ON c.id = el.chapter_id
            ORDER BY c.chapter_number, el.started_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    """Get system statistics."""
    conn = get_connection(db_path)
    try:
        chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
        blocks = conn.execute("SELECT COUNT(*) as c FROM content_blocks").fetchone()["c"]
        concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
        experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
        return {
            "chapters": chapters,
            "content_blocks": blocks,
            "concepts": concepts,
            "experts": experts,
        }
    finally:
        conn.close()
