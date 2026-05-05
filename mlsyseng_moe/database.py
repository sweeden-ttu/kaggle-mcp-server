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
    """Initialize the database schema."""
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER NOT NULL,
                title TEXT NOT NULL,
                pdf_path TEXT,
                markdown_content TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chapter_number)
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                concept_name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
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
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_chapter
                ON concepts(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_name
                ON concepts(concept_name);
            CREATE INDEX IF NOT EXISTS idx_experts_slug
                ON experts(slug);
        """)
        conn.commit()
    finally:
        conn.close()


def insert_chapter(
    chapter_number: int,
    title: str,
    pdf_path: str,
    markdown_content: str,
    db_path: Optional[str] = None,
) -> int:
    """Insert or update a chapter record."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO chapters (chapter_number, title, pdf_path, markdown_content)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chapter_number) DO UPDATE SET
                title=excluded.title,
                pdf_path=excluded.pdf_path,
                markdown_content=excluded.markdown_content,
                extracted_at=CURRENT_TIMESTAMP
            """,
            (chapter_number, title, pdf_path, markdown_content),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_concept(
    chapter_id: int,
    concept_name: str,
    description: str = "",
    category: str = "",
    db_path: Optional[str] = None,
) -> int:
    """Insert a concept extracted from a chapter."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO concepts (chapter_id, concept_name, description, category) VALUES (?, ?, ?, ?)",
            (chapter_id, concept_name, description, category),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_expert(
    expert_name: str,
    slug: str,
    chapter_id: Optional[int],
    capabilities: list,
    skills: list,
    strategy: str,
    formula: dict,
    loop_config: dict,
    db_path: Optional[str] = None,
) -> int:
    """Insert or update an expert definition."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                expert_name=excluded.expert_name,
                chapter_id=excluded.chapter_id,
                capabilities=excluded.capabilities,
                skills=excluded.skills,
                strategy=excluded.strategy,
                formula=excluded.formula,
                loop_config=excluded.loop_config
            """,
            (
                expert_name,
                slug,
                chapter_id,
                json.dumps(capabilities),
                json.dumps(skills),
                strategy,
                json.dumps(formula),
                json.dumps(loop_config),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all registered experts."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            expert = dict(row)
            expert["capabilities"] = json.loads(expert["capabilities"] or "[]")
            expert["skills"] = json.loads(expert["skills"] or "[]")
            expert["formula"] = json.loads(expert["formula"] or "{}")
            expert["loop_config"] = json.loads(expert["loop_config"] or "{}")
            results.append(expert)
        return results
    finally:
        conn.close()


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Retrieve an expert by slug."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        expert = dict(row)
        expert["capabilities"] = json.loads(expert["capabilities"] or "[]")
        expert["skills"] = json.loads(expert["skills"] or "[]")
        expert["formula"] = json.loads(expert["formula"] or "{}")
        expert["loop_config"] = json.loads(expert["loop_config"] or "{}")
        return expert
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all indexed chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, chapter_number, title, pdf_path, extracted_at FROM chapters ORDER BY chapter_number"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_chapter_content(chapter_id: int, db_path: Optional[str] = None) -> Optional[str]:
    """Retrieve markdown content for a chapter."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT markdown_content FROM chapters WHERE id = ?", (chapter_id,)
        ).fetchone()
        return row["markdown_content"] if row else None
    finally:
        conn.close()


def get_concepts_for_chapter(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all concepts for a given chapter."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM concepts WHERE chapter_id = ?", (chapter_id,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def log_extraction(
    chapter_id: int, status: str, error_message: str = "", db_path: Optional[str] = None
) -> None:
    """Log extraction progress."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO extraction_log (chapter_id, status, error_message, started_at, completed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CASE WHEN ? IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE NULL END)
            """,
            (chapter_id, status, error_message, status),
        )
        conn.commit()
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    """Get system statistics."""
    conn = get_connection(db_path)
    try:
        chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        concept_count = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        return {
            "chapters_indexed": chapter_count,
            "concepts_extracted": concept_count,
            "experts_registered": expert_count,
        }
    finally:
        conn.close()
