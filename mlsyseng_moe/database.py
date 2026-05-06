"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
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
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER UNIQUE,
                title TEXT NOT NULL,
                folder_path TEXT,
                pdf_path TEXT,
                extracted_at TIMESTAMP,
                content_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS chapter_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                section_title TEXT,
                content TEXT NOT NULL,
                page_number INTEGER,
                content_type TEXT DEFAULT 'text',
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
                capabilities TEXT,
                skills TEXT,
                strategy TEXT,
                formula TEXT,
                loop_config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER,
                status TEXT NOT NULL,
                message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_content_chapter ON chapter_content(chapter_id);
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
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT OR REPLACE INTO chapters
               (chapter_number, title, folder_path, pdf_path, extracted_at)
               VALUES (?, ?, ?, ?, ?)""",
            (chapter_number, title, folder_path, pdf_path, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_content(
    chapter_id: int,
    content: str,
    section_title: Optional[str] = None,
    page_number: Optional[int] = None,
    content_type: str = "text",
    db_path: Optional[str] = None,
) -> int:
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO chapter_content
               (chapter_id, section_title, content, page_number, content_type)
               VALUES (?, ?, ?, ?, ?)""",
            (chapter_id, section_title, content, page_number, content_type),
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
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT OR REPLACE INTO experts
               (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["capabilities"] = json.loads(d["capabilities"]) if d["capabilities"] else []
            d["skills"] = json.loads(d["skills"]) if d["skills"] else []
            d["formula"] = json.loads(d["formula"]) if d["formula"] else {}
            d["loop_config"] = json.loads(d["loop_config"]) if d["loop_config"] else {}
            results.append(d)
        return results
    finally:
        conn.close()


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["capabilities"] = json.loads(d["capabilities"]) if d["capabilities"] else []
        d["skills"] = json.loads(d["skills"]) if d["skills"] else []
        d["formula"] = json.loads(d["formula"]) if d["formula"] else {}
        d["loop_config"] = json.loads(d["loop_config"]) if d["loop_config"] else {}
        return d
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_number").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_chapter_content(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM chapter_content WHERE chapter_id = ? ORDER BY page_number",
            (chapter_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_concepts_for_chapter(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM concepts WHERE chapter_id = ?", (chapter_id,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def log_extraction(
    chapter_id: int,
    status: str,
    message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    conn = get_connection(db_path)
    try:
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT INTO extraction_log (chapter_id, status, message, started_at, completed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (chapter_id, status, message, now, now if status in ("completed", "failed") else None),
        )
        conn.commit()
    finally:
        conn.close()


def get_extraction_status(db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT c.chapter_number, c.title, el.status, el.message, el.completed_at
               FROM extraction_log el
               JOIN chapters c ON c.id = el.chapter_id
               ORDER BY el.id DESC"""
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    conn = get_connection(db_path)
    try:
        chapters = conn.execute("SELECT COUNT(*) as cnt FROM chapters").fetchone()["cnt"]
        content_blocks = conn.execute("SELECT COUNT(*) as cnt FROM chapter_content").fetchone()["cnt"]
        concepts = conn.execute("SELECT COUNT(*) as cnt FROM concepts").fetchone()["cnt"]
        experts = conn.execute("SELECT COUNT(*) as cnt FROM experts").fetchone()["cnt"]
        return {
            "chapters": chapters,
            "content_blocks": content_blocks,
            "concepts": concepts,
            "experts": experts,
        }
    finally:
        conn.close()
