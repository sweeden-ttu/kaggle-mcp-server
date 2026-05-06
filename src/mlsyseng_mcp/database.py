"""SQLite database operations for MLSysEng MoE knowledge storage."""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db")


def get_db_path() -> str:
    return os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
                chapter_number INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source_path TEXT,
                markdown_content TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                concept_name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_name TEXT UNIQUE NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                chapter_id INTEGER,
                capabilities TEXT,
                skills TEXT,
                strategy TEXT,
                formula TEXT,
                loop_config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS extraction_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition_name TEXT NOT NULL,
                experts_used TEXT,
                skills_applied TEXT,
                state_history TEXT,
                converged INTEGER DEFAULT 0,
                final_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(concept_name);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
        """)
        conn.commit()
    finally:
        conn.close()


def store_chapter(chapter_number: int, title: str, source_path: str,
                  markdown_content: str, db_path: Optional[str] = None) -> int:
    """Store extracted chapter content."""
    conn = get_connection(db_path)
    try:
        word_count = len(markdown_content.split())
        cursor = conn.execute("""
            INSERT OR REPLACE INTO chapters (chapter_number, title, source_path, markdown_content, word_count, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (chapter_number, title, source_path, markdown_content, word_count, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def store_concepts(chapter_id: int, concepts: list[dict], db_path: Optional[str] = None) -> None:
    """Store extracted concepts for a chapter."""
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM concepts WHERE chapter_id = ?", (chapter_id,))
        conn.executemany("""
            INSERT INTO concepts (chapter_id, concept_name, description, category, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, [(chapter_id, c["name"], c.get("description", ""), c.get("category", "general"),
               c.get("confidence", 1.0)) for c in concepts])
        conn.commit()
    finally:
        conn.close()


def store_expert(expert_data: dict, db_path: Optional[str] = None) -> int:
    """Store or update an expert definition."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            INSERT OR REPLACE INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            expert_data["expert_name"],
            expert_data["slug"],
            expert_data.get("chapter_id"),
            json.dumps(expert_data.get("capabilities", [])),
            json.dumps(expert_data.get("skills", [])),
            expert_data.get("strategy", ""),
            json.dumps(expert_data.get("formula", {})),
            json.dumps(expert_data.get("loop_config", {})),
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Get all registered experts."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        return [_expert_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a single expert by slug."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
        return _expert_row_to_dict(row) if row else None
    finally:
        conn.close()


def get_chapters(db_path: Optional[str] = None) -> list[dict]:
    """Get all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_number").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_chapter_content(chapter_number: int, db_path: Optional[str] = None) -> Optional[str]:
    """Get markdown content for a chapter."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT markdown_content FROM chapters WHERE chapter_number = ?",
                           (chapter_number,)).fetchone()
        return row["markdown_content"] if row else None
    finally:
        conn.close()


def get_concepts_for_chapter(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    """Get concepts associated with a chapter."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM concepts WHERE chapter_id = ?", (chapter_id,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_extraction_status(chapter_number: int, status: str,
                             error_message: Optional[str] = None,
                             db_path: Optional[str] = None) -> None:
    """Update extraction status for a chapter."""
    conn = get_connection(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        if status == "started":
            conn.execute("""
                INSERT OR REPLACE INTO extraction_status (chapter_number, status, started_at)
                VALUES (?, ?, ?)
            """, (chapter_number, status, now))
        elif status in ("completed", "failed"):
            conn.execute("""
                UPDATE extraction_status SET status = ?, completed_at = ?, error_message = ?
                WHERE chapter_number = ?
            """, (status, now, error_message, chapter_number))
        conn.commit()
    finally:
        conn.close()


def get_extraction_status(db_path: Optional[str] = None) -> list[dict]:
    """Get extraction status for all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM extraction_status ORDER BY chapter_number").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def store_competition_entry(competition_name: str, experts_used: list[str],
                            skills_applied: list[str], state_history: list[dict],
                            converged: bool, final_score: Optional[float] = None,
                            db_path: Optional[str] = None) -> int:
    """Store a competition entry result."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            INSERT INTO competition_entries (competition_name, experts_used, skills_applied,
                                            state_history, converged, final_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            competition_name,
            json.dumps(experts_used),
            json.dumps(skills_applied),
            json.dumps(state_history),
            int(converged),
            final_score,
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    """Get system statistics."""
    conn = get_connection(db_path)
    try:
        chapters = conn.execute("SELECT COUNT(*) as cnt FROM chapters").fetchone()["cnt"]
        concepts = conn.execute("SELECT COUNT(*) as cnt FROM concepts").fetchone()["cnt"]
        experts = conn.execute("SELECT COUNT(*) as cnt FROM experts").fetchone()["cnt"]
        entries = conn.execute("SELECT COUNT(*) as cnt FROM competition_entries").fetchone()["cnt"]
        return {
            "chapters_indexed": chapters,
            "concepts_extracted": concepts,
            "experts_registered": experts,
            "competition_entries": entries,
        }
    finally:
        conn.close()


def _expert_row_to_dict(row: sqlite3.Row) -> dict:
    """Convert an expert database row to a dictionary."""
    d = dict(row)
    for field in ("capabilities", "skills", "formula", "loop_config"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
