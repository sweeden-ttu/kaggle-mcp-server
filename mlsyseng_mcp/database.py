"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


def _ensure_db_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a SQLite connection, creating the database if needed."""
    path = db_path or DEFAULT_DB_PATH
    _ensure_db_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the database schema."""
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER UNIQUE,
                title TEXT NOT NULL,
                source_path TEXT,
                extracted_text TEXT,
                markdown_content TEXT,
                concepts TEXT,  -- JSON array of concept strings
                extracted_at TEXT,
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
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER,
                status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition_name TEXT NOT NULL,
                experts_used TEXT,  -- JSON array of expert slugs
                skills_applied TEXT,  -- JSON array of skill paths
                state_history TEXT,  -- JSON array of state vectors
                converged INTEGER DEFAULT 0,
                final_score REAL,
                created_at TEXT,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(chapter_number);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_extractions_status ON extractions(status);
        """)
        conn.commit()
    finally:
        conn.close()


def store_chapter(
    chapter_number: int,
    title: str,
    source_path: str,
    extracted_text: str,
    markdown_content: str,
    concepts: List[str],
    db_path: Optional[str] = None,
) -> int:
    """Store extracted chapter content. Returns the chapter ID."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO chapters
                (chapter_number, title, source_path, extracted_text,
                 markdown_content, concepts, extracted_at, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chapter_number,
                title,
                source_path,
                extracted_text,
                markdown_content,
                json.dumps(concepts),
                datetime.now().isoformat(),
                len(extracted_text.split()),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_chapter(chapter_number: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a chapter by number."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM chapters WHERE chapter_number = ?", (chapter_number,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["concepts"] = json.loads(result["concepts"]) if result["concepts"] else []
        return result
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_number"
        ).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["concepts"] = json.loads(r["concepts"]) if r["concepts"] else []
            results.append(r)
        return results
    finally:
        conn.close()


def store_expert(expert_data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Store or update an expert definition. Returns the expert ID."""
    conn = get_connection(db_path)
    now = datetime.now().isoformat()
    try:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO experts
                (expert_name, slug, chapter_id, capabilities, skills,
                 strategy, formula, loop_config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                now,
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_expert(slug: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an expert by slug."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if result[field]:
                result[field] = json.loads(result[field])
        return result
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all experts."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            r = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if r[field]:
                    r[field] = json.loads(r[field])
            results.append(r)
        return results
    finally:
        conn.close()


def record_extraction(
    chapter_id: int, status: str, error_message: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Record an extraction event."""
    conn = get_connection(db_path)
    now = datetime.now().isoformat()
    try:
        if status == "running":
            cursor = conn.execute(
                "INSERT INTO extractions (chapter_id, status, started_at) VALUES (?, ?, ?)",
                (chapter_id, status, now),
            )
        elif status in ("completed", "failed"):
            cursor = conn.execute(
                """UPDATE extractions SET status = ?, completed_at = ?, error_message = ?
                   WHERE chapter_id = ? AND status = 'running'""",
                (status, now, error_message, chapter_id),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO extractions (chapter_id, status) VALUES (?, ?)",
                (chapter_id, status),
            )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_extraction_status(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get overall extraction status."""
    conn = get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        completed = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE status = 'completed'"
        ).fetchone()[0]
        running = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE status = 'running'"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE status = 'failed'"
        ).fetchone()[0]
        return {
            "total_chapters": total,
            "completed": completed,
            "running": running,
            "failed": failed,
            "pending": total - completed - running - failed,
        }
    finally:
        conn.close()


def store_competition_entry(
    competition_name: str,
    experts_used: List[str],
    skills_applied: List[str],
    state_history: List[List[float]],
    converged: bool,
    final_score: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """Store a competition entry result."""
    conn = get_connection(db_path)
    now = datetime.now().isoformat()
    try:
        cursor = conn.execute(
            """
            INSERT INTO competition_entries
                (competition_name, experts_used, skills_applied,
                 state_history, converged, final_score, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                competition_name,
                json.dumps(experts_used),
                json.dumps(skills_applied),
                json.dumps(state_history),
                int(converged),
                final_score,
                now,
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get system statistics."""
    conn = get_connection(db_path)
    try:
        chapters = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        experts = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        entries = conn.execute("SELECT COUNT(*) FROM competition_entries").fetchone()[0]
        total_words = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) FROM chapters"
        ).fetchone()[0]
        return {
            "total_chapters": chapters,
            "total_experts": experts,
            "total_competition_entries": entries,
            "total_words_extracted": total_words,
        }
    finally:
        conn.close()
