"""SQLite database operations for MLSysEng MoE system."""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


def get_db_path() -> str:
    path = os.path.expanduser(
        os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema."""
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number TEXT NOT NULL,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                content_md TEXT,
                concepts TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chapter_number)
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
                chapter_number TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS convergence_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                state_vector TEXT NOT NULL,
                l2_norm REAL,
                converged INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(chapter_number);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_convergence_competition ON convergence_state(competition);
        """)
        conn.commit()
    finally:
        conn.close()


def store_chapter(chapter_number: str, title: str, source_path: str,
                  content_md: str, concepts: list[str],
                  db_path: Optional[str] = None) -> int:
    """Store extracted chapter content."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT OR REPLACE INTO chapters
               (chapter_number, title, source_path, content_md, concepts)
               VALUES (?, ?, ?, ?, ?)""",
            (chapter_number, title, source_path, content_md, json.dumps(concepts))
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_chapter(chapter_number: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Retrieve chapter by number."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM chapters WHERE chapter_number = ?",
            (chapter_number,)
        ).fetchone()
        if row:
            result = dict(row)
            result["concepts"] = json.loads(result["concepts"]) if result["concepts"] else []
            return result
        return None
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_number").fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["concepts"] = json.loads(r["concepts"]) if r["concepts"] else []
            results.append(r)
        return results
    finally:
        conn.close()


def store_expert(expert_name: str, slug: str, chapter_id: Optional[int],
                 capabilities: list[str], skills: list[str],
                 strategy: str, formula: dict, loop_config: dict,
                 db_path: Optional[str] = None) -> int:
    """Store expert definition."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT OR REPLACE INTO experts
               (expert_name, slug, chapter_id, capabilities, skills,
                strategy, formula, loop_config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (expert_name, slug, chapter_id,
             json.dumps(capabilities), json.dumps(skills),
             strategy, json.dumps(formula), json.dumps(loop_config))
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_expert(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Retrieve expert by slug."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row:
            result = dict(row)
            result["capabilities"] = json.loads(result["capabilities"]) if result["capabilities"] else []
            result["skills"] = json.loads(result["skills"]) if result["skills"] else []
            result["formula"] = json.loads(result["formula"]) if result["formula"] else {}
            result["loop_config"] = json.loads(result["loop_config"]) if result["loop_config"] else {}
            return result
        return None
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all experts."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["capabilities"] = json.loads(r["capabilities"]) if r["capabilities"] else []
            r["skills"] = json.loads(r["skills"]) if r["skills"] else []
            r["formula"] = json.loads(r["formula"]) if r["formula"] else {}
            r["loop_config"] = json.loads(r["loop_config"]) if r["loop_config"] else {}
            results.append(r)
        return results
    finally:
        conn.close()


def log_extraction(chapter_number: str, status: str,
                   error_message: Optional[str] = None,
                   db_path: Optional[str] = None) -> None:
    """Log extraction status."""
    conn = get_connection(db_path)
    try:
        if status == "completed" or status == "failed":
            conn.execute(
                """UPDATE extraction_log
                   SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP
                   WHERE chapter_number = ? AND status = 'in_progress'""",
                (status, error_message, chapter_number)
            )
        else:
            conn.execute(
                "INSERT INTO extraction_log (chapter_number, status) VALUES (?, ?)",
                (chapter_number, status)
            )
        conn.commit()
    finally:
        conn.close()


def get_extraction_status(db_path: Optional[str] = None) -> list[dict]:
    """Get extraction status for all chapters."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT chapter_number, status, error_message, started_at, completed_at
               FROM extraction_log ORDER BY started_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def store_convergence_state(competition: str, iteration: int,
                            state_vector: list[float], l2_norm: float,
                            converged: bool,
                            db_path: Optional[str] = None) -> None:
    """Store convergence state for a competition iteration."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO convergence_state
               (competition, iteration, state_vector, l2_norm, converged)
               VALUES (?, ?, ?, ?, ?)""",
            (competition, iteration, json.dumps(state_vector), l2_norm, int(converged))
        )
        conn.commit()
    finally:
        conn.close()


def get_convergence_history(competition: str,
                            db_path: Optional[str] = None) -> list[dict]:
    """Get convergence history for a competition."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM convergence_state
               WHERE competition = ? ORDER BY iteration""",
            (competition,)
        ).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["state_vector"] = json.loads(r["state_vector"])
            r["converged"] = bool(r["converged"])
            results.append(r)
        return results
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    """Get system statistics."""
    conn = get_connection(db_path)
    try:
        chapters_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        experts_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        extractions_pending = conn.execute(
            "SELECT COUNT(*) FROM extraction_log WHERE status = 'pending'"
        ).fetchone()[0]
        extractions_completed = conn.execute(
            "SELECT COUNT(*) FROM extraction_log WHERE status = 'completed'"
        ).fetchone()[0]
        extractions_failed = conn.execute(
            "SELECT COUNT(*) FROM extraction_log WHERE status = 'failed'"
        ).fetchone()[0]
        return {
            "chapters_indexed": chapters_count,
            "experts_registered": experts_count,
            "extractions_pending": extractions_pending,
            "extractions_completed": extractions_completed,
            "extractions_failed": extractions_failed,
        }
    finally:
        conn.close()
