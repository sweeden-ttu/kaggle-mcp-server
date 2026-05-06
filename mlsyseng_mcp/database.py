"""SQLite database operations for MLSysEng MoE system."""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
)


def get_db_path() -> str:
    path = Path(DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """Context manager for database connections."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
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


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the database schema."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source_path TEXT,
                content_md TEXT,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition_name TEXT NOT NULL,
                expert_id INTEGER,
                state_vector TEXT,
                iteration INTEGER DEFAULT 0,
                converged INTEGER DEFAULT 0,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (expert_id) REFERENCES experts(id)
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER,
                status TEXT NOT NULL,
                message TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(concept_name);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_entries_competition ON competition_entries(competition_name);
        """)


def insert_chapter(
    chapter_number: int,
    title: str,
    source_path: str,
    content_md: str,
    db_path: Optional[str] = None,
) -> int:
    """Insert or update a chapter record."""
    word_count = len(content_md.split())
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chapters (chapter_number, title, source_path, content_md, word_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chapter_number) DO UPDATE SET
                title=excluded.title,
                source_path=excluded.source_path,
                content_md=excluded.content_md,
                word_count=excluded.word_count,
                extracted_at=CURRENT_TIMESTAMP
            """,
            (chapter_number, title, source_path, content_md, word_count),
        )
        return cursor.lastrowid


def insert_concept(
    chapter_id: int,
    concept_name: str,
    description: str = "",
    category: str = "general",
    confidence: float = 1.0,
    db_path: Optional[str] = None,
) -> int:
    """Insert a concept for a chapter."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO concepts (chapter_id, concept_name, description, category, confidence) VALUES (?, ?, ?, ?, ?)",
            (chapter_id, concept_name, description, category, confidence),
        )
        return cursor.lastrowid


def insert_expert(expert_data: dict[str, Any], db_path: Optional[str] = None) -> int:
    """Insert or update an expert."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                expert_name=excluded.expert_name,
                capabilities=excluded.capabilities,
                skills=excluded.skills,
                strategy=excluded.strategy,
                formula=excluded.formula,
                loop_config=excluded.loop_config
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
            ),
        )
        return cursor.lastrowid


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Get all registered experts."""
    with get_connection(db_path) as conn:
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


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get an expert by slug."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        expert = dict(row)
        expert["capabilities"] = json.loads(expert["capabilities"] or "[]")
        expert["skills"] = json.loads(expert["skills"] or "[]")
        expert["formula"] = json.loads(expert["formula"] or "{}")
        expert["loop_config"] = json.loads(expert["loop_config"] or "{}")
        return expert


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    """Get all chapters."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_number").fetchall()
        return [dict(row) for row in rows]


def get_chapter_concepts(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    """Get concepts for a chapter."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM concepts WHERE chapter_id = ? ORDER BY concept_name",
            (chapter_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def log_extraction(
    chapter_id: int,
    status: str,
    message: str = "",
    db_path: Optional[str] = None,
) -> int:
    """Log an extraction event."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO extraction_log (chapter_id, status, message) VALUES (?, ?, ?)",
            (chapter_id, status, message),
        )
        return cursor.lastrowid


def get_extraction_status(db_path: Optional[str] = None) -> dict:
    """Get overall extraction status."""
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        concepts_count = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        experts_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        recent_logs = conn.execute(
            "SELECT * FROM extraction_log ORDER BY started_at DESC LIMIT 10"
        ).fetchall()
        return {
            "chapters_indexed": total,
            "concepts_extracted": concepts_count,
            "experts_registered": experts_count,
            "recent_activity": [dict(r) for r in recent_logs],
        }


def get_stats(db_path: Optional[str] = None) -> dict:
    """Get system statistics."""
    with get_connection(db_path) as conn:
        chapters = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        experts = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        entries = conn.execute("SELECT COUNT(*) FROM competition_entries").fetchone()[0]
        total_words = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) FROM chapters"
        ).fetchone()[0]
        return {
            "chapters": chapters,
            "concepts": concepts,
            "experts": experts,
            "competition_entries": entries,
            "total_words_indexed": total_words,
        }
