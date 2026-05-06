"""SQLite database operations for MLSysEng MoE.

Manages chapters, experts, concepts, and extraction state.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db")


def _db_path() -> str:
    return os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)


@contextmanager
def _connect():
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_name TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                pdf_path TEXT,
                markdown_content TEXT,
                extracted_at TEXT,
                word_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                concept TEXT NOT NULL,
                description TEXT,
                category TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id),
                UNIQUE(chapter_id, concept)
            );

            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                expert_name TEXT NOT NULL,
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
                chapter_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT NOT NULL,
                expert_slug TEXT NOT NULL,
                notebook_path TEXT,
                score REAL,
                iteration INTEGER DEFAULT 0,
                state_vector TEXT,  -- JSON array
                created_at TEXT
            );
        """)


def upsert_chapter(folder_name: str, title: str, pdf_path: str,
                    markdown_content: str, word_count: int) -> int:
    """Insert or update a chapter, returning the chapter id."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO chapters (folder_name, title, pdf_path, markdown_content,
                                  extracted_at, word_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(folder_name) DO UPDATE SET
                title=excluded.title,
                pdf_path=excluded.pdf_path,
                markdown_content=excluded.markdown_content,
                extracted_at=excluded.extracted_at,
                word_count=excluded.word_count
        """, (folder_name, title, pdf_path, markdown_content,
              datetime.now(timezone.utc).isoformat(), word_count))
        cur = conn.execute("SELECT id FROM chapters WHERE folder_name=?",
                           (folder_name,))
        return cur.fetchone()["id"]


def add_concepts(chapter_id: int, concepts: List[Dict[str, str]]):
    """Bulk-insert concepts for a chapter."""
    with _connect() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO concepts (chapter_id, concept, description, category)
            VALUES (?, ?, ?, ?)
        """, [(chapter_id, c["concept"], c.get("description", ""),
               c.get("category", "general")) for c in concepts])


def upsert_expert(slug: str, expert_name: str, chapter_id: Optional[int],
                   capabilities: List[str], skills: List[str],
                   strategy: str, formula: Dict, loop_config: Dict) -> int:
    """Insert or update an expert definition."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO experts (slug, expert_name, chapter_id, capabilities,
                                 skills, strategy, formula, loop_config, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                expert_name=excluded.expert_name,
                chapter_id=excluded.chapter_id,
                capabilities=excluded.capabilities,
                skills=excluded.skills,
                strategy=excluded.strategy,
                formula=excluded.formula,
                loop_config=excluded.loop_config
        """, (slug, expert_name, chapter_id,
              json.dumps(capabilities), json.dumps(skills),
              strategy, json.dumps(formula), json.dumps(loop_config),
              datetime.now(timezone.utc).isoformat()))
        cur = conn.execute("SELECT id FROM experts WHERE slug=?", (slug,))
        return cur.fetchone()["id"]


def get_all_experts() -> List[Dict[str, Any]]:
    """Return all registered experts."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM experts ORDER BY slug").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(field):
                    d[field] = json.loads(d[field])
            result.append(d)
        return result


def get_expert(slug: str) -> Optional[Dict[str, Any]]:
    """Return a single expert by slug."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM experts WHERE slug=?",
                           (slug,)).fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d


def get_all_chapters() -> List[Dict[str, Any]]:
    """Return all chapters."""
    with _connect() as conn:
        return [dict(r) for r in
                conn.execute("SELECT * FROM chapters ORDER BY folder_name").fetchall()]


def get_chapter(chapter_id: int) -> Optional[Dict[str, Any]]:
    """Return a single chapter by id."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM chapters WHERE id=?",
                           (chapter_id,)).fetchone()
        return dict(row) if row else None


def get_concepts_for_chapter(chapter_id: int) -> List[Dict[str, str]]:
    """Return concepts for a given chapter."""
    with _connect() as conn:
        return [dict(r) for r in
                conn.execute("SELECT * FROM concepts WHERE chapter_id=? ORDER BY concept",
                             (chapter_id,)).fetchall()]


def log_extraction(chapter_id: int, status: str, error: Optional[str] = None):
    """Log extraction progress."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        if status == "started":
            conn.execute("""
                INSERT INTO extraction_log (chapter_id, status, started_at)
                VALUES (?, ?, ?)
            """, (chapter_id, status, now))
        else:
            conn.execute("""
                UPDATE extraction_log SET status=?, completed_at=?, error_message=?
                WHERE chapter_id=? AND status='started'
            """, (status, now, error, chapter_id))


def get_extraction_status() -> List[Dict[str, Any]]:
    """Return extraction status for all chapters."""
    with _connect() as conn:
        return [dict(r) for r in conn.execute("""
            SELECT c.folder_name, c.title, el.status, el.started_at,
                   el.completed_at, el.error_message
            FROM chapters c
            LEFT JOIN extraction_log el ON c.id = el.chapter_id
            ORDER BY c.folder_name
        """).fetchall()]


def save_entry(competition: str, expert_slug: str, notebook_path: str,
               score: float, iteration: int, state_vector: List[float]):
    """Save a competition entry."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO competition_entries
                (competition, expert_slug, notebook_path, score, iteration,
                 state_vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (competition, expert_slug, notebook_path, score, iteration,
              json.dumps(state_vector), datetime.now(timezone.utc).isoformat()))


def get_stats() -> Dict[str, Any]:
    """Return system statistics."""
    with _connect() as conn:
        chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
        experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
        concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
        entries = conn.execute(
            "SELECT COUNT(*) as c FROM competition_entries").fetchone()["c"]
        total_words = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) as w FROM chapters"
        ).fetchone()["w"]
        return {
            "chapters_indexed": chapters,
            "experts_registered": experts,
            "concepts_extracted": concepts,
            "competition_entries": entries,
            "total_words_extracted": total_words,
            "db_path": _db_path(),
        }
