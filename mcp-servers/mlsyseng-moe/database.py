"""SQLite database operations for MLSysEng MoE system.

Stores extracted chapter content, expert definitions, and extraction metadata.
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = _ensure_parent(db_path or DEFAULT_DB_PATH)
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
                chapter_name TEXT UNIQUE NOT NULL,
                folder_path TEXT NOT NULL,
                pdf_path TEXT,
                extracted_text TEXT,
                concepts TEXT,  -- JSON array of extracted concepts
                extraction_status TEXT DEFAULT 'pending',
                extracted_at REAL,
                created_at REAL DEFAULT (unixepoch('now'))
            );

            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_name TEXT UNIQUE NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                chapter_id INTEGER REFERENCES chapters(id),
                capabilities TEXT NOT NULL,  -- JSON array
                skills TEXT NOT NULL,         -- JSON array of skill paths
                strategy TEXT NOT NULL,
                formula TEXT NOT NULL,        -- JSON object
                loop_config TEXT NOT NULL,    -- JSON object
                created_at REAL DEFAULT (unixepoch('now')),
                updated_at REAL DEFAULT (unixepoch('now'))
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER REFERENCES chapters(id),
                status TEXT NOT NULL,
                message TEXT,
                created_at REAL DEFAULT (unixepoch('now'))
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition_slug TEXT NOT NULL,
                expert_slug TEXT NOT NULL,
                notebook_path TEXT,
                state_vector TEXT,  -- JSON array of floats
                iteration INTEGER DEFAULT 0,
                converged INTEGER DEFAULT 0,
                created_at REAL DEFAULT (unixepoch('now')),
                updated_at REAL DEFAULT (unixepoch('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_chapters_status
                ON chapters(extraction_status);
            CREATE INDEX IF NOT EXISTS idx_experts_slug
                ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_entries_competition
                ON competition_entries(competition_slug);
        """)
        conn.commit()
    finally:
        conn.close()


def upsert_chapter(
    chapter_name: str,
    folder_path: str,
    pdf_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO chapters (chapter_name, folder_path, pdf_path)
               VALUES (?, ?, ?)
               ON CONFLICT(chapter_name)
               DO UPDATE SET folder_path=excluded.folder_path,
                             pdf_path=COALESCE(excluded.pdf_path, chapters.pdf_path)
               RETURNING id""",
            (chapter_name, folder_path, pdf_path),
        )
        row = cur.fetchone()
        conn.commit()
        return row["id"]
    finally:
        conn.close()


def update_chapter_extraction(
    chapter_id: int,
    extracted_text: str,
    concepts: list[str],
    status: str = "completed",
    db_path: Optional[str] = None,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            """UPDATE chapters
               SET extracted_text=?, concepts=?, extraction_status=?, extracted_at=?
               WHERE id=?""",
            (extracted_text, json.dumps(concepts), status, time.time(), chapter_id),
        )
        conn.execute(
            "INSERT INTO extraction_log (chapter_id, status, message) VALUES (?, ?, ?)",
            (chapter_id, status, f"Extracted {len(concepts)} concepts"),
        )
        conn.commit()
    finally:
        conn.close()


def mark_chapter_failed(
    chapter_id: int, error: str, db_path: Optional[str] = None
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE chapters SET extraction_status='failed' WHERE id=?",
            (chapter_id,),
        )
        conn.execute(
            "INSERT INTO extraction_log (chapter_id, status, message) VALUES (?, 'failed', ?)",
            (chapter_id, error),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_expert(expert_def: dict[str, Any], db_path: Optional[str] = None) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO experts
                (expert_name, slug, chapter_id, capabilities, skills,
                 strategy, formula, loop_config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                 expert_name=excluded.expert_name,
                 capabilities=excluded.capabilities,
                 skills=excluded.skills,
                 strategy=excluded.strategy,
                 formula=excluded.formula,
                 loop_config=excluded.loop_config,
                 updated_at=unixepoch('now')
               RETURNING id""",
            (
                expert_def["expert_name"],
                expert_def["slug"],
                expert_def.get("chapter_id"),
                json.dumps(expert_def["capabilities"]),
                json.dumps(expert_def["skills"]),
                expert_def["strategy"],
                json.dumps(expert_def["formula"]),
                json.dumps(expert_def["loop_config"]),
            ),
        )
        row = cur.fetchone()
        conn.commit()
        return row["id"]
    finally:
        conn.close()


def get_all_experts(db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(field):
                    d[field] = json.loads(d[field])
            results.append(d)
        return results
    finally:
        conn.close()


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM experts WHERE slug=?", (slug,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_name"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("concepts"):
                d["concepts"] = json.loads(d["concepts"])
            results.append(d)
        return results
    finally:
        conn.close()


def get_chapters_by_status(
    status: str, db_path: Optional[str] = None
) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM chapters WHERE extraction_status=? ORDER BY chapter_name",
            (status,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("concepts"):
                d["concepts"] = json.loads(d["concepts"])
            results.append(d)
        return results
    finally:
        conn.close()


def upsert_competition_entry(
    competition_slug: str,
    expert_slug: str,
    notebook_path: Optional[str] = None,
    state_vector: Optional[list[float]] = None,
    iteration: int = 0,
    converged: bool = False,
    db_path: Optional[str] = None,
) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO competition_entries
                (competition_slug, expert_slug, notebook_path, state_vector,
                 iteration, converged)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING
               RETURNING id""",
            (
                competition_slug,
                expert_slug,
                notebook_path,
                json.dumps(state_vector) if state_vector else None,
                iteration,
                int(converged),
            ),
        )
        row = cur.fetchone()
        if row:
            conn.commit()
            return row["id"]
        cur = conn.execute(
            """UPDATE competition_entries
               SET notebook_path=COALESCE(?, notebook_path),
                   state_vector=COALESCE(?, state_vector),
                   iteration=?, converged=?, updated_at=unixepoch('now')
               WHERE competition_slug=? AND expert_slug=?
               RETURNING id""",
            (
                notebook_path,
                json.dumps(state_vector) if state_vector else None,
                iteration,
                int(converged),
                competition_slug,
                expert_slug,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        return row["id"]
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    conn = get_connection(db_path)
    try:
        chapters_total = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
        chapters_done = conn.execute(
            "SELECT COUNT(*) as c FROM chapters WHERE extraction_status='completed'"
        ).fetchone()["c"]
        experts_total = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
        entries_total = conn.execute(
            "SELECT COUNT(*) as c FROM competition_entries"
        ).fetchone()["c"]
        return {
            "chapters_total": chapters_total,
            "chapters_extracted": chapters_done,
            "chapters_pending": chapters_total - chapters_done,
            "experts_total": experts_total,
            "competition_entries": entries_total,
        }
    finally:
        conn.close()
