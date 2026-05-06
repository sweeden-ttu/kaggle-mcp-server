"""SQLite database operations for the MLSysEng MoE system.

Stores extracted chapter content, expert definitions, and convergence state.
"""

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


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection(db_path: Optional[str] = None):
    db_path = _ensure_parent(db_path or DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
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
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_name TEXT UNIQUE NOT NULL,
                chapter_number INTEGER,
                title TEXT,
                pdf_path TEXT,
                markdown_content TEXT,
                concepts TEXT,  -- JSON array
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_name TEXT UNIQUE NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                chapter_id INTEGER REFERENCES chapters(id),
                capabilities TEXT NOT NULL,   -- JSON array
                skills TEXT NOT NULL,          -- JSON array
                strategy TEXT NOT NULL,
                formula TEXT NOT NULL,         -- JSON object
                loop_config TEXT NOT NULL,     -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS convergence_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                state_vector TEXT NOT NULL,  -- JSON array of floats
                l2_norm REAL,
                converged BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS extraction_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',  -- pending/running/done/error
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chapters_folder ON chapters(folder_name);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_convergence_comp ON convergence_state(competition, iteration);
            """
        )


# --- Chapter operations ---

def upsert_chapter(
    folder_name: str,
    chapter_number: int,
    title: str,
    pdf_path: str,
    markdown_content: str,
    concepts: list[str],
    db_path: Optional[str] = None,
) -> int:
    word_count = len(markdown_content.split()) if markdown_content else 0
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chapters (folder_name, chapter_number, title, pdf_path,
                                  markdown_content, concepts, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(folder_name) DO UPDATE SET
                chapter_number=excluded.chapter_number,
                title=excluded.title,
                pdf_path=excluded.pdf_path,
                markdown_content=excluded.markdown_content,
                concepts=excluded.concepts,
                word_count=excluded.word_count,
                extracted_at=CURRENT_TIMESTAMP
            """,
            (folder_name, chapter_number, title, pdf_path,
             markdown_content, json.dumps(concepts), word_count),
        )
        row = conn.execute(
            "SELECT id FROM chapters WHERE folder_name=?", (folder_name,)
        ).fetchone()
        return row["id"]


def get_chapter(folder_name: str, db_path: Optional[str] = None) -> Optional[dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM chapters WHERE folder_name=?", (folder_name,)
        ).fetchone()
        if row:
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            return d
    return None


def list_chapters(db_path: Optional[str] = None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_number"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            results.append(d)
        return results


# --- Expert operations ---

def upsert_expert(
    expert_name: str,
    slug: str,
    chapter_id: int,
    capabilities: list[str],
    skills: list[str],
    strategy: str,
    formula: dict,
    loop_config: dict,
    db_path: Optional[str] = None,
) -> int:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                 skills, strategy, formula, loop_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_name) DO UPDATE SET
                slug=excluded.slug,
                chapter_id=excluded.chapter_id,
                capabilities=excluded.capabilities,
                skills=excluded.skills,
                strategy=excluded.strategy,
                formula=excluded.formula,
                loop_config=excluded.loop_config
            """,
            (expert_name, slug, chapter_id,
             json.dumps(capabilities), json.dumps(skills),
             strategy, json.dumps(formula), json.dumps(loop_config)),
        )
        row = conn.execute(
            "SELECT id FROM experts WHERE expert_name=?", (expert_name,)
        ).fetchone()
        return row["id"]


def get_expert(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experts WHERE slug=?", (slug,)
        ).fetchone()
        if row:
            return _parse_expert_row(row)
    return None


def list_experts(db_path: Optional[str] = None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        return [_parse_expert_row(r) for r in rows]


def _parse_expert_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("capabilities", "skills"):
        d[field] = json.loads(d[field]) if d[field] else []
    for field in ("formula", "loop_config"):
        d[field] = json.loads(d[field]) if d[field] else {}
    return d


# --- Convergence state ---

def save_state(
    competition: str,
    iteration: int,
    state_vector: list[float],
    l2_norm: float,
    converged: bool,
    db_path: Optional[str] = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO convergence_state
                (competition, iteration, state_vector, l2_norm, converged)
            VALUES (?, ?, ?, ?, ?)
            """,
            (competition, iteration, json.dumps(state_vector), l2_norm, converged),
        )


def get_latest_state(
    competition: str, db_path: Optional[str] = None
) -> Optional[dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM convergence_state
            WHERE competition=?
            ORDER BY iteration DESC LIMIT 1
            """,
            (competition,),
        ).fetchone()
        if row:
            d = dict(row)
            d["state_vector"] = json.loads(d["state_vector"])
            return d
    return None


def get_state_history(
    competition: str, db_path: Optional[str] = None
) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM convergence_state
            WHERE competition=?
            ORDER BY iteration
            """,
            (competition,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["state_vector"] = json.loads(d["state_vector"])
            results.append(d)
        return results


# --- Extraction jobs ---

def create_extraction_job(
    folder_name: str, db_path: Optional[str] = None
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO extraction_jobs (folder_name) VALUES (?)",
            (folder_name,),
        )
        return cur.lastrowid


def update_extraction_job(
    job_id: int,
    status: str,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    ts_field = "completed_at" if status in ("done", "error") else "started_at"
    with get_connection(db_path) as conn:
        conn.execute(
            f"""
            UPDATE extraction_jobs
            SET status=?, error_message=?, {ts_field}=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (status, error_message, job_id),
        )


def get_extraction_status(db_path: Optional[str] = None) -> dict[str, Any]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM extraction_jobs GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}


def get_stats(db_path: Optional[str] = None) -> dict[str, Any]:
    with get_connection(db_path) as conn:
        chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        total_words = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) FROM chapters"
        ).fetchone()[0]
        return {
            "chapters": chapter_count,
            "experts": expert_count,
            "total_words": total_words,
        }
