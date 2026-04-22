"""SQLite operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
    )


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or _default_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chapters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_num   INTEGER UNIQUE,
            title         TEXT NOT NULL,
            pdf_path      TEXT,
            markdown      TEXT,
            extracted_at  TEXT,
            concepts      TEXT,  -- JSON array
            metadata      TEXT   -- JSON object
        );

        CREATE TABLE IF NOT EXISTS experts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            slug          TEXT UNIQUE NOT NULL,
            expert_name   TEXT NOT NULL,
            chapter_id    INTEGER REFERENCES chapters(id),
            capabilities  TEXT,  -- JSON array
            skills        TEXT,  -- JSON array of skill paths
            strategy      TEXT,
            formula       TEXT,  -- JSON object
            loop_config   TEXT,  -- JSON object
            created_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id    INTEGER REFERENCES chapters(id),
            concept       TEXT NOT NULL,
            description   TEXT,
            category      TEXT,
            relevance     REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS extraction_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_num   INTEGER,
            status        TEXT,  -- pending, running, completed, failed
            started_at    TEXT,
            completed_at  TEXT,
            error_msg     TEXT,
            pages_count   INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
        CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
    """)
    conn.commit()


def upsert_chapter(
    conn: sqlite3.Connection,
    chapter_num: int,
    title: str,
    pdf_path: str,
    markdown: str,
    concepts: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO chapters (chapter_num, title, pdf_path, markdown, extracted_at, concepts, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chapter_num) DO UPDATE SET
            title=excluded.title,
            pdf_path=excluded.pdf_path,
            markdown=excluded.markdown,
            extracted_at=excluded.extracted_at,
            concepts=excluded.concepts,
            metadata=excluded.metadata
        """,
        (chapter_num, title, pdf_path, markdown, now, json.dumps(concepts), json.dumps(metadata or {})),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM chapters WHERE chapter_num=?", (chapter_num,)).fetchone()
    return row["id"]


def upsert_expert(
    conn: sqlite3.Connection,
    slug: str,
    expert_name: str,
    chapter_id: int,
    capabilities: List[str],
    skills: List[str],
    strategy: str,
    formula: Dict[str, Any],
    loop_config: Dict[str, Any],
) -> int:
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO experts (slug, expert_name, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            expert_name=excluded.expert_name,
            chapter_id=excluded.chapter_id,
            capabilities=excluded.capabilities,
            skills=excluded.skills,
            strategy=excluded.strategy,
            formula=excluded.formula,
            loop_config=excluded.loop_config,
            created_at=excluded.created_at
        """,
        (slug, expert_name, chapter_id, json.dumps(capabilities), json.dumps(skills),
         strategy, json.dumps(formula), json.dumps(loop_config), now),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM experts WHERE slug=?", (slug,)).fetchone()
    return row["id"]


def insert_concepts(conn: sqlite3.Connection, chapter_id: int, concepts: List[Dict[str, Any]]) -> None:
    conn.execute("DELETE FROM concepts WHERE chapter_id=?", (chapter_id,))
    for c in concepts:
        conn.execute(
            "INSERT INTO concepts (chapter_id, concept, description, category, relevance) VALUES (?, ?, ?, ?, ?)",
            (chapter_id, c["concept"], c.get("description", ""), c.get("category", "general"), c.get("relevance", 1.0)),
        )
    conn.commit()


def log_extraction(
    conn: sqlite3.Connection,
    chapter_num: int,
    status: str,
    error_msg: Optional[str] = None,
    pages_count: int = 0,
) -> None:
    now = datetime.utcnow().isoformat()
    if status in ("pending", "running"):
        conn.execute(
            "INSERT INTO extraction_log (chapter_num, status, started_at, pages_count) VALUES (?, ?, ?, ?)",
            (chapter_num, status, now, pages_count),
        )
    elif status in ("completed", "failed"):
        conn.execute(
            """UPDATE extraction_log SET status=?, completed_at=?, error_msg=?, pages_count=?
               WHERE chapter_num=? AND status='running'""",
            (status, now, error_msg, pages_count, chapter_num),
        )
    conn.commit()


def get_all_experts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM experts ORDER BY slug").fetchall()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "slug": r["slug"],
            "expert_name": r["expert_name"],
            "chapter_id": r["chapter_id"],
            "capabilities": json.loads(r["capabilities"]) if r["capabilities"] else [],
            "skills": json.loads(r["skills"]) if r["skills"] else [],
            "strategy": r["strategy"],
            "formula": json.loads(r["formula"]) if r["formula"] else {},
            "loop_config": json.loads(r["loop_config"]) if r["loop_config"] else {},
            "created_at": r["created_at"],
        })
    return result


def get_expert_by_slug(conn: sqlite3.Connection, slug: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM experts WHERE slug=?", (slug,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "slug": row["slug"],
        "expert_name": row["expert_name"],
        "chapter_id": row["chapter_id"],
        "capabilities": json.loads(row["capabilities"]) if row["capabilities"] else [],
        "skills": json.loads(row["skills"]) if row["skills"] else [],
        "strategy": row["strategy"],
        "formula": json.loads(row["formula"]) if row["formula"] else {},
        "loop_config": json.loads(row["loop_config"]) if row["loop_config"] else {},
        "created_at": row["created_at"],
    }


def get_all_chapters(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_num").fetchall()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "chapter_num": r["chapter_num"],
            "title": r["title"],
            "pdf_path": r["pdf_path"],
            "markdown": r["markdown"],
            "extracted_at": r["extracted_at"],
            "concepts": json.loads(r["concepts"]) if r["concepts"] else [],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
        })
    return result


def get_chapter_markdown(conn: sqlite3.Connection, chapter_num: int) -> Optional[str]:
    row = conn.execute("SELECT markdown FROM chapters WHERE chapter_num=?", (chapter_num,)).fetchone()
    return row["markdown"] if row else None


def get_extraction_status(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM extraction_log ORDER BY started_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
    experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
    concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
    return {"chapters": chapters, "experts": experts, "concepts": concepts}
