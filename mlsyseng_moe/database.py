"""SQLite operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
)


def get_db_path() -> str:
    path = Path(DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
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
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_name TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                chapter_id INTEGER,
                capabilities TEXT,  -- JSON array
                skills TEXT,        -- JSON array of skill paths
                strategy TEXT,
                formula TEXT,       -- JSON object
                loop_config TEXT,   -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER,
                status TEXT NOT NULL,  -- 'pending', 'extracting', 'done', 'error'
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(concept_name);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
        """)
        conn.commit()
    finally:
        conn.close()


def store_chapter(
    chapter_number: int,
    title: str,
    source_path: str,
    markdown_content: str,
    db_path: Optional[str] = None,
) -> int:
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO chapters (chapter_number, title, source_path, markdown_content, word_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chapter_number) DO UPDATE SET
                title=excluded.title,
                source_path=excluded.source_path,
                markdown_content=excluded.markdown_content,
                word_count=excluded.word_count,
                extracted_at=CURRENT_TIMESTAMP
            """,
            (chapter_number, title, source_path, markdown_content, len(markdown_content.split())),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def store_concepts(
    chapter_id: int,
    concepts: list[dict],
    db_path: Optional[str] = None,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM concepts WHERE chapter_id = ?", (chapter_id,))
        for concept in concepts:
            conn.execute(
                "INSERT INTO concepts (chapter_id, concept_name, description, category) VALUES (?, ?, ?, ?)",
                (chapter_id, concept["name"], concept.get("description", ""), concept.get("category", "")),
            )
        conn.commit()
    finally:
        conn.close()


def store_expert(expert_def: dict, db_path: Optional[str] = None) -> int:
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
                expert_def["expert_name"],
                expert_def["slug"],
                expert_def.get("chapter_id"),
                json.dumps(expert_def.get("capabilities", [])),
                json.dumps(expert_def.get("skills", [])),
                expert_def.get("strategy", ""),
                json.dumps(expert_def.get("formula", {})),
                json.dumps(expert_def.get("loop_config", {})),
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
            results.append({
                "id": row["id"],
                "expert_name": row["expert_name"],
                "slug": row["slug"],
                "chapter_id": row["chapter_id"],
                "capabilities": json.loads(row["capabilities"] or "[]"),
                "skills": json.loads(row["skills"] or "[]"),
                "strategy": row["strategy"],
                "formula": json.loads(row["formula"] or "{}"),
                "loop_config": json.loads(row["loop_config"] or "{}"),
            })
        return results
    finally:
        conn.close()


def get_expert_by_slug(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "expert_name": row["expert_name"],
            "slug": row["slug"],
            "chapter_id": row["chapter_id"],
            "capabilities": json.loads(row["capabilities"] or "[]"),
            "skills": json.loads(row["skills"] or "[]"),
            "strategy": row["strategy"],
            "formula": json.loads(row["formula"] or "{}"),
            "loop_config": json.loads(row["loop_config"] or "{}"),
        }
    finally:
        conn.close()


def get_all_chapters(db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, chapter_number, title, source_path, word_count, extracted_at FROM chapters ORDER BY chapter_number").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_chapter_content(chapter_id: int, db_path: Optional[str] = None) -> Optional[str]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT markdown_content FROM chapters WHERE id = ?", (chapter_id,)).fetchone()
        return row["markdown_content"] if row else None
    finally:
        conn.close()


def get_concepts_for_chapter(chapter_id: int, db_path: Optional[str] = None) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM concepts WHERE chapter_id = ?", (chapter_id,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def log_extraction(chapter_id: int, status: str, message: str = "", db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO extraction_log (chapter_id, status, message) VALUES (?, ?, ?)",
            (chapter_id, status, message),
        )
        conn.commit()
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    conn = get_connection(db_path)
    try:
        chapters = conn.execute("SELECT COUNT(*) as count FROM chapters").fetchone()["count"]
        concepts = conn.execute("SELECT COUNT(*) as count FROM concepts").fetchone()["count"]
        experts = conn.execute("SELECT COUNT(*) as count FROM experts").fetchone()["count"]
        total_words = conn.execute("SELECT COALESCE(SUM(word_count), 0) as total FROM chapters").fetchone()["total"]
        return {
            "chapters_indexed": chapters,
            "concepts_extracted": concepts,
            "experts_registered": experts,
            "total_words": total_words,
        }
    finally:
        conn.close()
