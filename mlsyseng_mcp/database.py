"""SQLite database operations for MLSysEng MoE system.

Manages chapter content, expert definitions, extraction status,
and convergence state persistence.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        str(Path.home() / ".openclaw" / "workspace" / "mlsyseng" / "mlsyseng.db"),
    )


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number INTEGER UNIQUE,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array
                    extracted_at TEXT,
                    word_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
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

                CREATE TABLE IF NOT EXISTS extraction_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number INTEGER,
                    status TEXT DEFAULT 'pending',  -- pending, extracting, completed, failed
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    pages_extracted INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS convergence_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT,  -- JSON array of floats
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    timestamp TEXT,
                    metadata TEXT  -- JSON object
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(chapter_number);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_convergence_competition ON convergence_state(competition, iteration);
            """)

    def upsert_chapter(
        self,
        chapter_number: int,
        title: str,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
    ) -> int:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter_number, title, source_path, markdown_content, concepts, extracted_at, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_number) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    word_count=excluded.word_count
                """,
                (
                    chapter_number,
                    title,
                    source_path,
                    markdown_content,
                    json.dumps(concepts),
                    datetime.utcnow().isoformat(),
                    len(markdown_content.split()),
                ),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_number = ?", (chapter_number,)
            ).fetchone()
            return row["id"]

    def get_chapter(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_number = ?", (chapter_number,)
            ).fetchone()
            if row:
                result = dict(row)
                result["concepts"] = json.loads(result["concepts"] or "[]")
                return result
            return None

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_number"
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["concepts"] = json.loads(r["concepts"] or "[]")
                results.append(r)
            return results

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    expert_name=excluded.expert_name,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config,
                    updated_at=excluded.updated_at
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
            row = conn.execute(
                "SELECT id FROM experts WHERE slug = ?", (expert_data["slug"],)
            ).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                result = dict(row)
                result["capabilities"] = json.loads(result["capabilities"] or "[]")
                result["skills"] = json.loads(result["skills"] or "[]")
                result["formula"] = json.loads(result["formula"] or "{}")
                result["loop_config"] = json.loads(result["loop_config"] or "{}")
                return result
            return None

    def get_all_experts(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["capabilities"] = json.loads(r["capabilities"] or "[]")
                r["skills"] = json.loads(r["skills"] or "[]")
                r["formula"] = json.loads(r["formula"] or "{}")
                r["loop_config"] = json.loads(r["loop_config"] or "{}")
                results.append(r)
            return results

    def set_extraction_status(
        self,
        chapter_number: int,
        status: str,
        error_message: Optional[str] = None,
        pages_extracted: int = 0,
    ):
        now = datetime.utcnow().isoformat()
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM extraction_status WHERE chapter_number = ?",
                (chapter_number,),
            ).fetchone()

            if existing:
                updates = ["status = ?", "error_message = ?", "pages_extracted = ?"]
                params: list = [status, error_message, pages_extracted]
                if status == "extracting":
                    updates.append("started_at = ?")
                    params.append(now)
                elif status in ("completed", "failed"):
                    updates.append("completed_at = ?")
                    params.append(now)
                params.append(existing["id"])
                conn.execute(
                    f"UPDATE extraction_status SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO extraction_status (chapter_number, status, started_at, error_message, pages_extracted)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chapter_number,
                        status,
                        now if status == "extracting" else None,
                        error_message,
                        pages_extracted,
                    ),
                )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM extraction_status ORDER BY chapter_number"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_convergence_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO convergence_state (competition, iteration, state_vector, l2_norm, converged, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    1 if converged else 0,
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM convergence_state WHERE competition = ? ORDER BY iteration",
                (competition,),
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["state_vector"] = json.loads(r["state_vector"] or "[]")
                r["metadata"] = json.loads(r["metadata"] or "{}")
                r["converged"] = bool(r["converged"])
                results.append(r)
            return results

    def get_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            chapters_count = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            experts_count = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            total_words = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) as c FROM chapters"
            ).fetchone()["c"]
            completed_extractions = conn.execute(
                "SELECT COUNT(*) as c FROM extraction_status WHERE status = 'completed'"
            ).fetchone()["c"]

            return {
                "chapters_indexed": chapters_count,
                "experts_registered": experts_count,
                "total_words_extracted": total_words,
                "completed_extractions": completed_extractions,
                "db_path": self.db_path,
            }
