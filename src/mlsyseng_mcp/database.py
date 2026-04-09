"""SQLite operations for MLSysEng MoE system.

Manages chapter content, expert definitions, concepts, and extraction status.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
    )


class Database:
    """SQLite database for MLSysEng MoE state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    page_count INTEGER DEFAULT 0,
                    extracted_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept_name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id),
                    UNIQUE(chapter_id, concept_name)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER,
                    capabilities TEXT,       -- JSON array
                    skills TEXT,             -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,            -- JSON object
                    loop_config TEXT,        -- JSON object
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    pages_extracted INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS convergence_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT,        -- JSON array of floats
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    timestamp TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter
                    ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_chapter
                    ON experts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_convergence_comp
                    ON convergence_log(competition, iteration);
            """)

    # -- Chapter operations ---------------------------------------------------

    def upsert_chapter(
        self,
        chapter_name: str,
        slug: str,
        source_path: str,
        markdown_content: str,
        page_count: int = 0,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter_name, slug, source_path, markdown_content, page_count, extracted_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_name) DO UPDATE SET
                    markdown_content = excluded.markdown_content,
                    page_count = excluded.page_count,
                    updated_at = excluded.updated_at
                """,
                (chapter_name, slug, source_path, markdown_content, page_count, now, now),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_name = ?", (chapter_name,)
            ).fetchone()
            return row["id"]

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_name = ?", (chapter_name,)
            ).fetchone()
            return dict(row) if row else None

    def get_chapter_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE slug = ?", (slug,)
            ).fetchone()
            return dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, chapter_name, slug, source_path, page_count, extracted_at FROM chapters ORDER BY chapter_name"
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Concept operations ---------------------------------------------------

    def add_concepts(self, chapter_id: int, concepts: List[Dict[str, Any]]):
        with self._conn() as conn:
            for c in concepts:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO concepts (chapter_id, concept_name, description, category, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chapter_id,
                        c["concept_name"],
                        c.get("description", ""),
                        c.get("category", "general"),
                        c.get("confidence", 1.0),
                    ),
                )

    def get_concepts_for_chapter(self, chapter_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM concepts WHERE chapter_id = ? ORDER BY concept_name",
                (chapter_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.*, ch.chapter_name, ch.slug AS chapter_slug
                FROM concepts c
                JOIN chapters ch ON c.chapter_id = ch.id
                WHERE c.concept_name LIKE ? OR c.description LIKE ?
                ORDER BY c.confidence DESC
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Expert operations ----------------------------------------------------

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name) DO UPDATE SET
                    capabilities = excluded.capabilities,
                    skills = excluded.skills,
                    strategy = excluded.strategy,
                    formula = excluded.formula,
                    loop_config = excluded.loop_config,
                    updated_at = excluded.updated_at
                """,
                (
                    expert["expert_name"],
                    expert["slug"],
                    expert.get("chapter_id"),
                    json.dumps(expert.get("capabilities", [])),
                    json.dumps(expert.get("skills", [])),
                    expert.get("strategy", ""),
                    json.dumps(expert.get("formula", {})),
                    json.dumps(expert.get("loop_config", {})),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE expert_name = ?",
                (expert["expert_name"],),
            ).fetchone()
            return row["id"]

    def get_expert(self, expert_name: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(field):
                    d[field] = json.loads(d[field])
            return d

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            for field in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(field):
                    d[field] = json.loads(d[field])
            return d

    def list_experts(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(field):
                        d[field] = json.loads(d[field])
                result.append(d)
            return result

    # -- Extraction status ----------------------------------------------------

    def set_extraction_status(
        self,
        chapter_name: str,
        status: str,
        error_message: Optional[str] = None,
        pages_extracted: int = 0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM extraction_status WHERE chapter_name = ?",
                (chapter_name,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE extraction_status
                    SET status = ?, completed_at = ?, error_message = ?, pages_extracted = ?
                    WHERE chapter_name = ?
                    """,
                    (status, now if status in ("completed", "failed") else None,
                     error_message, pages_extracted, chapter_name),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO extraction_status (chapter_name, status, started_at, pages_extracted)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chapter_name, status, now, pages_extracted),
                )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM extraction_status ORDER BY chapter_name"
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Convergence log ------------------------------------------------------

    def log_convergence(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO convergence_log (competition, iteration, state_vector, l2_norm, converged, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    1 if converged else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM convergence_log
                WHERE competition = ?
                ORDER BY iteration
                """,
                (competition,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("state_vector"):
                    d["state_vector"] = json.loads(d["state_vector"])
                result.append(d)
            return result

    # -- Statistics -----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
            experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            extractions = conn.execute(
                "SELECT status, COUNT(*) as c FROM extraction_status GROUP BY status"
            ).fetchall()
            return {
                "total_chapters": chapters,
                "total_concepts": concepts,
                "total_experts": experts,
                "extraction_status": {r["status"]: r["c"] for r in extractions},
                "db_path": self.db_path,
            }
