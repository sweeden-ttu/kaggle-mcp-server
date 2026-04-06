"""SQLite operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
    )


class Database:
    """SQLite-backed storage for extracted knowledge, experts, and state."""

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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    extracted_at TEXT,
                    concept_count INTEGER DEFAULT 0
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
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER,
                    capabilities TEXT,  -- JSON list
                    skills TEXT,        -- JSON list
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT NOT NULL,
                    status TEXT NOT NULL,  -- pending, running, completed, failed
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    pages_extracted INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT NOT NULL,  -- JSON list of floats
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter
                    ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_slug
                    ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_state_competition
                    ON state_history(competition, iteration);
                """
            )

    # --- Chapter operations ---

    def upsert_chapter(
        self,
        chapter_name: str,
        source_path: str,
        markdown_content: str,
        concept_count: int = 0,
    ) -> int:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter_name, source_path, markdown_content,
                                      extracted_at, concept_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chapter_name)
                DO UPDATE SET source_path=excluded.source_path,
                              markdown_content=excluded.markdown_content,
                              extracted_at=excluded.extracted_at,
                              concept_count=excluded.concept_count
                """,
                (
                    chapter_name,
                    source_path,
                    markdown_content,
                    datetime.now(timezone.utc).isoformat(),
                    concept_count,
                ),
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

    def list_chapters(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, chapter_name, source_path, extracted_at, concept_count "
                "FROM chapters ORDER BY chapter_name"
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Concept operations ---

    def add_concepts(self, chapter_id: int, concepts: List[Dict[str, str]]):
        with self._conn() as conn:
            conn.execute("DELETE FROM concepts WHERE chapter_id = ?", (chapter_id,))
            conn.executemany(
                """
                INSERT INTO concepts (chapter_id, concept_name, description, category)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chapter_id,
                        c["name"],
                        c.get("description", ""),
                        c.get("category", "general"),
                    )
                    for c in concepts
                ],
            )

    def get_concepts(self, chapter_id: int) -> List[Dict[str, Any]]:
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
                SELECT c.*, ch.chapter_name
                FROM concepts c
                JOIN chapters ch ON c.chapter_id = ch.id
                WHERE c.concept_name LIKE ? OR c.description LIKE ?
                ORDER BY c.concept_name
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Expert operations ---

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                     skills, strategy, formula, loop_config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name)
                DO UPDATE SET slug=excluded.slug,
                              chapter_id=excluded.chapter_id,
                              capabilities=excluded.capabilities,
                              skills=excluded.skills,
                              strategy=excluded.strategy,
                              formula=excluded.formula,
                              loop_config=excluded.loop_config
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
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE expert_name = ?",
                (expert["expert_name"],),
            ).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            for key in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            return d

    def list_experts(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for key in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(key):
                        d[key] = json.loads(d[key])
                result.append(d)
            return result

    # --- Extraction log ---

    def log_extraction(
        self,
        chapter_name: str,
        status: str,
        error_message: Optional[str] = None,
        pages_extracted: int = 0,
    ) -> int:
        with self._conn() as conn:
            now = datetime.now(timezone.utc).isoformat()
            completed_at = now if status in ("completed", "failed") else None
            conn.execute(
                """
                INSERT INTO extraction_log
                    (chapter_name, status, started_at, completed_at, error_message, pages_extracted)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chapter_name, status, now, completed_at, error_message, pages_extracted),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chapter_name, status, started_at, completed_at,
                       error_message, pages_extracted
                FROM extraction_log
                ORDER BY started_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    # --- State history ---

    def save_state(
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
                INSERT INTO state_history
                    (competition, iteration, state_vector, l2_norm, converged, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    int(converged),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_state_history(self, competition: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM state_history
                WHERE competition = ?
                ORDER BY iteration ASC
                """,
                (competition,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["state_vector"] = json.loads(d["state_vector"])
                d["converged"] = bool(d["converged"])
                result.append(d)
            return result

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            chapters = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
            experts = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            extractions = conn.execute(
                "SELECT status, COUNT(*) FROM extraction_log GROUP BY status"
            ).fetchall()
            return {
                "total_chapters": chapters,
                "total_concepts": concepts,
                "total_experts": experts,
                "extractions": {r[0]: r[1] for r in extractions},
            }
