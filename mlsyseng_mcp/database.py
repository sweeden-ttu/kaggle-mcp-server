"""SQLite database operations for MLSysEng MoE system.

Manages chapter content, expert definitions, extraction status,
and convergence loop state.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = _ensure_parent(db_path or DEFAULT_DB_PATH)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
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

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    source_path TEXT,
                    content_md TEXT,
                    concepts TEXT,  -- JSON array
                    extracted_at TEXT,
                    page_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,  -- JSON array
                    skills TEXT,        -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_number TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    pages_extracted INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS convergence_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT,   -- JSON array of floats
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    timestamp TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(chapter_number);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_convergence_comp ON convergence_state(competition);
            """)

    # ── Chapter operations ──

    def upsert_chapter(
        self,
        chapter_number: str,
        title: str,
        content_md: str,
        source_path: str = "",
        concepts: Optional[List[str]] = None,
        page_count: int = 0,
    ) -> int:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO chapters (chapter_number, title, source_path, content_md,
                                         concepts, extracted_at, page_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_number) DO UPDATE SET
                       title=excluded.title,
                       source_path=excluded.source_path,
                       content_md=excluded.content_md,
                       concepts=excluded.concepts,
                       extracted_at=excluded.extracted_at,
                       page_count=excluded.page_count""",
                (
                    chapter_number,
                    title,
                    source_path,
                    content_md,
                    json.dumps(concepts or []),
                    datetime.now(timezone.utc).isoformat(),
                    page_count,
                ),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_number = ?", (chapter_number,)
            ).fetchone()
            return row["id"]

    def get_chapter(self, chapter_number: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_number = ?", (chapter_number,)
            ).fetchone()
            return dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_number"
            ).fetchall()
            return [dict(r) for r in rows]

    def search_chapters(self, query: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters WHERE content_md LIKE ? OR title LIKE ?",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Expert operations ──

    def upsert_expert(
        self,
        expert_name: str,
        slug: str,
        chapter_id: Optional[int] = None,
        capabilities: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
        strategy: str = "",
        formula: Optional[Dict[str, Any]] = None,
        loop_config: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                        skills, strategy, formula, loop_config, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(slug) DO UPDATE SET
                       expert_name=excluded.expert_name,
                       chapter_id=excluded.chapter_id,
                       capabilities=excluded.capabilities,
                       skills=excluded.skills,
                       strategy=excluded.strategy,
                       formula=excluded.formula,
                       loop_config=excluded.loop_config""",
                (
                    expert_name,
                    slug,
                    chapter_id,
                    json.dumps(capabilities or []),
                    json.dumps(skills or []),
                    strategy,
                    json.dumps(formula or {}),
                    json.dumps(loop_config or {}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ("capabilities", "skills", "formula", "loop_config"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return d

    def list_experts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                for k in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(k):
                        try:
                            d[k] = json.loads(d[k])
                        except (json.JSONDecodeError, TypeError):
                            pass
                results.append(d)
            return results

    # ── Extraction status ──

    def set_extraction_status(
        self,
        chapter_number: str,
        status: str,
        error_message: str = "",
        pages_extracted: int = 0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO extraction_status
                       (chapter_number, status, error_message, started_at, completed_at, pages_extracted)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    chapter_number,
                    status,
                    error_message,
                    now if status == "running" else None,
                    now if status in ("completed", "failed") else None,
                    pages_extracted,
                ),
            )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT chapter_number, status, error_message, started_at,
                          completed_at, pages_extracted
                   FROM extraction_status
                   ORDER BY id DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Convergence state ──

    def record_convergence_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO convergence_state
                       (competition, iteration, state_vector, l2_norm, converged, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
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
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM convergence_state
                   WHERE competition = ?
                   ORDER BY iteration""",
                (competition,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("state_vector"):
                    try:
                        d["state_vector"] = json.loads(d["state_vector"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(d)
            return results

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            extraction_count = conn.execute(
                "SELECT COUNT(*) FROM extraction_status WHERE status = 'completed'"
            ).fetchone()[0]
            total_pages = conn.execute(
                "SELECT COALESCE(SUM(page_count), 0) FROM chapters"
            ).fetchone()[0]
            return {
                "chapters_indexed": chapter_count,
                "experts_registered": expert_count,
                "extractions_completed": extraction_count,
                "total_pages": total_pages,
                "db_path": self.db_path,
            }
