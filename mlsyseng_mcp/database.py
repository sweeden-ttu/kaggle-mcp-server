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
        os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
    )


class MLSysEngDB:
    """SQLite database for storing extracted knowledge, experts, and state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self):
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array
                    extracted_at TEXT,
                    page_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,  -- JSON array
                    skills TEXT,        -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS embeddings_meta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER REFERENCES chapters(id),
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    embedding_id TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS convergence_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER,
                    state_vector TEXT,   -- JSON array
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    error_message TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );
                """
            )

    # ── Chapter CRUD ────────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_name: str,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
        page_count: int = 0,
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chapters (chapter_name, source_path, markdown_content, concepts, extracted_at, page_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_name) DO UPDATE SET
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    page_count=excluded.page_count
                """,
                (chapter_name, source_path, markdown_content, json.dumps(concepts), now, page_count),
            )
        row = self.conn.execute(
            "SELECT id FROM chapters WHERE chapter_name=?", (chapter_name,)
        ).fetchone()
        return row["id"]

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_name=?", (chapter_name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, json_fields=["concepts"])

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM chapters ORDER BY chapter_name").fetchall()
        return [self._row_to_dict(r, json_fields=["concepts"]) for r in rows]

    # ── Expert CRUD ─────────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name) DO UPDATE SET
                    slug=excluded.slug,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config,
                    updated_at=excluded.updated_at
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
        row = self.conn.execute(
            "SELECT id FROM experts WHERE expert_name=?", (expert["expert_name"],)
        ).fetchone()
        return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM experts WHERE slug=?", (slug,)).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, json_fields=["capabilities", "skills", "formula", "loop_config"])

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        return [
            self._row_to_dict(r, json_fields=["capabilities", "skills", "formula", "loop_config"])
            for r in rows
        ]

    # ── Embeddings meta ─────────────────────────────────────────────

    def insert_embedding_meta(
        self, chapter_id: int, chunk_index: int, chunk_text: str, embedding_id: str
    ):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embeddings_meta (chapter_id, chunk_index, chunk_text, embedding_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chapter_id, chunk_index, chunk_text, embedding_id, now),
            )

    def get_embedding_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM embeddings_meta").fetchone()
        return row["cnt"]

    # ── Convergence runs ────────────────────────────────────────────

    def insert_convergence_run(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
    ):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO convergence_runs (competition, iteration, state_vector, l2_norm, converged, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (competition, iteration, json.dumps(state_vector), l2_norm, int(converged), now),
            )

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM convergence_runs WHERE competition=? ORDER BY iteration",
            (competition,),
        ).fetchall()
        return [self._row_to_dict(r, json_fields=["state_vector"]) for r in rows]

    # ── Extraction status ───────────────────────────────────────────

    def update_extraction_status(
        self, chapter_name: str, status: str, progress: float = 0.0, error_message: str = ""
    ):
        now = datetime.utcnow().isoformat()
        with self.conn:
            existing = self.conn.execute(
                "SELECT id FROM extraction_status WHERE chapter_name=?", (chapter_name,)
            ).fetchone()
            if existing:
                fields = {"status": status, "progress": progress, "error_message": error_message}
                if status == "running" and progress == 0.0:
                    fields["started_at"] = now
                if status in ("completed", "failed"):
                    fields["completed_at"] = now
                set_clause = ", ".join(f"{k}=?" for k in fields)
                self.conn.execute(
                    f"UPDATE extraction_status SET {set_clause} WHERE chapter_name=?",
                    (*fields.values(), chapter_name),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO extraction_status (chapter_name, status, progress, error_message, started_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chapter_name, status, progress, error_message, now),
                )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM extraction_status ORDER BY chapter_name").fetchall()
        return [dict(r) for r in rows]

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        chapters = self.conn.execute("SELECT COUNT(*) as cnt FROM chapters").fetchone()["cnt"]
        experts = self.conn.execute("SELECT COUNT(*) as cnt FROM experts").fetchone()["cnt"]
        embeddings = self.get_embedding_count()
        runs = self.conn.execute("SELECT COUNT(*) as cnt FROM convergence_runs").fetchone()["cnt"]
        return {
            "chapters": chapters,
            "experts": experts,
            "embeddings": embeddings,
            "convergence_runs": runs,
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, json_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        d = dict(row)
        for field in json_fields or []:
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
