"""SQLite database for MLSysEng MoE system.

Stores extracted chapter content, expert definitions, embeddings metadata,
and convergence loop state.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
)


class MLSysEngDB:
    """SQLite-backed storage for the MoE system."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chapters (
                chapter_id   TEXT PRIMARY KEY,
                folder_name  TEXT NOT NULL,
                title        TEXT NOT NULL,
                content_md   TEXT,
                concepts     TEXT,  -- JSON array
                pdf_path     TEXT,
                extracted_at TEXT,
                metadata     TEXT   -- JSON object
            );

            CREATE TABLE IF NOT EXISTS experts (
                expert_id    TEXT PRIMARY KEY,
                chapter_id   TEXT REFERENCES chapters(chapter_id),
                expert_name  TEXT NOT NULL,
                slug         TEXT NOT NULL UNIQUE,
                capabilities TEXT,  -- JSON array
                skills       TEXT,  -- JSON array
                strategy     TEXT,
                formula      TEXT,  -- JSON object
                loop_config  TEXT,  -- JSON object
                created_at   TEXT,
                metadata     TEXT   -- JSON object
            );

            CREATE TABLE IF NOT EXISTS embeddings_meta (
                embedding_id TEXT PRIMARY KEY,
                chapter_id   TEXT REFERENCES chapters(chapter_id),
                chunk_index  INTEGER,
                chunk_text   TEXT,
                created_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS convergence_runs (
                run_id       TEXT PRIMARY KEY,
                competition  TEXT NOT NULL,
                expert_ids   TEXT,  -- JSON array
                states       TEXT,  -- JSON array of state vectors
                converged    INTEGER DEFAULT 0,
                final_norm   REAL,
                iterations   INTEGER DEFAULT 0,
                started_at   TEXT,
                finished_at  TEXT,
                metadata     TEXT   -- JSON object
            );
            """
        )
        self.conn.commit()

    # ── Chapter CRUD ──────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_id: str,
        folder_name: str,
        title: str,
        content_md: str = "",
        concepts: Optional[List[str]] = None,
        pdf_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO chapters (chapter_id, folder_name, title, content_md,
                                  concepts, pdf_path, extracted_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                content_md   = excluded.content_md,
                concepts     = excluded.concepts,
                pdf_path     = excluded.pdf_path,
                extracted_at = excluded.extracted_at,
                metadata     = excluded.metadata
            """,
            (
                chapter_id,
                folder_name,
                title,
                content_md,
                json.dumps(concepts or []),
                pdf_path,
                now,
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()
        return self.get_chapter(chapter_id)

    def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chapters ORDER BY chapter_id")
        return [self._row_to_dict(r) for r in cur.fetchall()]

    # ── Expert CRUD ───────────────────────────────────────────────

    def upsert_expert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO experts (expert_id, chapter_id, expert_name, slug,
                                 capabilities, skills, strategy, formula,
                                 loop_config, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_id) DO UPDATE SET
                expert_name  = excluded.expert_name,
                capabilities = excluded.capabilities,
                skills       = excluded.skills,
                strategy     = excluded.strategy,
                formula      = excluded.formula,
                loop_config  = excluded.loop_config,
                metadata     = excluded.metadata
            """,
            (
                data["expert_id"],
                data.get("chapter_id", ""),
                data["expert_name"],
                data["slug"],
                json.dumps(data.get("capabilities", [])),
                json.dumps(data.get("skills", [])),
                data.get("strategy", ""),
                json.dumps(data.get("formula", {})),
                json.dumps(data.get("loop_config", {})),
                now,
                json.dumps(data.get("metadata", {})),
            ),
        )
        self.conn.commit()
        return self.get_expert(data["expert_id"])

    def get_expert(self, expert_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM experts WHERE expert_id = ?", (expert_id,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM experts WHERE slug = ?", (slug,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_experts(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM experts ORDER BY expert_name")
        return [self._row_to_dict(r) for r in cur.fetchall()]

    # ── Embeddings metadata ───────────────────────────────────────

    def add_embedding_meta(
        self, embedding_id: str, chapter_id: str, chunk_index: int, chunk_text: str
    ):
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT OR REPLACE INTO embeddings_meta
                (embedding_id, chapter_id, chunk_index, chunk_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (embedding_id, chapter_id, chunk_index, chunk_text, now),
        )
        self.conn.commit()

    # ── Convergence runs ──────────────────────────────────────────

    def create_run(
        self, run_id: str, competition: str, expert_ids: List[str]
    ) -> Dict[str, Any]:
        cur = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO convergence_runs
                (run_id, competition, expert_ids, states, started_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, competition, json.dumps(expert_ids), "[]", now, "{}"),
        )
        self.conn.commit()
        return self.get_run(run_id)

    def update_run(
        self,
        run_id: str,
        states: Optional[List] = None,
        converged: Optional[bool] = None,
        final_norm: Optional[float] = None,
        iterations: Optional[int] = None,
        finished_at: Optional[str] = None,
    ):
        parts, params = [], []
        if states is not None:
            parts.append("states = ?")
            params.append(json.dumps(states))
        if converged is not None:
            parts.append("converged = ?")
            params.append(int(converged))
        if final_norm is not None:
            parts.append("final_norm = ?")
            params.append(final_norm)
        if iterations is not None:
            parts.append("iterations = ?")
            params.append(iterations)
        if finished_at is not None:
            parts.append("finished_at = ?")
            params.append(finished_at)
        if not parts:
            return
        params.append(run_id)
        self.conn.execute(
            f"UPDATE convergence_runs SET {', '.join(parts)} WHERE run_id = ?",
            params,
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM convergence_runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        stats = {}
        for table in ("chapters", "experts", "embeddings_meta", "convergence_runs"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cur.fetchone()[0]
        return stats

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("concepts", "capabilities", "skills", "expert_ids", "states"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        for key in ("formula", "loop_config", "metadata"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        self.conn.close()
