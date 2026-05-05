"""
SQLite database operations for the MLSysEng MoE system.

Stores extracted chapter content, expert definitions, embeddings metadata,
and convergence loop state.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
)


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class MoEDatabase:
    """SQLite-backed store for chapters, experts, and loop state."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        _ensure_dir(db_path)
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                chapter_id   TEXT PRIMARY KEY,
                chapter_num  INTEGER,
                title        TEXT NOT NULL,
                source_path  TEXT,
                markdown     TEXT,
                concepts     TEXT,  -- JSON array of extracted concepts
                extracted_at TEXT,
                status       TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS experts (
                expert_id    TEXT PRIMARY KEY,
                chapter_id   TEXT REFERENCES chapters(chapter_id),
                expert_name  TEXT NOT NULL,
                slug         TEXT UNIQUE NOT NULL,
                capabilities TEXT,  -- JSON array
                skills       TEXT,  -- JSON array of skill paths
                strategy     TEXT,
                formula      TEXT,  -- JSON object
                loop_config  TEXT,  -- JSON object
                created_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS loop_state (
                loop_id      TEXT PRIMARY KEY,
                competition  TEXT NOT NULL,
                iteration    INTEGER DEFAULT 0,
                state_vector TEXT,  -- JSON array of floats
                prev_vector  TEXT,
                l2_norm      REAL,
                converged    INTEGER DEFAULT 0,
                patience_cnt INTEGER DEFAULT 0,
                started_at   TEXT,
                updated_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id   TEXT,
                event        TEXT,
                detail       TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );
        """)
        self.conn.commit()

    # ── Chapter CRUD ──────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_id: str,
        chapter_num: int,
        title: str,
        source_path: str = "",
        markdown: str = "",
        concepts: Optional[List[str]] = None,
        status: str = "pending",
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO chapters (chapter_id, chapter_num, title, source_path,
                                  markdown, concepts, extracted_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                markdown     = excluded.markdown,
                concepts     = excluded.concepts,
                extracted_at = excluded.extracted_at,
                status       = excluded.status
            """,
            (
                chapter_id,
                chapter_num,
                title,
                source_path,
                markdown,
                json.dumps(concepts or []),
                now,
                status,
            ),
        )
        self.conn.commit()
        return self.get_chapter(chapter_id)

    def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["concepts"] = json.loads(d.get("concepts") or "[]")
        return d

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_num"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["concepts"] = json.loads(d.get("concepts") or "[]")
            result.append(d)
        return result

    def get_extraction_status(self) -> Dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        done = self.conn.execute(
            "SELECT COUNT(*) FROM chapters WHERE status = 'extracted'"
        ).fetchone()[0]
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM chapters WHERE status = 'pending'"
        ).fetchone()[0]
        failed = self.conn.execute(
            "SELECT COUNT(*) FROM chapters WHERE status = 'failed'"
        ).fetchone()[0]
        return {
            "total": total,
            "extracted": done,
            "pending": pending,
            "failed": failed,
        }

    # ── Expert CRUD ───────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO experts (expert_id, chapter_id, expert_name, slug,
                                 capabilities, skills, strategy, formula,
                                 loop_config, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_id) DO UPDATE SET
                capabilities = excluded.capabilities,
                skills       = excluded.skills,
                strategy     = excluded.strategy,
                formula      = excluded.formula,
                loop_config  = excluded.loop_config
            """,
            (
                expert["expert_id"],
                expert.get("chapter_id", ""),
                expert["expert_name"],
                expert["slug"],
                json.dumps(expert.get("capabilities", [])),
                json.dumps(expert.get("skills", [])),
                expert.get("strategy", ""),
                json.dumps(expert.get("formula", {})),
                json.dumps(expert.get("loop_config", {})),
                now,
            ),
        )
        self.conn.commit()
        return self.get_expert(expert["expert_id"])

    def get_expert(self, expert_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_id = ?", (expert_id,)
        ).fetchone()
        return self._row_to_expert(row) if row else None

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        return self._row_to_expert(row) if row else None

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        return [self._row_to_expert(r) for r in rows]

    def _row_to_expert(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("capabilities", "skills", "formula", "loop_config"):
            d[key] = json.loads(d.get(key) or "{}" if key in ("formula", "loop_config") else d.get(key) or "[]")
        return d

    # ── Loop state ────────────────────────────────────────────────

    def upsert_loop_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO loop_state (loop_id, competition, iteration,
                                    state_vector, prev_vector, l2_norm,
                                    converged, patience_cnt, started_at,
                                    updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(loop_id) DO UPDATE SET
                iteration    = excluded.iteration,
                state_vector = excluded.state_vector,
                prev_vector  = excluded.prev_vector,
                l2_norm      = excluded.l2_norm,
                converged    = excluded.converged,
                patience_cnt = excluded.patience_cnt,
                updated_at   = excluded.updated_at
            """,
            (
                state["loop_id"],
                state["competition"],
                state.get("iteration", 0),
                json.dumps(state.get("state_vector", [])),
                json.dumps(state.get("prev_vector", [])),
                state.get("l2_norm", 0.0),
                int(state.get("converged", False)),
                state.get("patience_cnt", 0),
                state.get("started_at", now),
                now,
            ),
        )
        self.conn.commit()
        return self.get_loop_state(state["loop_id"])

    def get_loop_state(self, loop_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM loop_state WHERE loop_id = ?", (loop_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["state_vector"] = json.loads(d.get("state_vector") or "[]")
        d["prev_vector"] = json.loads(d.get("prev_vector") or "[]")
        d["converged"] = bool(d.get("converged", 0))
        return d

    # ── Logging ───────────────────────────────────────────────────

    def log_event(self, chapter_id: str, event: str, detail: str = ""):
        self.conn.execute(
            "INSERT INTO extraction_log (chapter_id, event, detail) VALUES (?, ?, ?)",
            (chapter_id, event, detail),
        )
        self.conn.commit()

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        chapters = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        experts = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        loops = self.conn.execute("SELECT COUNT(*) FROM loop_state").fetchone()[0]
        logs = self.conn.execute("SELECT COUNT(*) FROM extraction_log").fetchone()[0]
        return {
            "chapters": chapters,
            "experts": experts,
            "active_loops": loops,
            "log_entries": logs,
        }
