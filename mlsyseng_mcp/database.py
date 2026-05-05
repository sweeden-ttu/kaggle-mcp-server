"""SQLite database operations for the MLSysEng MoE system.

Stores extracted chapter content, expert definitions, extraction status,
and embedding metadata.
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


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class MoEDatabase:
    """SQLite-backed store for chapters, experts, and extraction metadata."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = _ensure_parent(db_path)
        self.conn = sqlite3.connect(self.db_path)
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
                pdf_path     TEXT,
                extracted_at TEXT,
                metadata     TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS experts (
                expert_name  TEXT PRIMARY KEY,
                slug         TEXT UNIQUE NOT NULL,
                chapter_id   TEXT REFERENCES chapters(chapter_id),
                capabilities TEXT DEFAULT '[]',
                skills       TEXT DEFAULT '[]',
                strategy     TEXT DEFAULT '',
                formula      TEXT DEFAULT '{}',
                loop_config  TEXT DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                metadata     TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS extraction_status (
                chapter_id   TEXT PRIMARY KEY REFERENCES chapters(chapter_id),
                status       TEXT NOT NULL DEFAULT 'pending',
                progress     REAL DEFAULT 0.0,
                error        TEXT,
                started_at   TEXT,
                finished_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS concepts (
                concept_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id   TEXT REFERENCES chapters(chapter_id),
                term         TEXT NOT NULL,
                definition   TEXT,
                context      TEXT,
                page_number  INTEGER
            );

            CREATE TABLE IF NOT EXISTS convergence_logs (
                log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                competition  TEXT NOT NULL,
                iteration    INTEGER NOT NULL,
                state_vector TEXT NOT NULL,
                l2_norm      REAL,
                converged    INTEGER DEFAULT 0,
                timestamp    TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ── Chapter CRUD ────────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_id: str,
        folder_name: str,
        title: str,
        content_md: str = "",
        pdf_path: str = "",
        metadata: Optional[Dict] = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO chapters (chapter_id, folder_name, title, content_md, pdf_path, extracted_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                content_md   = excluded.content_md,
                pdf_path     = excluded.pdf_path,
                extracted_at = excluded.extracted_at,
                metadata     = excluded.metadata
            """,
            (
                chapter_id,
                folder_name,
                title,
                content_md,
                pdf_path,
                now,
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()

    def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM chapters ORDER BY chapter_id").fetchall()
        return [dict(r) for r in rows]

    # ── Expert CRUD ─────────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO experts
                (expert_name, slug, chapter_id, capabilities, skills, strategy,
                 formula, loop_config, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_name) DO UPDATE SET
                slug         = excluded.slug,
                chapter_id   = excluded.chapter_id,
                capabilities = excluded.capabilities,
                skills       = excluded.skills,
                strategy     = excluded.strategy,
                formula      = excluded.formula,
                loop_config  = excluded.loop_config,
                updated_at   = excluded.updated_at,
                metadata     = excluded.metadata
            """,
            (
                expert["expert_name"],
                expert["slug"],
                expert.get("chapter_id", ""),
                json.dumps(expert.get("capabilities", [])),
                json.dumps(expert.get("skills", [])),
                expert.get("strategy", ""),
                json.dumps(expert.get("formula", {})),
                json.dumps(expert.get("loop_config", {})),
                now,
                now,
                json.dumps(expert.get("metadata", {})),
            ),
        )
        self.conn.commit()

    def get_expert(self, expert_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("capabilities", "skills", "formula", "loop_config", "metadata"):
            if d.get(key):
                d[key] = json.loads(d[key])
        return d

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("capabilities", "skills", "formula", "loop_config", "metadata"):
            if d.get(key):
                d[key] = json.loads(d[key])
        return d

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for key in ("capabilities", "skills", "formula", "loop_config", "metadata"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            results.append(d)
        return results

    # ── Extraction status ───────────────────────────────────────────

    def set_extraction_status(
        self, chapter_id: str, status: str, progress: float = 0.0, error: str = ""
    ):
        now = datetime.now(timezone.utc).isoformat()
        started = now if status == "running" else None
        finished = now if status in ("done", "error") else None
        self.conn.execute(
            """
            INSERT INTO extraction_status (chapter_id, status, progress, error, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                status      = excluded.status,
                progress    = excluded.progress,
                error       = excluded.error,
                started_at  = COALESCE(excluded.started_at, extraction_status.started_at),
                finished_at = excluded.finished_at
            """,
            (chapter_id, status, progress, error, started, finished),
        )
        self.conn.commit()

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM extraction_status").fetchall()
        return [dict(r) for r in rows]

    # ── Concepts ────────────────────────────────────────────────────

    def add_concept(
        self,
        chapter_id: str,
        term: str,
        definition: str = "",
        context: str = "",
        page_number: int = 0,
    ):
        self.conn.execute(
            """
            INSERT INTO concepts (chapter_id, term, definition, context, page_number)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chapter_id, term, definition, context, page_number),
        )
        self.conn.commit()

    def search_concepts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM concepts
            WHERE term LIKE ? OR definition LIKE ? OR context LIKE ?
            ORDER BY concept_id
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_concepts_for_chapter(self, chapter_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM concepts WHERE chapter_id = ?", (chapter_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Convergence logs ────────────────────────────────────────────

    def log_convergence(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
    ):
        self.conn.execute(
            """
            INSERT INTO convergence_logs (competition, iteration, state_vector, l2_norm, converged, timestamp)
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
        self.conn.commit()

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM convergence_logs WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["state_vector"] = json.loads(d["state_vector"])
            d["converged"] = bool(d["converged"])
            results.append(d)
        return results

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        chapter_count = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        expert_count = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        concept_count = self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        extraction_rows = self.conn.execute("SELECT status, COUNT(*) FROM extraction_status GROUP BY status").fetchall()
        extraction_stats = {row[0]: row[1] for row in extraction_rows}
        return {
            "chapters": chapter_count,
            "experts": expert_count,
            "concepts": concept_count,
            "extraction_status": extraction_stats,
        }

    def close(self):
        self.conn.close()
