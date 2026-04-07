"""SQLite database operations for MLSysEng MoE.

Manages chapters, experts, extraction status, and expert definitions
in a local SQLite database.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    env = os.environ.get("SQLITE_DB_PATH")
    if env:
        return env
    return str(Path.home() / ".openclaw" / "workspace" / "mlsyseng" / "mlsyseng.db")


class Database:
    """Thin wrapper around SQLite for the MoE knowledge store."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def initialize(self):
        """Create all tables if they don't exist."""
        self.conn.executescript(_SCHEMA)

    # -- Chapter operations ---------------------------------------------------

    def upsert_chapter(
        self,
        chapter_id: str,
        title: str,
        folder_path: str,
        content_md: str,
        concepts: List[str],
        pdf_path: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO chapters (chapter_id, title, folder_path, pdf_path,
                                  content_md, concepts, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                title=excluded.title,
                folder_path=excluded.folder_path,
                pdf_path=excluded.pdf_path,
                content_md=excluded.content_md,
                concepts=excluded.concepts,
                extracted_at=excluded.extracted_at
            """,
            (
                chapter_id,
                title,
                folder_path,
                pdf_path,
                content_md,
                json.dumps(concepts),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
        return d

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM chapters ORDER BY chapter_id"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            result.append(d)
        return result

    # -- Expert operations ----------------------------------------------------

    def upsert_expert(self, expert_name: str, definition: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO experts (expert_name, slug, definition, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(expert_name) DO UPDATE SET
                slug=excluded.slug,
                definition=excluded.definition,
                updated_at=excluded.updated_at
            """,
            (
                expert_name,
                definition.get("slug", ""),
                json.dumps(definition),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_expert(self, expert_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["definition"] = json.loads(d["definition"]) if d["definition"] else {}
        return d

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["definition"] = json.loads(d["definition"]) if d["definition"] else {}
            result.append(d)
        return result

    # -- Extraction status ----------------------------------------------------

    def set_extraction_status(
        self, chapter_id: str, status: str, message: str = ""
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO extraction_status (chapter_id, status, message, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                status=excluded.status,
                message=excluded.message,
                updated_at=excluded.updated_at
            """,
            (chapter_id, status, message, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM extraction_status ORDER BY chapter_id"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Loop state -----------------------------------------------------------

    def save_loop_state(
        self, competition: str, iteration: int, state_vector: List[float], metadata: Dict[str, Any]
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO loop_states (competition, iteration, state_vector, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                competition,
                iteration,
                json.dumps(state_vector),
                json.dumps(metadata),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_loop_states(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM loop_states WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["state_vector"] = json.loads(d["state_vector"]) if d["state_vector"] else []
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
            result.append(d)
        return result

    # -- Stats ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        ch = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        ex = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        ls = self.conn.execute("SELECT COUNT(*) FROM loop_states").fetchone()[0]
        return {"chapters": ch, "experts": ex, "loop_states": ls}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    folder_path  TEXT NOT NULL,
    pdf_path     TEXT,
    content_md   TEXT,
    concepts     TEXT,
    extracted_at TEXT
);

CREATE TABLE IF NOT EXISTS experts (
    expert_name TEXT PRIMARY KEY,
    slug        TEXT NOT NULL,
    definition  TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS extraction_status (
    chapter_id TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    message    TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS loop_states (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    competition  TEXT NOT NULL,
    iteration    INTEGER NOT NULL,
    state_vector TEXT,
    metadata     TEXT,
    created_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_loop_competition ON loop_states(competition);
"""
