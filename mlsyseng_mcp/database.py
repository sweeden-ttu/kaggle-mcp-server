"""SQLite operations for chapters, concepts, experts, and extraction status."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class Database:
    """Thin wrapper around SQLite for the MoE knowledge store."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = _ensure_parent(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug        TEXT UNIQUE NOT NULL,
                    title       TEXT NOT NULL,
                    source_path TEXT,
                    markdown    TEXT,
                    extracted_at TEXT,
                    status      TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id  INTEGER NOT NULL REFERENCES chapters(id),
                    term        TEXT NOT NULL,
                    definition  TEXT,
                    category    TEXT,
                    UNIQUE(chapter_id, term)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug            TEXT UNIQUE NOT NULL,
                    expert_name     TEXT NOT NULL,
                    capabilities    TEXT,  -- JSON array
                    skills          TEXT,  -- JSON array
                    strategy        TEXT,
                    formula         TEXT,  -- JSON object
                    loop_config     TEXT,  -- JSON object
                    chapter_id      INTEGER REFERENCES chapters(id),
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id  INTEGER REFERENCES chapters(id),
                    event       TEXT NOT NULL,
                    detail      TEXT,
                    ts          TEXT DEFAULT (datetime('now'))
                );
                """
            )

    # ── Chapters ──────────────────────────────────────────────────────

    def upsert_chapter(
        self,
        slug: str,
        title: str,
        source_path: str = "",
        markdown: str = "",
        status: str = "extracted",
    ) -> int:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chapters (slug, title, source_path, markdown, extracted_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    title=excluded.title,
                    source_path=excluded.source_path,
                    markdown=excluded.markdown,
                    extracted_at=excluded.extracted_at,
                    status=excluded.status
                """,
                (slug, title, source_path, markdown, datetime.utcnow().isoformat(), status),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE slug=?", (slug,)
            ).fetchone()
            return row["id"]

    def get_chapter(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE slug=?", (slug,)
            ).fetchone()
            return dict(row) if row else None

    def list_chapters(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, slug, title, status, extracted_at FROM chapters ORDER BY slug"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Concepts ──────────────────────────────────────────────────────

    def add_concept(
        self,
        chapter_id: int,
        term: str,
        definition: str = "",
        category: str = "general",
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO concepts (chapter_id, term, definition, category)
                VALUES (?, ?, ?, ?)
                """,
                (chapter_id, term, definition, category),
            )

    def get_concepts(self, chapter_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM concepts WHERE chapter_id=?", (chapter_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def search_concepts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.*, ch.slug AS chapter_slug, ch.title AS chapter_title
                FROM concepts c JOIN chapters ch ON c.chapter_id = ch.id
                WHERE c.term LIKE ? OR c.definition LIKE ?
                ORDER BY c.term LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Experts ───────────────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        slug = expert["slug"]
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO experts (slug, expert_name, capabilities, skills, strategy,
                                     formula, loop_config, chapter_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    expert_name=excluded.expert_name,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config,
                    chapter_id=excluded.chapter_id
                """,
                (
                    slug,
                    expert["expert_name"],
                    json.dumps(expert.get("capabilities", [])),
                    json.dumps(expert.get("skills", [])),
                    expert.get("strategy", ""),
                    json.dumps(expert.get("formula", {})),
                    json.dumps(expert.get("loop_config", {})),
                    expert.get("chapter_id"),
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE slug=?", (slug,)
            ).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug=?", (slug,)
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
                "SELECT * FROM experts ORDER BY slug"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                for key in ("capabilities", "skills", "formula", "loop_config"):
                    if d.get(key):
                        d[key] = json.loads(d[key])
                results.append(d)
            return results

    # ── Extraction log ────────────────────────────────────────────────

    def log_event(self, chapter_id: Optional[int], event: str, detail: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO extraction_log (chapter_id, event, detail) VALUES (?, ?, ?)",
                (chapter_id, event, detail),
            )

    def get_extraction_status(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM chapters").fetchone()["n"]
            done = conn.execute(
                "SELECT COUNT(*) AS n FROM chapters WHERE status='extracted'"
            ).fetchone()["n"]
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM chapters WHERE status='pending'"
            ).fetchone()["n"]
            failed = conn.execute(
                "SELECT COUNT(*) AS n FROM chapters WHERE status='failed'"
            ).fetchone()["n"]
            experts_count = conn.execute(
                "SELECT COUNT(*) AS n FROM experts"
            ).fetchone()["n"]
            concepts_count = conn.execute(
                "SELECT COUNT(*) AS n FROM concepts"
            ).fetchone()["n"]
            return {
                "total_chapters": total,
                "extracted": done,
                "pending": pending,
                "failed": failed,
                "experts_registered": experts_count,
                "concepts_indexed": concepts_count,
            }

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_extraction_status()
        with self._conn() as conn:
            recent_log = conn.execute(
                "SELECT * FROM extraction_log ORDER BY ts DESC LIMIT 10"
            ).fetchall()
            status["recent_events"] = [dict(r) for r in recent_log]
        return status
