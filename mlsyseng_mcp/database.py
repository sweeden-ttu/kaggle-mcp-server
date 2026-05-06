"""SQLite operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.getenv("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


class MLSysEngDatabase:
    """SQLite database for storing extracted knowledge, experts, and state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    concepts TEXT,  -- JSON array of extracted concepts
                    extracted_at TEXT,
                    status TEXT DEFAULT 'pending'
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

                CREATE TABLE IF NOT EXISTS convergence_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT,  -- JSON array of floats
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    metadata TEXT,      -- JSON object
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT,
                    event TEXT,
                    details TEXT,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_status ON chapters(status);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_convergence_session ON convergence_states(session_id);
            """)
            conn.commit()
        finally:
            conn.close()

    # --- Chapter operations ---

    def upsert_chapter(
        self,
        chapter_name: str,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
        status: str = "extracted",
    ) -> int:
        conn = self._get_conn()
        try:
            now = datetime.utcnow().isoformat()
            cursor = conn.execute(
                """
                INSERT INTO chapters (chapter_name, source_path, markdown_content, concepts, extracted_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_name) DO UPDATE SET
                    source_path=excluded.source_path,
                    markdown_content=excluded.markdown_content,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    status=excluded.status
                """,
                (chapter_name, source_path, markdown_content, json.dumps(concepts), now, status),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_name = ?", (chapter_name,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            return d
        finally:
            conn.close()

    def list_chapters(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM chapters WHERE status = ? ORDER BY chapter_name", (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM chapters ORDER BY chapter_name").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
                result.append(d)
            return result
        finally:
            conn.close()

    # --- Expert operations ---

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        conn = self._get_conn()
        try:
            now = datetime.utcnow().isoformat()
            cursor = conn.execute(
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
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
            if row is None:
                return None
            return self._deserialize_expert(dict(row))
        finally:
            conn.close()

    def list_experts(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
            return [self._deserialize_expert(dict(r)) for r in rows]
        finally:
            conn.close()

    def _deserialize_expert(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("capabilities", "skills"):
            d[key] = json.loads(d[key]) if d[key] else []
        for key in ("formula", "loop_config"):
            d[key] = json.loads(d[key]) if d[key] else {}
        return d

    # --- Convergence state operations ---

    def save_state(
        self,
        session_id: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        conn = self._get_conn()
        try:
            now = datetime.utcnow().isoformat()
            cursor = conn.execute(
                """
                INSERT INTO convergence_states (session_id, iteration, state_vector, l2_norm, converged, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    int(converged),
                    json.dumps(metadata or {}),
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_session_states(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM convergence_states WHERE session_id = ? ORDER BY iteration",
                (session_id,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["state_vector"] = json.loads(d["state_vector"]) if d["state_vector"] else []
                d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
                d["converged"] = bool(d["converged"])
                result.append(d)
            return result
        finally:
            conn.close()

    # --- Extraction log ---

    def log_event(self, chapter_name: str, event: str, details: str = ""):
        conn = self._get_conn()
        try:
            now = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO extraction_log (chapter_name, event, details, created_at) VALUES (?, ?, ?, ?)",
                (chapter_name, event, details, now),
            )
            conn.commit()
        finally:
            conn.close()

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            chapters_total = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            chapters_extracted = conn.execute(
                "SELECT COUNT(*) FROM chapters WHERE status = 'extracted'"
            ).fetchone()[0]
            experts_total = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM convergence_states"
            ).fetchone()[0]
            return {
                "chapters_total": chapters_total,
                "chapters_extracted": chapters_extracted,
                "experts_total": experts_total,
                "convergence_sessions": sessions,
            }
        finally:
            conn.close()
