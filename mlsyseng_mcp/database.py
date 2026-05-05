"""SQLite database operations for the MLSysEng MoE system.

Stores extracted chapter content, expert definitions, embeddings metadata,
and convergence loop state.
"""

import json
import os
import sqlite3
from datetime import datetime
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
    """SQLite-backed storage for the MoE knowledge base."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = _ensure_parent(db_path or DEFAULT_DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chapters (
                chapter_id   TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                folder_path  TEXT,
                pdf_path     TEXT,
                content_md   TEXT,
                concepts     TEXT,  -- JSON list
                extracted_at TEXT,
                status       TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS experts (
                expert_id    TEXT PRIMARY KEY,
                chapter_id   TEXT,
                expert_name  TEXT NOT NULL,
                slug         TEXT NOT NULL UNIQUE,
                capabilities TEXT,  -- JSON list
                skills       TEXT,  -- JSON list
                strategy     TEXT,
                formula      TEXT,  -- JSON object
                loop_config  TEXT,  -- JSON object
                created_at   TEXT,
                updated_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS embeddings_meta (
                embedding_id TEXT PRIMARY KEY,
                chapter_id   TEXT,
                chunk_index  INTEGER,
                chunk_text   TEXT,
                model_name   TEXT,
                created_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS loop_states (
                loop_id       TEXT,
                iteration     INTEGER,
                state_vector  TEXT,  -- JSON list of floats
                metrics       TEXT,  -- JSON object
                delta_norm    REAL,
                converged     INTEGER DEFAULT 0,
                created_at    TEXT,
                PRIMARY KEY (loop_id, iteration)
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                entry_id      TEXT PRIMARY KEY,
                competition   TEXT NOT NULL,
                experts_used  TEXT,  -- JSON list of expert slugs
                skills_used   TEXT,  -- JSON list of skill paths
                notebook_path TEXT,
                status        TEXT DEFAULT 'draft',
                created_at    TEXT,
                updated_at    TEXT
            );
            """
        )
        self.conn.commit()

    # ── Chapter CRUD ──────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_id: str,
        title: str,
        folder_path: str = "",
        pdf_path: str = "",
        content_md: str = "",
        concepts: Optional[List[str]] = None,
        status: str = "pending",
    ) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        self.conn.execute(
            """
            INSERT INTO chapters (chapter_id, title, folder_path, pdf_path,
                                  content_md, concepts, extracted_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                title=excluded.title,
                folder_path=excluded.folder_path,
                pdf_path=excluded.pdf_path,
                content_md=excluded.content_md,
                concepts=excluded.concepts,
                extracted_at=excluded.extracted_at,
                status=excluded.status
            """,
            (
                chapter_id,
                title,
                folder_path,
                pdf_path,
                content_md,
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
        d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
        return d

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM chapters ORDER BY title").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            result.append(d)
        return result

    # ── Expert CRUD ───────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        eid = expert["expert_id"]
        self.conn.execute(
            """
            INSERT INTO experts (expert_id, chapter_id, expert_name, slug,
                                 capabilities, skills, strategy, formula,
                                 loop_config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_id) DO UPDATE SET
                expert_name=excluded.expert_name,
                slug=excluded.slug,
                capabilities=excluded.capabilities,
                skills=excluded.skills,
                strategy=excluded.strategy,
                formula=excluded.formula,
                loop_config=excluded.loop_config,
                updated_at=excluded.updated_at
            """,
            (
                eid,
                expert.get("chapter_id", ""),
                expert["expert_name"],
                expert["slug"],
                json.dumps(expert.get("capabilities", [])),
                json.dumps(expert.get("skills", [])),
                expert.get("strategy", ""),
                json.dumps(expert.get("formula", {})),
                json.dumps(expert.get("loop_config", {})),
                now,
                now,
            ),
        )
        self.conn.commit()
        return self.get_expert(eid)

    def get_expert(self, expert_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_id = ?", (expert_id,)
        ).fetchone()
        return self._parse_expert_row(row)

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        return self._parse_expert_row(row)

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        return [self._parse_expert_row(r) for r in rows if r]

    def _parse_expert_row(self, row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        for key in ("capabilities", "skills"):
            d[key] = json.loads(d[key]) if d[key] else []
        for key in ("formula", "loop_config"):
            d[key] = json.loads(d[key]) if d[key] else {}
        return d

    # ── Embeddings metadata ───────────────────────────────────────

    def insert_embedding_meta(
        self,
        embedding_id: str,
        chapter_id: str,
        chunk_index: int,
        chunk_text: str,
        model_name: str,
    ) -> None:
        now = datetime.now().isoformat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings_meta
            (embedding_id, chapter_id, chunk_index, chunk_text, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (embedding_id, chapter_id, chunk_index, chunk_text, model_name, now),
        )
        self.conn.commit()

    def count_embeddings(self, chapter_id: Optional[str] = None) -> int:
        if chapter_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM embeddings_meta WHERE chapter_id = ?",
                (chapter_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM embeddings_meta"
            ).fetchone()
        return row[0] if row else 0

    # ── Loop state tracking ───────────────────────────────────────

    def save_loop_state(
        self,
        loop_id: str,
        iteration: int,
        state_vector: List[float],
        metrics: Dict[str, Any],
        delta_norm: float,
        converged: bool,
    ) -> None:
        now = datetime.now().isoformat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO loop_states
            (loop_id, iteration, state_vector, metrics, delta_norm, converged, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                loop_id,
                iteration,
                json.dumps(state_vector),
                json.dumps(metrics),
                delta_norm,
                int(converged),
                now,
            ),
        )
        self.conn.commit()

    def get_loop_history(self, loop_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM loop_states WHERE loop_id = ? ORDER BY iteration",
            (loop_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["state_vector"] = json.loads(d["state_vector"]) if d["state_vector"] else []
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            d["converged"] = bool(d["converged"])
            result.append(d)
        return result

    # ── Competition entries ────────────────────────────────────────

    def upsert_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        self.conn.execute(
            """
            INSERT INTO competition_entries
            (entry_id, competition, experts_used, skills_used, notebook_path, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                experts_used=excluded.experts_used,
                skills_used=excluded.skills_used,
                notebook_path=excluded.notebook_path,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                entry["entry_id"],
                entry["competition"],
                json.dumps(entry.get("experts_used", [])),
                json.dumps(entry.get("skills_used", [])),
                entry.get("notebook_path", ""),
                entry.get("status", "draft"),
                now,
                now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM competition_entries WHERE entry_id = ?",
            (entry["entry_id"],),
        ).fetchone()
        d = dict(row)
        for key in ("experts_used", "skills_used"):
            d[key] = json.loads(d[key]) if d[key] else []
        return d

    # ── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        chapters = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        extracted = self.conn.execute(
            "SELECT COUNT(*) FROM chapters WHERE status='extracted'"
        ).fetchone()[0]
        experts = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        embeddings = self.count_embeddings()
        entries = self.conn.execute(
            "SELECT COUNT(*) FROM competition_entries"
        ).fetchone()[0]
        return {
            "total_chapters": chapters,
            "extracted_chapters": extracted,
            "total_experts": experts,
            "total_embeddings": embeddings,
            "total_entries": entries,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
