"""SQLite database operations for MLSysEng MoE system."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
)


class Database:
    """SQLite database for storing extracted chapters, experts, and metadata."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    folder_path TEXT NOT NULL,
                    pdf_path TEXT,
                    content_md TEXT,
                    concepts TEXT,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,
                    skills TEXT,
                    strategy TEXT,
                    formula TEXT,
                    loop_config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER REFERENCES chapters(id),
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS competition_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition_name TEXT NOT NULL,
                    expert_ids TEXT,
                    skills_used TEXT,
                    state_history TEXT,
                    converged BOOLEAN DEFAULT FALSE,
                    final_metric REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_status ON chapters(status);
                CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_embeddings_chapter ON embeddings(chapter_id);
            """)

    def upsert_chapter(
        self,
        chapter_name: str,
        folder_path: str,
        pdf_path: Optional[str] = None,
        content_md: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        status: str = "pending",
    ) -> int:
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO chapters (chapter_name, folder_path, pdf_path, content_md, concepts, status)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_name) DO UPDATE SET
                       folder_path=excluded.folder_path,
                       pdf_path=excluded.pdf_path,
                       content_md=COALESCE(excluded.content_md, chapters.content_md),
                       concepts=COALESCE(excluded.concepts, chapters.concepts),
                       status=excluded.status,
                       extracted_at=CURRENT_TIMESTAMP""",
                (
                    chapter_name,
                    folder_path,
                    pdf_path,
                    content_md,
                    json.dumps(concepts) if concepts else None,
                    status,
                ),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_name = ?", (chapter_name,)
            ).fetchone()
            return row["id"]

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_name = ?", (chapter_name,)
            ).fetchone()
            if row:
                d = dict(row)
                if d.get("concepts"):
                    d["concepts"] = json.loads(d["concepts"])
                return d
            return None

    def list_chapters(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM chapters WHERE status = ? ORDER BY chapter_name",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chapters ORDER BY chapter_name"
                ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("concepts"):
                    d["concepts"] = json.loads(d["concepts"])
                results.append(d)
            return results

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(expert_name) DO UPDATE SET
                       slug=excluded.slug,
                       chapter_id=excluded.chapter_id,
                       capabilities=excluded.capabilities,
                       skills=excluded.skills,
                       strategy=excluded.strategy,
                       formula=excluded.formula,
                       loop_config=excluded.loop_config""",
                (
                    expert_data["expert_name"],
                    expert_data["slug"],
                    expert_data.get("chapter_id"),
                    json.dumps(expert_data.get("capabilities", [])),
                    json.dumps(expert_data.get("skills", [])),
                    expert_data.get("strategy", ""),
                    json.dumps(expert_data.get("formula", {})),
                    json.dumps(expert_data.get("loop_config", {})),
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE expert_name = ?",
                (expert_data["expert_name"],),
            ).fetchone()
            return row["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                return self._deserialize_expert(dict(row))
            return None

    def list_experts(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            return [self._deserialize_expert(dict(r)) for r in rows]

    def _deserialize_expert(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field) and isinstance(d[field], str):
                d[field] = json.loads(d[field])
        return d

    def add_embedding_chunk(
        self, chapter_id: int, chunk_text: str, chunk_index: int
    ) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO embeddings (chapter_id, chunk_text, chunk_index) VALUES (?, ?, ?)",
                (chapter_id, chunk_text, chunk_index),
            )
            return cur.lastrowid

    def get_embedding_chunks(self, chapter_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM embeddings WHERE chapter_id = ? ORDER BY chunk_index",
                (chapter_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_competition_entry(self, entry_data: Dict[str, Any]) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """INSERT INTO competition_entries
                   (competition_name, expert_ids, skills_used, state_history, converged, final_metric)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entry_data["competition_name"],
                    json.dumps(entry_data.get("expert_ids", [])),
                    json.dumps(entry_data.get("skills_used", [])),
                    json.dumps(entry_data.get("state_history", [])),
                    entry_data.get("converged", False),
                    entry_data.get("final_metric"),
                ),
            )
            return cur.lastrowid

    def get_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            extracted = conn.execute(
                "SELECT COUNT(*) as c FROM chapters WHERE status = 'extracted'"
            ).fetchone()["c"]
            experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) as c FROM embeddings").fetchone()["c"]
            entries = conn.execute(
                "SELECT COUNT(*) as c FROM competition_entries"
            ).fetchone()["c"]
            return {
                "total_chapters": chapters,
                "extracted_chapters": extracted,
                "total_experts": experts,
                "embedding_chunks": chunks,
                "competition_entries": entries,
            }
