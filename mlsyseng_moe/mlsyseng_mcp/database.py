"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


class Database:
    """SQLite database for storing extracted knowledge, experts, and state."""

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
                    slug TEXT UNIQUE NOT NULL,
                    pdf_path TEXT,
                    extracted_text TEXT,
                    concepts TEXT,  -- JSON array
                    extraction_status TEXT DEFAULT 'pending',
                    extracted_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER,
                    capabilities TEXT,  -- JSON array
                    skills TEXT,  -- JSON array of skill paths
                    strategy TEXT,
                    formula TEXT,  -- JSON object
                    loop_config TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS competition_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition_slug TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT,  -- JSON array of floats
                    metrics TEXT,  -- JSON object
                    experts_used TEXT,  -- JSON array
                    converged INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_chapter
                    ON knowledge_chunks(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_states_competition
                    ON competition_states(competition_slug, iteration);
            """)

    def upsert_chapter(
        self,
        chapter_name: str,
        slug: str,
        pdf_path: Optional[str] = None,
        extracted_text: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        extraction_status: str = "pending",
    ) -> int:
        """Insert or update a chapter record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter_name, slug, pdf_path, extracted_text,
                                      concepts, extraction_status, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_name) DO UPDATE SET
                    pdf_path = COALESCE(excluded.pdf_path, chapters.pdf_path),
                    extracted_text = COALESCE(excluded.extracted_text, chapters.extracted_text),
                    concepts = COALESCE(excluded.concepts, chapters.concepts),
                    extraction_status = excluded.extraction_status,
                    extracted_at = CASE WHEN excluded.extraction_status = 'completed'
                                       THEN CURRENT_TIMESTAMP
                                       ELSE chapters.extracted_at END
                """,
                (
                    chapter_name,
                    slug,
                    pdf_path,
                    extracted_text,
                    json.dumps(concepts) if concepts else None,
                    extraction_status,
                    datetime.now(timezone.utc).isoformat() if extraction_status == "completed" else None,
                ),
            )
            cursor = conn.execute(
                "SELECT id FROM chapters WHERE chapter_name = ?", (chapter_name,)
            )
            return cursor.fetchone()["id"]

    def get_chapter(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get a chapter by slug."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                result = dict(row)
                if result.get("concepts"):
                    result["concepts"] = json.loads(result["concepts"])
                return result
        return None

    def list_chapters(self) -> List[Dict[str, Any]]:
        """List all chapters."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, chapter_name, slug, extraction_status, extracted_at FROM chapters"
            ).fetchall()
            return [dict(r) for r in rows]

    def store_knowledge_chunks(
        self, chapter_id: int, chunks: List[Dict[str, Any]]
    ):
        """Store extracted knowledge chunks for a chapter."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM knowledge_chunks WHERE chapter_id = ?", (chapter_id,)
            )
            conn.executemany(
                """
                INSERT INTO knowledge_chunks (chapter_id, chunk_index, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chapter_id,
                        i,
                        chunk["content"],
                        json.dumps(chunk.get("metadata", {})),
                    )
                    for i, chunk in enumerate(chunks)
                ],
            )

    def get_knowledge_chunks(self, chapter_id: int) -> List[Dict[str, Any]]:
        """Get all knowledge chunks for a chapter."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_chunks WHERE chapter_id = ? ORDER BY chunk_index",
                (chapter_id,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                if d.get("metadata"):
                    d["metadata"] = json.loads(d["metadata"])
                results.append(d)
            return results

    def upsert_expert(self, expert_data: Dict[str, Any]) -> int:
        """Insert or update an expert."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                     skills, strategy, formula, loop_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name) DO UPDATE SET
                    capabilities = excluded.capabilities,
                    skills = excluded.skills,
                    strategy = excluded.strategy,
                    formula = excluded.formula,
                    loop_config = excluded.loop_config
                """,
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
            cursor = conn.execute(
                "SELECT id FROM experts WHERE expert_name = ?",
                (expert_data["expert_name"],),
            )
            return cursor.fetchone()["id"]

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get an expert by slug."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            if row:
                result = dict(row)
                for field in ("capabilities", "skills", "formula", "loop_config"):
                    if result.get(field):
                        result[field] = json.loads(result[field])
                return result
        return None

    def list_experts(self) -> List[Dict[str, Any]]:
        """List all registered experts."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, expert_name, slug, strategy FROM experts"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_competition_state(
        self,
        competition_slug: str,
        iteration: int,
        state_vector: List[float],
        metrics: Dict[str, float],
        experts_used: List[str],
        converged: bool = False,
    ):
        """Save a competition state iteration."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO competition_states
                    (competition_slug, iteration, state_vector, metrics, experts_used, converged)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition_slug,
                    iteration,
                    json.dumps(state_vector),
                    json.dumps(metrics),
                    json.dumps(experts_used),
                    int(converged),
                ),
            )

    def get_competition_history(
        self, competition_slug: str
    ) -> List[Dict[str, Any]]:
        """Get all state iterations for a competition."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM competition_states
                WHERE competition_slug = ?
                ORDER BY iteration
                """,
                (competition_slug,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["state_vector"] = json.loads(d["state_vector"])
                d["metrics"] = json.loads(d["metrics"])
                d["experts_used"] = json.loads(d["experts_used"])
                d["converged"] = bool(d["converged"])
                results.append(d)
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]
            extracted = conn.execute(
                "SELECT COUNT(*) FROM chapters WHERE extraction_status = 'completed'"
            ).fetchone()[0]
            return {
                "total_chapters": chapter_count,
                "extracted_chapters": extracted,
                "total_experts": expert_count,
                "total_chunks": chunk_count,
            }
