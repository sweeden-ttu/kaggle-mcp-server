"""SQLite database operations for MLSysEng MoE knowledge storage."""

import os
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.expanduser("~/.openclaw/workspace/mlsyseng/mlsyseng.db"),
)


def _ensure_parent(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class Database:
    """Thin wrapper around a SQLite database used by the MoE system."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = _ensure_parent(db_path or DEFAULT_DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_name TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    markdown TEXT,
                    concepts TEXT,  -- JSON list
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT UNIQUE NOT NULL,
                    expert_name TEXT NOT NULL,
                    chapter_id INTEGER REFERENCES chapters(id),
                    capabilities TEXT,  -- JSON list
                    skills TEXT,        -- JSON list of skill paths
                    strategy TEXT,
                    formula TEXT,       -- JSON object
                    loop_config TEXT,   -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_folder TEXT NOT NULL,
                    status TEXT NOT NULL,  -- pending | running | done | error
                    message TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS competition_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    expert_slug TEXT NOT NULL,
                    notebook_path TEXT,
                    state_vector TEXT,   -- JSON list of floats
                    iteration INTEGER DEFAULT 0,
                    converged INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def upsert_chapter(
        self,
        folder_name: str,
        title: str,
        markdown: str,
        concepts: List[str],
    ) -> int:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chapters (folder_name, title, markdown, concepts)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(folder_name) DO UPDATE SET
                    title=excluded.title,
                    markdown=excluded.markdown,
                    concepts=excluded.concepts,
                    extracted_at=CURRENT_TIMESTAMP
                """,
                (folder_name, title, markdown, json.dumps(concepts)),
            )
        row = self.conn.execute(
            "SELECT id FROM chapters WHERE folder_name = ?", (folder_name,)
        ).fetchone()
        return row["id"]

    def upsert_expert(self, expert_def: Dict[str, Any]) -> int:
        slug = expert_def["slug"]
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experts (slug, expert_name, chapter_id, capabilities, skills, strategy, formula, loop_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    expert_name=excluded.expert_name,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config
                """,
                (
                    slug,
                    expert_def["expert_name"],
                    expert_def.get("chapter_id"),
                    json.dumps(expert_def.get("capabilities", [])),
                    json.dumps(expert_def.get("skills", [])),
                    expert_def.get("strategy", ""),
                    json.dumps(expert_def.get("formula", {})),
                    json.dumps(expert_def.get("loop_config", {})),
                ),
            )
        row = self.conn.execute(
            "SELECT id FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        return row["id"]

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experts ORDER BY slug").fetchall()
        results = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "slug": r["slug"],
                    "expert_name": r["expert_name"],
                    "chapter_id": r["chapter_id"],
                    "capabilities": json.loads(r["capabilities"] or "[]"),
                    "skills": json.loads(r["skills"] or "[]"),
                    "strategy": r["strategy"],
                    "formula": json.loads(r["formula"] or "{}"),
                    "loop_config": json.loads(r["loop_config"] or "{}"),
                }
            )
        return results

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "slug": row["slug"],
            "expert_name": row["expert_name"],
            "chapter_id": row["chapter_id"],
            "capabilities": json.loads(row["capabilities"] or "[]"),
            "skills": json.loads(row["skills"] or "[]"),
            "strategy": row["strategy"],
            "formula": json.loads(row["formula"] or "{}"),
            "loop_config": json.loads(row["loop_config"] or "{}"),
        }

    def list_chapters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM chapters ORDER BY folder_name").fetchall()
        results = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "folder_name": r["folder_name"],
                    "title": r["title"],
                    "concepts": json.loads(r["concepts"] or "[]"),
                    "extracted_at": r["extracted_at"],
                }
            )
        return results

    def get_chapter_markdown(self, chapter_id: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT markdown FROM chapters WHERE id = ?", (chapter_id,)
        ).fetchone()
        return row["markdown"] if row else None

    def log_extraction(self, chapter_folder: str, status: str, message: str = "") -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO extraction_log (chapter_folder, status, message)
                VALUES (?, ?, ?)
                """,
                (chapter_folder, status, message),
            )

    def update_extraction_status(
        self, chapter_folder: str, status: str, message: str = ""
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE extraction_log
                SET status = ?, message = ?, finished_at = CURRENT_TIMESTAMP
                WHERE chapter_folder = ? AND finished_at IS NULL
                ORDER BY id DESC LIMIT 1
                """,
                (status, message, chapter_folder),
            )

    def get_extraction_status(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM extraction_log ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def save_competition_entry(self, entry: Dict[str, Any]) -> int:
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO competition_entries
                    (competition, expert_slug, notebook_path, state_vector, iteration, converged)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["competition"],
                    entry["expert_slug"],
                    entry.get("notebook_path", ""),
                    json.dumps(entry.get("state_vector", [])),
                    entry.get("iteration", 0),
                    int(entry.get("converged", False)),
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_stats(self) -> Dict[str, int]:
        chapters = self.conn.execute("SELECT COUNT(*) c FROM chapters").fetchone()["c"]
        experts = self.conn.execute("SELECT COUNT(*) c FROM experts").fetchone()["c"]
        entries = self.conn.execute("SELECT COUNT(*) c FROM competition_entries").fetchone()["c"]
        return {"chapters": chapters, "experts": experts, "entries": entries}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
