"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.mlsyseng/mlsyseng.db")


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class Database:
    """SQLite database for storing extracted knowledge, experts, and state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        _ensure_dir(self.db_path)
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
        cur.executescript("""
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
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_name TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS convergence_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                state_vector TEXT,  -- JSON array of floats
                metrics TEXT,       -- JSON object
                timestamp TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chapters_status ON chapters(status);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_convergence_comp ON convergence_state(competition);
        """)
        self.conn.commit()

    # ── Chapter operations ────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_name: str,
        source_path: str,
        markdown_content: str,
        concepts: List[str],
        status: str = "extracted",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO chapters (chapter_name, source_path, markdown_content,
                                  concepts, extracted_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_name) DO UPDATE SET
                source_path = excluded.source_path,
                markdown_content = excluded.markdown_content,
                concepts = excluded.concepts,
                extracted_at = excluded.extracted_at,
                status = excluded.status
            """,
            (
                chapter_name,
                source_path,
                markdown_content,
                json.dumps(concepts),
                datetime.utcnow().isoformat(),
                status,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_chapter(self, chapter_name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_name = ?", (chapter_name,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
        return d

    def list_chapters(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM chapters WHERE status = ? ORDER BY chapter_name",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_name"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["concepts"] = json.loads(d["concepts"]) if d["concepts"] else []
            result.append(d)
        return result

    # ── Expert operations ─────────────────────────────────────────────

    def upsert_expert(self, expert: Dict[str, Any]) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                 skills, strategy, formula, loop_config, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_name) DO UPDATE SET
                slug = excluded.slug,
                chapter_id = excluded.chapter_id,
                capabilities = excluded.capabilities,
                skills = excluded.skills,
                strategy = excluded.strategy,
                formula = excluded.formula,
                loop_config = excluded.loop_config
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
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return self._parse_expert_row(row)

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        return [self._parse_expert_row(r) for r in rows]

    def _parse_expert_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field):
                d[field] = json.loads(d[field])
            else:
                d[field] = [] if field in ("capabilities", "skills") else {}
        return d

    # ── Extraction log ────────────────────────────────────────────────

    def log_extraction(self, chapter_name: str, status: str, message: str = "") -> None:
        self.conn.execute(
            "INSERT INTO extraction_log (chapter_name, status, message, timestamp) VALUES (?, ?, ?, ?)",
            (chapter_name, status, message, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def get_extraction_status(self) -> Dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        extracted = self.conn.execute(
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
            "extracted": extracted,
            "pending": pending,
            "failed": failed,
        }

    # ── Convergence state ─────────────────────────────────────────────

    def save_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        metrics: Dict[str, float],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO convergence_state (competition, iteration, state_vector, metrics, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                competition,
                iteration,
                json.dumps(state_vector),
                json.dumps(metrics),
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def get_states(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM convergence_state WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["state_vector"] = json.loads(d["state_vector"]) if d["state_vector"] else []
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            result.append(d)
        return result

    def get_latest_state(self, competition: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM convergence_state WHERE competition = ? ORDER BY iteration DESC LIMIT 1",
            (competition,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["state_vector"] = json.loads(d["state_vector"]) if d["state_vector"] else []
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
        return d

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        chapters = self.get_extraction_status()
        expert_count = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        competitions = self.conn.execute(
            "SELECT DISTINCT competition FROM convergence_state"
        ).fetchall()
        return {
            "chapters": chapters,
            "experts": expert_count,
            "competitions": [r[0] for r in competitions],
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
