"""SQLite operations for MLSysEng knowledge storage."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class Database:
    """SQLite database for storing extracted knowledge and expert definitions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        _ensure_dir(self.db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT UNIQUE NOT NULL,
                    source_path TEXT,
                    markdown_content TEXT,
                    extracted_at TEXT,
                    page_count INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL,
                    concept_name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS experts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expert_name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    chapter_id INTEGER,
                    capabilities TEXT,
                    skills TEXT,
                    strategy TEXT,
                    formula TEXT,
                    loop_config TEXT,
                    created_at TEXT,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS convergence_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT NOT NULL,
                    l2_norm REAL,
                    converged INTEGER DEFAULT 0,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_concepts_chapter
                    ON concepts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_experts_slug
                    ON experts(slug);
                CREATE INDEX IF NOT EXISTS idx_convergence_competition
                    ON convergence_states(competition, iteration);
            """)

    def upsert_chapter(
        self,
        chapter_name: str,
        source_path: str,
        markdown_content: str,
        page_count: int = 0,
    ) -> int:
        word_count = len(markdown_content.split())
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO chapters (chapter_name, source_path, markdown_content,
                                         extracted_at, page_count, word_count)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_name) DO UPDATE SET
                       source_path=excluded.source_path,
                       markdown_content=excluded.markdown_content,
                       extracted_at=excluded.extracted_at,
                       page_count=excluded.page_count,
                       word_count=excluded.word_count""",
                (chapter_name, source_path, markdown_content, now, page_count, word_count),
            )
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_name=?", (chapter_name,)
            ).fetchone()
            return row["id"]

    def add_concepts(self, chapter_id: int, concepts: List[Dict[str, Any]]) -> int:
        with self._connect() as conn:
            conn.execute("DELETE FROM concepts WHERE chapter_id=?", (chapter_id,))
            for c in concepts:
                conn.execute(
                    """INSERT INTO concepts (chapter_id, concept_name, description,
                                             category, confidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        chapter_id,
                        c.get("name", ""),
                        c.get("description", ""),
                        c.get("category", "general"),
                        c.get("confidence", 1.0),
                    ),
                )
        return len(concepts)

    def upsert_expert(self, expert_def: Dict[str, Any]) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO experts (expert_name, slug, chapter_id, capabilities,
                                        skills, strategy, formula, loop_config, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(slug) DO UPDATE SET
                       expert_name=excluded.expert_name,
                       chapter_id=excluded.chapter_id,
                       capabilities=excluded.capabilities,
                       skills=excluded.skills,
                       strategy=excluded.strategy,
                       formula=excluded.formula,
                       loop_config=excluded.loop_config""",
                (
                    expert_def["expert_name"],
                    expert_def["slug"],
                    expert_def.get("chapter_id"),
                    json.dumps(expert_def.get("capabilities", [])),
                    json.dumps(expert_def.get("skills", [])),
                    expert_def.get("strategy", ""),
                    json.dumps(expert_def.get("formula", {})),
                    json.dumps(expert_def.get("loop_config", {})),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM experts WHERE slug=?", (expert_def["slug"],)
            ).fetchone()
            return row["id"]

    def get_experts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        return [self._expert_row_to_dict(r) for r in rows]

    def get_expert_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug=?", (slug,)
            ).fetchone()
        return self._expert_row_to_dict(row) if row else None

    def get_chapters(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, chapter_name, source_path, extracted_at, page_count, word_count "
                "FROM chapters ORDER BY chapter_name"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chapter_content(self, chapter_id: int) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT markdown_content FROM chapters WHERE id=?", (chapter_id,)
            ).fetchone()
        return row["markdown_content"] if row else None

    def get_concepts_for_chapter(self, chapter_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM concepts WHERE chapter_id=? ORDER BY concept_name",
                (chapter_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.*, ch.chapter_name
                   FROM concepts c JOIN chapters ch ON c.chapter_id = ch.id
                   ORDER BY ch.chapter_name, c.concept_name"""
            ).fetchall()
        return [dict(r) for r in rows]

    def log_extraction(
        self, chapter_name: str, status: str, message: str = ""
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO extraction_log (chapter_name, status, message,
                                               started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    chapter_name,
                    status,
                    message,
                    now if status == "started" else None,
                    now if status != "started" else None,
                ),
            )

    def save_convergence_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO convergence_states
                       (competition, iteration, state_vector, l2_norm, converged, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    1 if converged else 0,
                    now,
                ),
            )

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM convergence_states
                   WHERE competition=? ORDER BY iteration""",
                (competition,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["state_vector"] = json.loads(d["state_vector"])
            d["converged"] = bool(d["converged"])
            results.append(d)
        return results

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            chapters = conn.execute("SELECT COUNT(*) as c FROM chapters").fetchone()["c"]
            concepts = conn.execute("SELECT COUNT(*) as c FROM concepts").fetchone()["c"]
            experts = conn.execute("SELECT COUNT(*) as c FROM experts").fetchone()["c"]
            total_words = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) as w FROM chapters"
            ).fetchone()["w"]
        return {
            "chapters": chapters,
            "concepts": concepts,
            "experts": experts,
            "total_words": total_words,
            "db_path": self.db_path,
        }

    @staticmethod
    def _expert_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for field in ("capabilities", "skills", "formula", "loop_config"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
