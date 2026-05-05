"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.openclaw/workspace/mlsyseng/mlsyseng.db")
)


@dataclass
class ChapterRecord:
    chapter_id: str
    title: str
    source_pdf: str
    content_md: str
    concepts: List[str] = field(default_factory=list)
    extracted_at: str = ""
    word_count: int = 0

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.now(timezone.utc).isoformat()
        if not self.word_count:
            self.word_count = len(self.content_md.split())


@dataclass
class ExpertRecord:
    expert_name: str
    slug: str
    chapter_id: str
    capabilities: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    strategy: str = "Baseline → EDA → Feature Engineering → Model Selection → Submit"
    formula: Dict[str, Any] = field(default_factory=dict)
    loop_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.formula:
            self.formula = {
                "objective": "minimize_validation_loss",
                "function": "L = f(X, θ, α)",
                "metrics": ["accuracy", "f1_score"],
            }
        if not self.loop_config:
            self.loop_config = {
                "objective": "minimize_validation_loss",
                "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
                "epsilon": 0.001,
                "max_iterations": 10,
                "patience": 3,
            }


class MLSysEngDatabase:
    """SQLite wrapper for storing extracted chapters, experts, and convergence state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self):
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id   TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    source_pdf   TEXT NOT NULL,
                    content_md   TEXT NOT NULL,
                    concepts     TEXT DEFAULT '[]',
                    extracted_at TEXT NOT NULL,
                    word_count   INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    expert_name TEXT PRIMARY KEY,
                    slug        TEXT UNIQUE NOT NULL,
                    chapter_id  TEXT NOT NULL REFERENCES chapters(chapter_id),
                    capabilities TEXT DEFAULT '[]',
                    skills       TEXT DEFAULT '[]',
                    strategy     TEXT DEFAULT '',
                    formula      TEXT DEFAULT '{}',
                    loop_config  TEXT DEFAULT '{}',
                    created_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS convergence_states (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition  TEXT NOT NULL,
                    iteration    INTEGER NOT NULL,
                    state_vector TEXT NOT NULL,
                    l2_norm      REAL,
                    converged    INTEGER DEFAULT 0,
                    created_at   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_convergence_competition
                    ON convergence_states(competition);
                """
            )

    def upsert_chapter(self, ch: ChapterRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chapters (chapter_id, title, source_pdf, content_md, concepts, extracted_at, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_id) DO UPDATE SET
                    title=excluded.title,
                    source_pdf=excluded.source_pdf,
                    content_md=excluded.content_md,
                    concepts=excluded.concepts,
                    extracted_at=excluded.extracted_at,
                    word_count=excluded.word_count
                """,
                (
                    ch.chapter_id,
                    ch.title,
                    ch.source_pdf,
                    ch.content_md,
                    json.dumps(ch.concepts),
                    ch.extracted_at,
                    ch.word_count,
                ),
            )

    def get_chapter(self, chapter_id: str) -> Optional[ChapterRecord]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        if row is None:
            return None
        return ChapterRecord(
            chapter_id=row["chapter_id"],
            title=row["title"],
            source_pdf=row["source_pdf"],
            content_md=row["content_md"],
            concepts=json.loads(row["concepts"]),
            extracted_at=row["extracted_at"],
            word_count=row["word_count"],
        )

    def list_chapters(self) -> List[ChapterRecord]:
        rows = self.conn.execute("SELECT * FROM chapters ORDER BY chapter_id").fetchall()
        return [
            ChapterRecord(
                chapter_id=r["chapter_id"],
                title=r["title"],
                source_pdf=r["source_pdf"],
                content_md=r["content_md"],
                concepts=json.loads(r["concepts"]),
                extracted_at=r["extracted_at"],
                word_count=r["word_count"],
            )
            for r in rows
        ]

    def upsert_expert(self, expert: ExpertRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name) DO UPDATE SET
                    slug=excluded.slug,
                    chapter_id=excluded.chapter_id,
                    capabilities=excluded.capabilities,
                    skills=excluded.skills,
                    strategy=excluded.strategy,
                    formula=excluded.formula,
                    loop_config=excluded.loop_config,
                    created_at=excluded.created_at
                """,
                (
                    expert.expert_name,
                    expert.slug,
                    expert.chapter_id,
                    json.dumps(expert.capabilities),
                    json.dumps(expert.skills),
                    expert.strategy,
                    json.dumps(expert.formula),
                    json.dumps(expert.loop_config),
                    expert.created_at,
                ),
            )

    def get_expert(self, expert_name: str) -> Optional[ExpertRecord]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_expert(row)

    def get_expert_by_slug(self, slug: str) -> Optional[ExpertRecord]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_expert(row)

    def list_experts(self) -> List[ExpertRecord]:
        rows = self.conn.execute("SELECT * FROM experts ORDER BY expert_name").fetchall()
        return [self._row_to_expert(r) for r in rows]

    def _row_to_expert(self, row: sqlite3.Row) -> ExpertRecord:
        return ExpertRecord(
            expert_name=row["expert_name"],
            slug=row["slug"],
            chapter_id=row["chapter_id"],
            capabilities=json.loads(row["capabilities"]),
            skills=json.loads(row["skills"]),
            strategy=row["strategy"],
            formula=json.loads(row["formula"]),
            loop_config=json.loads(row["loop_config"]),
            created_at=row["created_at"],
        )

    def save_convergence_state(
        self,
        competition: str,
        iteration: int,
        state_vector: List[float],
        l2_norm: float,
        converged: bool,
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO convergence_states (competition, iteration, state_vector, l2_norm, converged, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    int(converged),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return cursor.lastrowid

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM convergence_states WHERE competition = ? ORDER BY iteration",
            (competition,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "competition": r["competition"],
                "iteration": r["iteration"],
                "state_vector": json.loads(r["state_vector"]),
                "l2_norm": r["l2_norm"],
                "converged": bool(r["converged"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        chapter_count = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        expert_count = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        total_words = self.conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) FROM chapters"
        ).fetchone()[0]
        convergence_runs = self.conn.execute(
            "SELECT COUNT(DISTINCT competition) FROM convergence_states"
        ).fetchone()[0]

        return {
            "chapters_indexed": chapter_count,
            "experts_registered": expert_count,
            "total_words_extracted": total_words,
            "convergence_runs": convergence_runs,
            "database_path": self.db_path,
        }

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
