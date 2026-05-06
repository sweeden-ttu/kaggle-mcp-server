"""SQLite database operations for MLSysEng MoE.

Stores extracted chapter content, expert definitions, embeddings metadata,
and convergence loop state.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ChapterRecord:
    chapter_id: str
    title: str
    slug: str
    source_path: str
    content_md: str
    concepts: list[str] = field(default_factory=list)
    extracted_at: float = 0.0
    word_count: int = 0

    def to_row(self) -> tuple:
        return (
            self.chapter_id,
            self.title,
            self.slug,
            self.source_path,
            self.content_md,
            json.dumps(self.concepts),
            self.extracted_at,
            self.word_count,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChapterRecord":
        return cls(
            chapter_id=row["chapter_id"],
            title=row["title"],
            slug=row["slug"],
            source_path=row["source_path"],
            content_md=row["content_md"],
            concepts=json.loads(row["concepts"]),
            extracted_at=row["extracted_at"],
            word_count=row["word_count"],
        )


@dataclass
class ExpertRecord:
    expert_name: str
    slug: str
    chapter_id: str
    capabilities: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    strategy: str = "Baseline → EDA → Feature Engineering → Model Selection → Submit"
    formula: dict = field(default_factory=lambda: {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    })
    loop_config: dict = field(default_factory=lambda: {
        "objective": "minimize_validation_loss",
        "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
        "epsilon": 0.001,
        "max_iterations": 10,
        "patience": 3,
    })

    def to_row(self) -> tuple:
        return (
            self.expert_name,
            self.slug,
            self.chapter_id,
            json.dumps(self.capabilities),
            json.dumps(self.skills),
            self.strategy,
            json.dumps(self.formula),
            json.dumps(self.loop_config),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ExpertRecord":
        return cls(
            expert_name=row["expert_name"],
            slug=row["slug"],
            chapter_id=row["chapter_id"],
            capabilities=json.loads(row["capabilities"]),
            skills=json.loads(row["skills"]),
            strategy=row["strategy"],
            formula=json.loads(row["formula"]),
            loop_config=json.loads(row["loop_config"]),
        )


@dataclass
class LoopState:
    loop_id: str
    competition: str
    iteration: int
    state_vector: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    converged: bool = False
    timestamp: float = 0.0

    def to_row(self) -> tuple:
        return (
            self.loop_id,
            self.competition,
            self.iteration,
            json.dumps(self.state_vector),
            json.dumps(self.metrics),
            int(self.converged),
            self.timestamp,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LoopState":
        return cls(
            loop_id=row["loop_id"],
            competition=row["competition"],
            iteration=row["iteration"],
            state_vector=json.loads(row["state_vector"]),
            metrics=json.loads(row["metrics"]),
            converged=bool(row["converged"]),
            timestamp=row["timestamp"],
        )


def _default_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH",
        os.path.expanduser("~/.mlsyseng/mlsyseng.db"),
    )


class Database:
    """SQLite database for the MLSysEng MoE system."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL,
                    content_md TEXT NOT NULL DEFAULT '',
                    concepts TEXT NOT NULL DEFAULT '[]',
                    extracted_at REAL NOT NULL DEFAULT 0,
                    word_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS experts (
                    expert_name TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    chapter_id TEXT NOT NULL,
                    capabilities TEXT NOT NULL DEFAULT '[]',
                    skills TEXT NOT NULL DEFAULT '[]',
                    strategy TEXT NOT NULL DEFAULT '',
                    formula TEXT NOT NULL DEFAULT '{}',
                    loop_config TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
                );

                CREATE TABLE IF NOT EXISTS loop_states (
                    loop_id TEXT NOT NULL,
                    competition TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_vector TEXT NOT NULL DEFAULT '[]',
                    metrics TEXT NOT NULL DEFAULT '{}',
                    converged INTEGER NOT NULL DEFAULT 0,
                    timestamp REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (loop_id, iteration)
                );

                CREATE INDEX IF NOT EXISTS idx_loop_competition
                    ON loop_states(competition);
                CREATE INDEX IF NOT EXISTS idx_chapters_slug
                    ON chapters(slug);
                CREATE INDEX IF NOT EXISTS idx_experts_chapter
                    ON experts(chapter_id);
            """)

    # -- Chapter CRUD --

    def upsert_chapter(self, chapter: ChapterRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO chapters
                   (chapter_id, title, slug, source_path, content_md,
                    concepts, extracted_at, word_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chapter_id) DO UPDATE SET
                     title=excluded.title,
                     slug=excluded.slug,
                     source_path=excluded.source_path,
                     content_md=excluded.content_md,
                     concepts=excluded.concepts,
                     extracted_at=excluded.extracted_at,
                     word_count=excluded.word_count""",
                chapter.to_row(),
            )

    def get_chapter(self, chapter_id: str) -> Optional[ChapterRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_id = ?",
                (chapter_id,),
            ).fetchone()
            return ChapterRecord.from_row(row) if row else None

    def get_chapter_by_slug(self, slug: str) -> Optional[ChapterRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE slug = ?", (slug,)
            ).fetchone()
            return ChapterRecord.from_row(row) if row else None

    def list_chapters(self) -> list[ChapterRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_id"
            ).fetchall()
            return [ChapterRecord.from_row(r) for r in rows]

    # -- Expert CRUD --

    def upsert_expert(self, expert: ExpertRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO experts
                   (expert_name, slug, chapter_id, capabilities, skills,
                    strategy, formula, loop_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(expert_name) DO UPDATE SET
                     slug=excluded.slug,
                     chapter_id=excluded.chapter_id,
                     capabilities=excluded.capabilities,
                     skills=excluded.skills,
                     strategy=excluded.strategy,
                     formula=excluded.formula,
                     loop_config=excluded.loop_config""",
                expert.to_row(),
            )

    def get_expert(self, expert_name: str) -> Optional[ExpertRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE expert_name = ?",
                (expert_name,),
            ).fetchone()
            return ExpertRecord.from_row(row) if row else None

    def get_expert_by_slug(self, slug: str) -> Optional[ExpertRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
            return ExpertRecord.from_row(row) if row else None

    def list_experts(self) -> list[ExpertRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
            return [ExpertRecord.from_row(r) for r in rows]

    # -- Loop State --

    def save_loop_state(self, state: LoopState) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO loop_states
                   (loop_id, competition, iteration, state_vector,
                    metrics, converged, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(loop_id, iteration) DO UPDATE SET
                     state_vector=excluded.state_vector,
                     metrics=excluded.metrics,
                     converged=excluded.converged,
                     timestamp=excluded.timestamp""",
                state.to_row(),
            )

    def get_loop_history(
        self, loop_id: str
    ) -> list[LoopState]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM loop_states WHERE loop_id = ? ORDER BY iteration",
                (loop_id,),
            ).fetchall()
            return [LoopState.from_row(r) for r in rows]

    def get_latest_loop_state(
        self, loop_id: str
    ) -> Optional[LoopState]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM loop_states
                   WHERE loop_id = ?
                   ORDER BY iteration DESC LIMIT 1""",
                (loop_id,),
            ).fetchone()
            return LoopState.from_row(row) if row else None

    # -- Stats --

    def get_stats(self) -> dict:
        with self._connect() as conn:
            chapters = conn.execute(
                "SELECT COUNT(*) as c FROM chapters"
            ).fetchone()["c"]
            experts = conn.execute(
                "SELECT COUNT(*) as c FROM experts"
            ).fetchone()["c"]
            loops = conn.execute(
                "SELECT COUNT(DISTINCT loop_id) as c FROM loop_states"
            ).fetchone()["c"]
            total_words = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) as c FROM chapters"
            ).fetchone()["c"]
            return {
                "chapters_indexed": chapters,
                "experts_registered": experts,
                "active_loops": loops,
                "total_words_extracted": total_words,
                "db_path": self.db_path,
            }
