"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


@dataclass
class ChapterRecord:
    chapter_id: str
    chapter_name: str
    source_path: str
    content_md: str
    concepts: list[str] = field(default_factory=list)
    extracted_at: float = 0.0
    word_count: int = 0

    def to_row(self) -> tuple:
        return (
            self.chapter_id,
            self.chapter_name,
            self.source_path,
            self.content_md,
            json.dumps(self.concepts),
            self.extracted_at or time.time(),
            self.word_count,
        )


@dataclass
class ExpertRecord:
    expert_name: str
    slug: str
    chapter_id: str
    capabilities: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    strategy: str = ""
    formula: dict = field(default_factory=dict)
    loop_config: dict = field(default_factory=dict)
    created_at: float = 0.0

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
            self.created_at or time.time(),
        )


class Database:
    """SQLite storage for chapters, experts, and extraction state."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
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
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id   TEXT PRIMARY KEY,
                    chapter_name TEXT NOT NULL,
                    source_path  TEXT NOT NULL,
                    content_md   TEXT NOT NULL,
                    concepts     TEXT DEFAULT '[]',
                    extracted_at REAL NOT NULL,
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
                    created_at   REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS extraction_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id  TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    message     TEXT DEFAULT '',
                    timestamp   REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition  TEXT NOT NULL,
                    iteration    INTEGER NOT NULL,
                    state_vector TEXT NOT NULL,
                    l2_norm      REAL,
                    converged    INTEGER DEFAULT 0,
                    timestamp    REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chapters_name
                    ON chapters(chapter_name);
                CREATE INDEX IF NOT EXISTS idx_experts_chapter
                    ON experts(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_state_competition
                    ON state_snapshots(competition, iteration);
                """
            )

    # ── Chapter operations ──

    def upsert_chapter(self, chapter: ChapterRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chapters
                    (chapter_id, chapter_name, source_path, content_md,
                     concepts, extracted_at, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_id) DO UPDATE SET
                    content_md = excluded.content_md,
                    concepts = excluded.concepts,
                    extracted_at = excluded.extracted_at,
                    word_count = excluded.word_count
                """,
                chapter.to_row(),
            )

    def get_chapter(self, chapter_id: str) -> Optional[ChapterRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
            ).fetchone()
        if not row:
            return None
        return ChapterRecord(
            chapter_id=row["chapter_id"],
            chapter_name=row["chapter_name"],
            source_path=row["source_path"],
            content_md=row["content_md"],
            concepts=json.loads(row["concepts"]),
            extracted_at=row["extracted_at"],
            word_count=row["word_count"],
        )

    def list_chapters(self) -> list[ChapterRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_id"
            ).fetchall()
        return [
            ChapterRecord(
                chapter_id=r["chapter_id"],
                chapter_name=r["chapter_name"],
                source_path=r["source_path"],
                content_md=r["content_md"],
                concepts=json.loads(r["concepts"]),
                extracted_at=r["extracted_at"],
                word_count=r["word_count"],
            )
            for r in rows
        ]

    def chapter_exists(self, chapter_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM chapters WHERE chapter_id = ?", (chapter_id,)
            ).fetchone()
        return row is not None

    # ── Expert operations ──

    def upsert_expert(self, expert: ExpertRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO experts
                    (expert_name, slug, chapter_id, capabilities, skills,
                     strategy, formula, loop_config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expert_name) DO UPDATE SET
                    capabilities = excluded.capabilities,
                    skills = excluded.skills,
                    strategy = excluded.strategy,
                    formula = excluded.formula,
                    loop_config = excluded.loop_config
                """,
                expert.to_row(),
            )

    def get_expert(self, expert_name: str) -> Optional[ExpertRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
            ).fetchone()
        if not row:
            return None
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

    def get_expert_by_slug(self, slug: str) -> Optional[ExpertRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experts WHERE slug = ?", (slug,)
            ).fetchone()
        if not row:
            return None
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

    def list_experts(self) -> list[ExpertRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experts ORDER BY expert_name"
            ).fetchall()
        return [
            ExpertRecord(
                expert_name=r["expert_name"],
                slug=r["slug"],
                chapter_id=r["chapter_id"],
                capabilities=json.loads(r["capabilities"]),
                skills=json.loads(r["skills"]),
                strategy=r["strategy"],
                formula=json.loads(r["formula"]),
                loop_config=json.loads(r["loop_config"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Extraction log ──

    def log_extraction(self, chapter_id: str, status: str, message: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO extraction_log (chapter_id, status, message, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (chapter_id, status, message, time.time()),
            )

    def get_extraction_status(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chapter_id, status, message, timestamp
                FROM extraction_log
                ORDER BY timestamp DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ── State snapshots ──

    def save_state_snapshot(
        self,
        competition: str,
        iteration: int,
        state_vector: list[float],
        l2_norm: float,
        converged: bool,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO state_snapshots
                    (competition, iteration, state_vector, l2_norm, converged, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition,
                    iteration,
                    json.dumps(state_vector),
                    l2_norm,
                    int(converged),
                    time.time(),
                ),
            )

    def get_state_history(self, competition: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT iteration, state_vector, l2_norm, converged, timestamp
                FROM state_snapshots
                WHERE competition = ?
                ORDER BY iteration
                """,
                (competition,),
            ).fetchall()
        return [
            {
                "iteration": r["iteration"],
                "state_vector": json.loads(r["state_vector"]),
                "l2_norm": r["l2_norm"],
                "converged": bool(r["converged"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    # ── Stats ──

    def get_stats(self) -> dict:
        with self._conn() as conn:
            chapter_count = conn.execute(
                "SELECT COUNT(*) FROM chapters"
            ).fetchone()[0]
            expert_count = conn.execute(
                "SELECT COUNT(*) FROM experts"
            ).fetchone()[0]
            total_words = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) FROM chapters"
            ).fetchone()[0]
            extraction_count = conn.execute(
                "SELECT COUNT(*) FROM extraction_log"
            ).fetchone()[0]
        return {
            "chapters": chapter_count,
            "experts": expert_count,
            "total_words": total_words,
            "extractions": extraction_count,
            "db_path": str(self.db_path),
        }
