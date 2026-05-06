"""
SQLite database operations for the MLSysEng MoE system.

Stores extracted chapter content, expert definitions, extraction status,
and convergence loop state.
"""

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ChapterRecord:
    """A single extracted chapter from the ML Principles PDFs."""

    chapter_id: str
    chapter_name: str
    chapter_number: int
    source_path: str
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
    """A registered chapter expert with skills, strategy, and formula."""

    expert_id: str
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


@dataclass
class ConvergenceState:
    """Snapshot of the convergence loop state vector."""

    iteration: int
    state_vector: List[float]
    delta_norm: float
    converged: bool
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id    TEXT PRIMARY KEY,
    chapter_name  TEXT NOT NULL,
    chapter_number INTEGER NOT NULL,
    source_path   TEXT NOT NULL,
    content_md    TEXT NOT NULL,
    concepts      TEXT NOT NULL DEFAULT '[]',
    extracted_at  TEXT NOT NULL,
    word_count    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS experts (
    expert_id    TEXT PRIMARY KEY,
    expert_name  TEXT NOT NULL,
    slug         TEXT NOT NULL UNIQUE,
    chapter_id   TEXT NOT NULL,
    capabilities TEXT NOT NULL DEFAULT '[]',
    skills       TEXT NOT NULL DEFAULT '[]',
    strategy     TEXT NOT NULL DEFAULT '',
    formula      TEXT NOT NULL DEFAULT '{}',
    loop_config  TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
);

CREATE TABLE IF NOT EXISTS convergence_states (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    competition   TEXT NOT NULL,
    iteration     INTEGER NOT NULL,
    state_vector  TEXT NOT NULL,
    delta_norm    REAL NOT NULL,
    converged     INTEGER NOT NULL DEFAULT 0,
    timestamp     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_status (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MLSysEngDatabase:
    """SQLite-backed storage for the MLSysEng MoE system."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get(
                "SQLITE_DB_PATH",
                str(Path.home() / ".mlsyseng" / "mlsyseng.db"),
            )
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Chapter operations ---

    def upsert_chapter(self, chapter: ChapterRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO chapters (chapter_id, chapter_name, chapter_number,
                                  source_path, content_md, concepts,
                                  extracted_at, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_id) DO UPDATE SET
                content_md = excluded.content_md,
                concepts = excluded.concepts,
                extracted_at = excluded.extracted_at,
                word_count = excluded.word_count
            """,
            (
                chapter.chapter_id,
                chapter.chapter_name,
                chapter.chapter_number,
                chapter.source_path,
                chapter.content_md,
                json.dumps(chapter.concepts),
                chapter.extracted_at,
                chapter.word_count,
            ),
        )
        conn.commit()

    def get_chapter(self, chapter_id: str) -> Optional[ChapterRecord]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,))
            .fetchone()
        )
        if row is None:
            return None
        return self._row_to_chapter(row)

    def list_chapters(self) -> List[ChapterRecord]:
        rows = (
            self._get_conn()
            .execute("SELECT * FROM chapters ORDER BY chapter_number")
            .fetchall()
        )
        return [self._row_to_chapter(r) for r in rows]

    def _row_to_chapter(self, row: sqlite3.Row) -> ChapterRecord:
        return ChapterRecord(
            chapter_id=row["chapter_id"],
            chapter_name=row["chapter_name"],
            chapter_number=row["chapter_number"],
            source_path=row["source_path"],
            content_md=row["content_md"],
            concepts=json.loads(row["concepts"]),
            extracted_at=row["extracted_at"],
            word_count=row["word_count"],
        )

    # --- Expert operations ---

    def upsert_expert(self, expert: ExpertRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO experts (expert_id, expert_name, slug, chapter_id,
                                 capabilities, skills, strategy, formula,
                                 loop_config, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expert_id) DO UPDATE SET
                capabilities = excluded.capabilities,
                skills = excluded.skills,
                strategy = excluded.strategy,
                formula = excluded.formula,
                loop_config = excluded.loop_config
            """,
            (
                expert.expert_id,
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
        conn.commit()

    def get_expert(self, expert_id: str) -> Optional[ExpertRecord]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM experts WHERE expert_id = ?", (expert_id,))
            .fetchone()
        )
        if row is None:
            return None
        return self._row_to_expert(row)

    def get_expert_by_slug(self, slug: str) -> Optional[ExpertRecord]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM experts WHERE slug = ?", (slug,))
            .fetchone()
        )
        if row is None:
            return None
        return self._row_to_expert(row)

    def list_experts(self) -> List[ExpertRecord]:
        rows = (
            self._get_conn()
            .execute("SELECT * FROM experts ORDER BY expert_name")
            .fetchall()
        )
        return [self._row_to_expert(r) for r in rows]

    def _row_to_expert(self, row: sqlite3.Row) -> ExpertRecord:
        return ExpertRecord(
            expert_id=row["expert_id"],
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

    # --- Convergence state operations ---

    def save_convergence_state(
        self, competition: str, state: ConvergenceState
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO convergence_states
                (competition, iteration, state_vector, delta_norm, converged, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                competition,
                state.iteration,
                json.dumps(state.state_vector),
                state.delta_norm,
                int(state.converged),
                state.timestamp,
            ),
        )
        conn.commit()

    def get_convergence_history(self, competition: str) -> List[ConvergenceState]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM convergence_states WHERE competition = ? ORDER BY iteration",
                (competition,),
            )
            .fetchall()
        )
        return [
            ConvergenceState(
                iteration=r["iteration"],
                state_vector=json.loads(r["state_vector"]),
                delta_norm=r["delta_norm"],
                converged=bool(r["converged"]),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # --- Extraction status ---

    def set_status(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO extraction_status (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()

    def get_status(self, key: str) -> Optional[str]:
        row = (
            self._get_conn()
            .execute("SELECT value FROM extraction_status WHERE key = ?", (key,))
            .fetchone()
        )
        return row["value"] if row else None

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        chapter_count = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        expert_count = conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        total_words = (
            conn.execute("SELECT COALESCE(SUM(word_count), 0) FROM chapters").fetchone()[0]
        )
        return {
            "chapters_indexed": chapter_count,
            "experts_registered": expert_count,
            "total_words_extracted": total_words,
            "db_path": self.db_path,
        }
