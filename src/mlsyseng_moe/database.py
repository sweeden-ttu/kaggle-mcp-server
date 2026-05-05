"""SQLite database operations for MLSysEng MoE knowledge storage."""

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("SQLITE_DB_PATH", "~/.mlsyseng/mlsyseng.db")
)


@dataclass
class Chapter:
    chapter_id: str
    title: str
    folder_path: str
    content_md: str = ""
    concepts: str = "[]"
    extracted_at: float = 0.0
    status: str = "pending"

    @property
    def concept_list(self) -> List[str]:
        try:
            return json.loads(self.concepts)
        except (json.JSONDecodeError, TypeError):
            return []

    @concept_list.setter
    def concept_list(self, value: List[str]):
        self.concepts = json.dumps(value)


@dataclass
class Expert:
    expert_name: str
    slug: str
    chapter_id: str
    capabilities: str = "[]"
    skills: str = "[]"
    strategy: str = "Baseline → EDA → Feature Engineering → Model Selection → Submit"
    formula: str = "{}"
    loop_config: str = "{}"
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("capabilities", "skills", "formula", "loop_config"):
            try:
                d[k] = json.loads(d[k])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Expert":
        for k in ("capabilities", "skills", "formula", "loop_config"):
            if isinstance(d.get(k), (list, dict)):
                d[k] = json.dumps(d[k])
        return cls(**d)


@dataclass
class CompetitionEntry:
    entry_id: str
    competition: str
    experts_used: str = "[]"
    skills_used: str = "[]"
    state_history: str = "[]"
    converged: bool = False
    final_metric: float = 0.0
    created_at: float = 0.0


class MoEDatabase:
    """SQLite-backed storage for the MoE knowledge base."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS chapters (
                chapter_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                folder_path TEXT NOT NULL,
                content_md TEXT DEFAULT '',
                concepts TEXT DEFAULT '[]',
                extracted_at REAL DEFAULT 0,
                status TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS experts (
                expert_name TEXT PRIMARY KEY,
                slug TEXT UNIQUE NOT NULL,
                chapter_id TEXT NOT NULL,
                capabilities TEXT DEFAULT '[]',
                skills TEXT DEFAULT '[]',
                strategy TEXT DEFAULT '',
                formula TEXT DEFAULT '{}',
                loop_config TEXT DEFAULT '{}',
                created_at REAL DEFAULT 0,
                FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
            );

            CREATE TABLE IF NOT EXISTS competition_entries (
                entry_id TEXT PRIMARY KEY,
                competition TEXT NOT NULL,
                experts_used TEXT DEFAULT '[]',
                skills_used TEXT DEFAULT '[]',
                state_history TEXT DEFAULT '[]',
                converged INTEGER DEFAULT 0,
                final_metric REAL DEFAULT 0,
                created_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS embeddings_meta (
                chunk_id TEXT PRIMARY KEY,
                chapter_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at REAL DEFAULT 0,
                FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
            );
        """)
        self.conn.commit()

    # --- Chapter CRUD ---

    def upsert_chapter(self, chapter: Chapter):
        self.conn.execute(
            """INSERT INTO chapters (chapter_id, title, folder_path, content_md, concepts, extracted_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(chapter_id) DO UPDATE SET
                   title=excluded.title, folder_path=excluded.folder_path,
                   content_md=excluded.content_md, concepts=excluded.concepts,
                   extracted_at=excluded.extracted_at, status=excluded.status""",
            (chapter.chapter_id, chapter.title, chapter.folder_path,
             chapter.content_md, chapter.concepts, chapter.extracted_at, chapter.status),
        )
        self.conn.commit()

    def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        row = self.conn.execute(
            "SELECT * FROM chapters WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        if row:
            return Chapter(**dict(row))
        return None

    def list_chapters(self, status: Optional[str] = None) -> List[Chapter]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM chapters WHERE status = ? ORDER BY chapter_id", (status,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM chapters ORDER BY chapter_id"
            ).fetchall()
        return [Chapter(**dict(r)) for r in rows]

    # --- Expert CRUD ---

    def upsert_expert(self, expert: Expert):
        self.conn.execute(
            """INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(expert_name) DO UPDATE SET
                   slug=excluded.slug, chapter_id=excluded.chapter_id,
                   capabilities=excluded.capabilities, skills=excluded.skills,
                   strategy=excluded.strategy, formula=excluded.formula,
                   loop_config=excluded.loop_config, created_at=excluded.created_at""",
            (expert.expert_name, expert.slug, expert.chapter_id,
             expert.capabilities, expert.skills, expert.strategy,
             expert.formula, expert.loop_config, expert.created_at),
        )
        self.conn.commit()

    def get_expert(self, expert_name: str) -> Optional[Expert]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE expert_name = ?", (expert_name,)
        ).fetchone()
        if row:
            return Expert(**dict(row))
        return None

    def get_expert_by_slug(self, slug: str) -> Optional[Expert]:
        row = self.conn.execute(
            "SELECT * FROM experts WHERE slug = ?", (slug,)
        ).fetchone()
        if row:
            return Expert(**dict(row))
        return None

    def list_experts(self) -> List[Expert]:
        rows = self.conn.execute(
            "SELECT * FROM experts ORDER BY expert_name"
        ).fetchall()
        return [Expert(**dict(r)) for r in rows]

    # --- Competition Entry CRUD ---

    def save_entry(self, entry: CompetitionEntry):
        self.conn.execute(
            """INSERT INTO competition_entries
               (entry_id, competition, experts_used, skills_used, state_history, converged, final_metric, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(entry_id) DO UPDATE SET
                   experts_used=excluded.experts_used, skills_used=excluded.skills_used,
                   state_history=excluded.state_history, converged=excluded.converged,
                   final_metric=excluded.final_metric""",
            (entry.entry_id, entry.competition, entry.experts_used,
             entry.skills_used, entry.state_history, int(entry.converged),
             entry.final_metric, entry.created_at),
        )
        self.conn.commit()

    def get_entry(self, entry_id: str) -> Optional[CompetitionEntry]:
        row = self.conn.execute(
            "SELECT * FROM competition_entries WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["converged"] = bool(d["converged"])
            return CompetitionEntry(**d)
        return None

    def list_entries(self, competition: Optional[str] = None) -> List[CompetitionEntry]:
        if competition:
            rows = self.conn.execute(
                "SELECT * FROM competition_entries WHERE competition = ? ORDER BY created_at DESC",
                (competition,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM competition_entries ORDER BY created_at DESC"
            ).fetchall()
        entries = []
        for r in rows:
            d = dict(r)
            d["converged"] = bool(d["converged"])
            entries.append(CompetitionEntry(**d))
        return entries

    # --- Embeddings metadata ---

    def save_chunk_meta(self, chunk_id: str, chapter_id: str, chunk_text: str, model: str = "all-MiniLM-L6-v2"):
        self.conn.execute(
            """INSERT INTO embeddings_meta (chunk_id, chapter_id, chunk_text, embedding_model, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(chunk_id) DO UPDATE SET
                   chunk_text=excluded.chunk_text, embedding_model=excluded.embedding_model""",
            (chunk_id, chapter_id, chunk_text, model, time.time()),
        )
        self.conn.commit()

    def get_chunk_count(self, chapter_id: Optional[str] = None) -> int:
        if chapter_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM embeddings_meta WHERE chapter_id = ?", (chapter_id,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM embeddings_meta").fetchone()
        return row[0] if row else 0

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        chapters_total = self.conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
        chapters_done = self.conn.execute("SELECT COUNT(*) FROM chapters WHERE status = 'done'").fetchone()[0]
        experts_total = self.conn.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
        entries_total = self.conn.execute("SELECT COUNT(*) FROM competition_entries").fetchone()[0]
        chunks_total = self.get_chunk_count()
        return {
            "chapters_total": chapters_total,
            "chapters_extracted": chapters_done,
            "experts_registered": experts_total,
            "competition_entries": entries_total,
            "embedding_chunks": chunks_total,
        }

    def close(self):
        self.conn.close()
