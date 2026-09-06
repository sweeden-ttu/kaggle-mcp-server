use chrono::Utc;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    pub id: i64,
    pub chapter_number: i64,
    pub title: String,
    pub source_path: Option<String>,
    pub markdown_content: Option<String>,
    pub concepts: Vec<String>,
    pub extracted_at: Option<String>,
    pub word_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    pub id: i64,
    pub expert_name: String,
    pub slug: String,
    pub chapter_id: Option<i64>,
    pub capabilities: Vec<String>,
    pub skills: Vec<String>,
    pub strategy: String,
    pub formula: serde_json::Value,
    pub loop_config: serde_json::Value,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionStatus {
    pub id: i64,
    pub chapter_number: i64,
    pub status: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub error_message: Option<String>,
    pub pages_extracted: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceEntry {
    pub id: i64,
    pub competition: String,
    pub iteration: i64,
    pub state_vector: Vec<f64>,
    pub l2_norm: f64,
    pub converged: bool,
    pub timestamp: Option<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbStats {
    pub chapters_indexed: i64,
    pub experts_registered: i64,
    pub total_words_extracted: i64,
    pub completed_extractions: i64,
    pub db_path: String,
}

pub struct Database {
    conn: Connection,
    pub db_path: String,
}

fn default_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("SQLITE_DB_PATH") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".openclaw")
        .join("workspace")
        .join("mlsyseng")
        .join("mlsyseng.db")
}

impl Database {
    pub fn open(path: Option<&str>) -> SqlResult<Self> {
        let db_path = match path {
            Some(p) => PathBuf::from(p),
            None => default_db_path(),
        };

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let conn = Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        let db = Self {
            conn,
            db_path: db_path.to_string_lossy().into_owned(),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER UNIQUE,
                title TEXT NOT NULL,
                source_path TEXT,
                markdown_content TEXT,
                concepts TEXT,
                extracted_at TEXT,
                word_count INTEGER DEFAULT 0
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
                updated_at TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            );

            CREATE TABLE IF NOT EXISTS extraction_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                pages_extracted INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS convergence_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                state_vector TEXT,
                l2_norm REAL,
                converged INTEGER DEFAULT 0,
                timestamp TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(chapter_number);
            CREATE INDEX IF NOT EXISTS idx_experts_slug ON experts(slug);
            CREATE INDEX IF NOT EXISTS idx_convergence_competition
                ON convergence_state(competition, iteration);
            ",
        )
    }

    pub fn upsert_chapter(
        &self,
        chapter_number: i64,
        title: &str,
        source_path: &str,
        markdown_content: &str,
        concepts: &[String],
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let word_count = markdown_content.split_whitespace().count() as i64;
        let concepts_json = serde_json::to_string(concepts).unwrap_or_else(|_| "[]".into());

        self.conn.execute(
            "INSERT INTO chapters (chapter_number, title, source_path, markdown_content, concepts, extracted_at, word_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(chapter_number) DO UPDATE SET
                 title=excluded.title,
                 source_path=excluded.source_path,
                 markdown_content=excluded.markdown_content,
                 concepts=excluded.concepts,
                 extracted_at=excluded.extracted_at,
                 word_count=excluded.word_count",
            params![chapter_number, title, source_path, markdown_content, concepts_json, now, word_count],
        )?;

        let id: i64 = self.conn.query_row(
            "SELECT id FROM chapters WHERE chapter_number = ?1",
            params![chapter_number],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    pub fn get_chapter(&self, chapter_number: i64) -> SqlResult<Option<Chapter>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM chapters WHERE chapter_number = ?1")?;
        let mut rows = stmt.query(params![chapter_number])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_chapter(row)?)),
            None => Ok(None),
        }
    }

    pub fn get_all_chapters(&self) -> SqlResult<Vec<Chapter>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM chapters ORDER BY chapter_number")?;
        let rows = stmt.query_map([], |row| row_to_chapter(row))?;
        rows.collect()
    }

    pub fn upsert_expert(&self, expert: &Expert) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let caps = serde_json::to_string(&expert.capabilities).unwrap_or_else(|_| "[]".into());
        let skills = serde_json::to_string(&expert.skills).unwrap_or_else(|_| "[]".into());
        let formula = serde_json::to_string(&expert.formula).unwrap_or_else(|_| "{}".into());
        let lc = serde_json::to_string(&expert.loop_config).unwrap_or_else(|_| "{}".into());

        self.conn.execute(
            "INSERT INTO experts (expert_name, slug, chapter_id, capabilities, skills, strategy, formula, loop_config, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(slug) DO UPDATE SET
                 expert_name=excluded.expert_name,
                 chapter_id=excluded.chapter_id,
                 capabilities=excluded.capabilities,
                 skills=excluded.skills,
                 strategy=excluded.strategy,
                 formula=excluded.formula,
                 loop_config=excluded.loop_config,
                 updated_at=excluded.updated_at",
            params![expert.expert_name, expert.slug, expert.chapter_id, caps, skills, expert.strategy, formula, lc, now, now],
        )?;

        let id: i64 = self.conn.query_row(
            "SELECT id FROM experts WHERE slug = ?1",
            params![expert.slug],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    pub fn get_expert(&self, slug: &str) -> SqlResult<Option<Expert>> {
        let mut stmt = self.conn.prepare("SELECT * FROM experts WHERE slug = ?1")?;
        let mut rows = stmt.query(params![slug])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_expert(row)?)),
            None => Ok(None),
        }
    }

    pub fn get_all_experts(&self) -> SqlResult<Vec<Expert>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM experts ORDER BY expert_name")?;
        let rows = stmt.query_map([], |row| row_to_expert(row))?;
        rows.collect()
    }

    pub fn set_extraction_status(
        &self,
        chapter_number: i64,
        status: &str,
        error_message: Option<&str>,
        pages_extracted: i64,
    ) -> SqlResult<()> {
        let now = Utc::now().to_rfc3339();
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM extraction_status WHERE chapter_number = ?1",
                params![chapter_number],
                |r| r.get(0),
            )
            .ok();

        match existing {
            Some(id) => {
                let (time_col, time_val) = match status {
                    "extracting" => ("started_at", Some(now)),
                    "completed" | "failed" => ("completed_at", Some(now)),
                    _ => ("started_at", None),
                };
                let query = format!(
                    "UPDATE extraction_status SET status=?1, error_message=?2, pages_extracted=?3{} WHERE id=?4",
                    if let Some(ref tv) = time_val {
                        format!(", {}='{}'", time_col, tv)
                    } else {
                        String::new()
                    }
                );
                self.conn.execute(
                    &query,
                    params![status, error_message, pages_extracted, id],
                )?;
            }
            None => {
                let started = if status == "extracting" { Some(&now) } else { None };
                self.conn.execute(
                    "INSERT INTO extraction_status (chapter_number, status, started_at, error_message, pages_extracted)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![chapter_number, status, started, error_message, pages_extracted],
                )?;
            }
        }
        Ok(())
    }

    pub fn save_convergence_state(
        &self,
        competition: &str,
        iteration: i64,
        state_vector: &[f64],
        l2_norm: f64,
        converged: bool,
        metadata: &serde_json::Value,
    ) -> SqlResult<()> {
        let now = Utc::now().to_rfc3339();
        let sv = serde_json::to_string(state_vector).unwrap_or_else(|_| "[]".into());
        let md = serde_json::to_string(metadata).unwrap_or_else(|_| "{}".into());

        self.conn.execute(
            "INSERT INTO convergence_state (competition, iteration, state_vector, l2_norm, converged, timestamp, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![competition, iteration, sv, l2_norm, converged as i64, now, md],
        )?;
        Ok(())
    }

    pub fn get_convergence_history(&self, competition: &str) -> SqlResult<Vec<ConvergenceEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM convergence_state WHERE competition = ?1 ORDER BY iteration",
        )?;
        let rows = stmt.query_map(params![competition], |row| {
            let sv_str: String = row.get(3)?;
            let md_str: String = row.get(7)?;
            Ok(ConvergenceEntry {
                id: row.get(0)?,
                competition: row.get(1)?,
                iteration: row.get(2)?,
                state_vector: serde_json::from_str(&sv_str).unwrap_or_default(),
                l2_norm: row.get(4)?,
                converged: row.get::<_, i64>(5)? != 0,
                timestamp: row.get(6)?,
                metadata: serde_json::from_str(&md_str).unwrap_or(serde_json::Value::Null),
            })
        })?;
        rows.collect()
    }

    pub fn get_stats(&self) -> SqlResult<DbStats> {
        let chapters: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chapters", [], |r| r.get(0))?;
        let experts: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM experts", [], |r| r.get(0))?;
        let words: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(word_count), 0) FROM chapters",
            [],
            |r| r.get(0),
        )?;
        let completed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM extraction_status WHERE status = 'completed'",
            [],
            |r| r.get(0),
        )?;

        Ok(DbStats {
            chapters_indexed: chapters,
            experts_registered: experts,
            total_words_extracted: words,
            completed_extractions: completed,
            db_path: self.db_path.clone(),
        })
    }
}

fn row_to_chapter(row: &rusqlite::Row) -> SqlResult<Chapter> {
    let concepts_str: String = row.get(5)?;
    Ok(Chapter {
        id: row.get(0)?,
        chapter_number: row.get(1)?,
        title: row.get(2)?,
        source_path: row.get(3)?,
        markdown_content: row.get(4)?,
        concepts: serde_json::from_str(&concepts_str).unwrap_or_default(),
        extracted_at: row.get(6)?,
        word_count: row.get(7)?,
    })
}

fn row_to_expert(row: &rusqlite::Row) -> SqlResult<Expert> {
    let caps_str: String = row.get(4)?;
    let skills_str: String = row.get(5)?;
    let formula_str: String = row.get(7)?;
    let lc_str: String = row.get(8)?;
    Ok(Expert {
        id: row.get(0)?,
        expert_name: row.get(1)?,
        slug: row.get(2)?,
        chapter_id: row.get(3)?,
        capabilities: serde_json::from_str(&caps_str).unwrap_or_default(),
        skills: serde_json::from_str(&skills_str).unwrap_or_default(),
        strategy: row.get(6)?,
        formula: serde_json::from_str(&formula_str).unwrap_or(serde_json::Value::Null),
        loop_config: serde_json::from_str(&lc_str).unwrap_or(serde_json::Value::Null),
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
    })
}
