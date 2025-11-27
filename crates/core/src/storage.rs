use std::{path::Path, time::Duration};

use chrono::{DateTime, Utc};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// Alias for pooled SQLite connections.
pub type DbPool = Pool<SqliteConnectionManager>;

/// Lightweight data-access layer for prompts, analyses, and attachments.
#[derive(Clone)]
pub struct Storage {
    pool: DbPool,
}

impl Storage {
    /// Create (or open) the SQLite database at the provided path and ensure
    /// the schema defined in the README is available.
    pub fn new(db_path: impl AsRef<Path>) -> Result<Self, StorageError> {
        if let Some(parent) = db_path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        let manager = SqliteConnectionManager::file(db_path).with_init(|conn| {
            // Soften lock contention and tune for snappy reads/writes on local disk.
            conn.busy_timeout(Duration::from_secs(10))?;
            conn.execute_batch(
                "PRAGMA foreign_keys = ON;
                 PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 PRAGMA cache_size = -8000;         -- ~8MB page cache
                 PRAGMA mmap_size = 268435456;      -- 256MB mmap, best-effort
                 PRAGMA page_size = 4096;",
            )?;
            Ok(())
        });

        let pool = Pool::new(manager)?;
        let storage = Self { pool };
        storage.run_migrations()?;
        Ok(storage)
    }

    fn conn(&self) -> Result<PooledConnection<SqliteConnectionManager>, StorageError> {
        Ok(self.pool.get()?)
    }

    fn run_migrations(&self) -> Result<(), StorageError> {
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                language TEXT,
                model_hint TEXT,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prompts_updated_at ON prompts (datetime(updated_at));
            CREATE INDEX IF NOT EXISTS idx_prompts_created_at ON prompts (datetime(created_at));

            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                prompt_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL,
                classification TEXT NOT NULL,
                qwen_model TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_prompt_id_created_at
                ON analyses (prompt_id, datetime(created_at) DESC);

            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY,
                prompt_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                bytes BLOB NOT NULL,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_attachments_prompt_id ON attachments (prompt_id);
            "#,
        )?;
        Ok(())
    }

    /// Insert a new prompt entry and return the hydrated record.
    pub fn create_prompt(&self, data: NewPrompt) -> Result<Prompt, StorageError> {
        let conn = self.conn()?;
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        conn.execute(
            r#"
            INSERT INTO prompts (id, title, body, language, model_hint, metadata, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                id,
                data.title,
                data.body,
                data.language,
                data.model_hint,
                data.metadata.to_string(),
                now.to_rfc3339(),
                now.to_rfc3339()
            ],
        )?;
        self.get_prompt(&id)?
            .ok_or(StorageError::NotFound("prompt".into()))
    }

    /// Update an existing prompt in-place. Returns `None` if not found.
    pub fn update_prompt(&self, id: &str, changes: UpdatePrompt) -> Result<Option<Prompt>, StorageError> {
        let conn = self.conn()?;
        let existing = match self.get_prompt(id)? {
            Some(prompt) => prompt,
            None => return Ok(None),
        };

        let mut updated = existing;
        if let Some(title) = changes.title {
            updated.title = title;
        }
        if let Some(body) = changes.body {
            updated.body = body;
        }
        if let Some(language) = changes.language {
            updated.language = language;
        }
        if let Some(model_hint) = changes.model_hint {
            updated.model_hint = model_hint;
        }
        if let Some(metadata) = changes.metadata {
            updated.metadata = metadata;
        }

        updated.updated_at = Utc::now();

        conn.execute(
            r#"
            UPDATE prompts
            SET title = ?2,
                body = ?3,
                language = ?4,
                model_hint = ?5,
                metadata = ?6,
                updated_at = ?7
            WHERE id = ?1
            "#,
            params![
                id,
                updated.title,
                updated.body,
                updated.language,
                updated.model_hint,
                updated.metadata.to_string(),
                updated.updated_at.to_rfc3339()
            ],
        )?;

        self.get_prompt(id)
    }

    /// Fetch a single prompt.
    pub fn get_prompt(&self, id: &str) -> Result<Option<Prompt>, StorageError> {
        let conn = self.conn()?;
        let prompt = conn
            .query_row(
                "SELECT id, title, body, language, model_hint, metadata, created_at, updated_at FROM prompts WHERE id = ?1",
                params![id],
                |row| row_to_prompt(row),
            )
            .optional()?;
        Ok(prompt)
    }

    /// Find a prompt by exact body content (used for clipboard deduplication).
    pub fn find_prompt_by_body(&self, body: &str) -> Result<Option<Prompt>, StorageError> {
        let conn = self.conn()?;
        let prompt = conn
            .query_row(
                "SELECT id, title, body, language, model_hint, metadata, created_at, updated_at FROM prompts WHERE body = ?1 LIMIT 1",
                params![body],
                |row| row_to_prompt(row),
            )
            .optional()?;
        Ok(prompt)
    }

    /// List prompts ordered by most recently updated.
    pub fn list_prompts(&self) -> Result<Vec<Prompt>, StorageError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, title, body, language, model_hint, metadata, created_at, updated_at
             FROM prompts
             ORDER BY datetime(updated_at) DESC",
        )?;

        let rows = stmt
            .query_map([], |row| row_to_prompt(row))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Delete a prompt (analyses/attachments cascade).
    pub fn delete_prompt(&self, id: &str) -> Result<bool, StorageError> {
        let conn = self.conn()?;
        let affected = conn.execute("DELETE FROM prompts WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    /// Store a new AI analysis result.
    pub fn create_analysis(&self, input: NewAnalysis) -> Result<Analysis, StorageError> {
        let conn = self.conn()?;
        let id = Uuid::new_v4().to_string();
        let created_at = Utc::now();

        conn.execute(
            r#"
            INSERT INTO analyses (id, prompt_id, summary, tags, classification, qwen_model, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                id,
                input.prompt_id,
                input.summary,
                serde_json::to_string(&input.tags)?,
                input.classification.to_string(),
                input.qwen_model,
                created_at.to_rfc3339()
            ],
        )?;

        self.get_analysis(&id)?
            .ok_or(StorageError::NotFound("analysis".into()))
    }

    /// Get a specific analysis by ID.
    pub fn get_analysis(&self, id: &str) -> Result<Option<Analysis>, StorageError> {
        let conn = self.conn()?;
        let analysis = conn
            .query_row(
                "SELECT id, prompt_id, summary, tags, classification, qwen_model, created_at FROM analyses WHERE id = ?1",
                params![id],
                |row| row_to_analysis(row),
            )
            .optional()?;
        Ok(analysis)
    }

    /// List analyses for a prompt.
    pub fn list_analyses_for_prompt(&self, prompt_id: &str) -> Result<Vec<Analysis>, StorageError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, prompt_id, summary, tags, classification, qwen_model, created_at
             FROM analyses
             WHERE prompt_id = ?1
             ORDER BY datetime(created_at) DESC",
        )?;

        let items = stmt
            .query_map(params![prompt_id], |row| row_to_analysis(row))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(items)
    }

    /// Fetch the latest analysis for a prompt (if any).
    pub fn latest_analysis_for_prompt(&self, prompt_id: &str) -> Result<Option<Analysis>, StorageError> {
        let conn = self.conn()?;
        let analysis = conn
            .query_row(
                "SELECT id, prompt_id, summary, tags, classification, qwen_model, created_at
                 FROM analyses
                 WHERE prompt_id = ?1
                 ORDER BY datetime(created_at) DESC
                 LIMIT 1",
                params![prompt_id],
                |row| row_to_analysis(row),
            )
            .optional()?;
        Ok(analysis)
    }

    /// Store a binary attachment for a prompt.
    pub fn add_attachment(&self, payload: NewAttachment) -> Result<Attachment, StorageError> {
        let conn = self.conn()?;
        let id = Uuid::new_v4().to_string();
        conn.execute(
            r#"
            INSERT INTO attachments (id, prompt_id, filename, bytes)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![id, payload.prompt_id, payload.filename, payload.bytes],
        )?;
        self.get_attachment(&id)?
            .ok_or(StorageError::NotFound("attachment".into()))
    }

    /// Fetch attachment metadata + bytes.
    pub fn get_attachment(&self, id: &str) -> Result<Option<Attachment>, StorageError> {
        let conn = self.conn()?;
        let attachment = conn
            .query_row(
                "SELECT id, prompt_id, filename, bytes FROM attachments WHERE id = ?1",
                params![id],
                |row| {
                    Ok(Attachment {
                        id: row.get(0)?,
                        prompt_id: row.get(1)?,
                        filename: row.get(2)?,
                        bytes: row.get(3)?,
                    })
                },
            )
            .optional()?;
        Ok(attachment)
    }

    /// Remove attachment by id.
    pub fn delete_attachment(&self, id: &str) -> Result<bool, StorageError> {
        let conn = self.conn()?;
        let affected = conn.execute("DELETE FROM attachments WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }
}

fn row_to_prompt(row: &rusqlite::Row<'_>) -> rusqlite::Result<Prompt> {
    Ok(Prompt {
        id: row.get(0)?,
        title: row.get(1)?,
        body: row.get(2)?,
        language: row.get(3)?,
        model_hint: row.get(4)?,
        metadata: serde_json::from_str::<Value>(&row.get::<_, String>(5)?).unwrap_or(Value::Null),
        created_at: parse_datetime(&row.get::<_, String>(6)?)?,
        updated_at: parse_datetime(&row.get::<_, String>(7)?)?,
    })
}

fn row_to_analysis(row: &rusqlite::Row<'_>) -> rusqlite::Result<Analysis> {
    Ok(Analysis {
        id: row.get(0)?,
        prompt_id: row.get(1)?,
        summary: row.get(2)?,
        tags: serde_json::from_str::<Vec<String>>(&row.get::<_, String>(3)?).unwrap_or_default(),
        classification: serde_json::from_str::<Value>(&row.get::<_, String>(4)?).unwrap_or(Value::Null),
        qwen_model: row.get(5)?,
        created_at: parse_datetime(&row.get::<_, String>(6)?)?,
    })
}

fn parse_datetime(value: &str) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err)))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub title: String,
    pub body: String,
    pub language: Option<String>,
    pub model_hint: Option<String>,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct NewPrompt {
    pub title: String,
    pub body: String,
    pub language: Option<String>,
    pub model_hint: Option<String>,
    pub metadata: Value,
}

impl NewPrompt {
    pub fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            language: None,
            model_hint: None,
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UpdatePrompt {
    pub title: Option<String>,
    pub body: Option<String>,
    pub language: Option<Option<String>>,
    pub model_hint: Option<Option<String>>,
    pub metadata: Option<Value>,
}

impl Default for UpdatePrompt {
    fn default() -> Self {
        Self {
            title: None,
            body: None,
            language: None,
            model_hint: None,
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub id: String,
    pub prompt_id: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub classification: Value,
    pub qwen_model: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct NewAnalysis {
    pub prompt_id: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub classification: Value,
    pub qwen_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub id: String,
    pub prompt_id: String,
    pub filename: String,
    #[serde(skip_serializing)]
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct NewAttachment {
    pub prompt_id: String,
    pub filename: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("record not found: {0}")]
    NotFound(String),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Pool(#[from] r2d2::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
