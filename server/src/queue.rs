//! SQLite-backed durable submission queue.
//!
//! `SubmissionStore` is the single source of truth for job state: what's
//! queued, what's running, what finished (and how). It has no dependency
//! on `backend.rs` or any HTTP types — Task 3 (worker pool) and Task 4
//! (HTTP layer) both build on the API here.
//!
//! # Concurrency
//!
//! The store wraps a single `rusqlite::Connection` in a `Mutex`. SQLite
//! only allows one writer at a time regardless of how many connections you
//! open, and this service's write volume (job submissions, periodic
//! heartbeats, worker claims) is far below what a single serialized
//! connection can handle, so a connection pool would add complexity for no
//! real throughput gain. Every operation below takes the lock for the
//! duration of one (possibly transactional) unit of work and releases it
//! before returning.
//!
//! # `input_cleaned` bookkeeping
//!
//! The schema in the task brief doesn't list an `input_cleaned` column,
//! but [`reap`](SubmissionStore::reap) needs one: rows that reach a
//! terminal state (`ready`, `failed`, `cancelled`) stay in the table
//! (only expired `ready` rows are ever deleted), so without a persistent
//! "have we already told the caller to delete this input file" flag, a
//! periodic reaper would keep re-reporting the same `input_path` on every
//! sweep. `input_cleaned` (`INTEGER NOT NULL DEFAULT 0`, treated as a
//! bool) is set to `1` the first time `reap` reports a terminal row's
//! `input_path` to the caller for deletion.

#![allow(dead_code)]

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};
use rusqlite::{Connection, OptionalExtension, Row, params};
use thiserror::Error;

/// SHA-256 of a submission's 64-hex token. The token itself is never
/// stored.
pub type TokenHash = [u8; 32];

/// How long a `ready` result stays available before `reap` deletes it.
const RESULT_TTL_SECS: i64 = 3600;
/// How long a `queued` job may wait before `reap` fails it as a timeout.
const QUEUE_TIMEOUT_SECS: i64 = 1800;
/// Attempts (post-increment) beyond which `restart_sweep` gives up on a
/// `running` row instead of requeuing it again.
const MAX_RESTART_ATTEMPTS: i64 = 2;

/// Schema DDL shared by every way of constructing a `SubmissionStore`
/// (on-disk `open()` and, in tests, `open_in_memory()`), so the two paths
/// can never drift out of sync with each other. `IF NOT EXISTS`/`OR
/// IGNORE` throughout so it's also safe to run against an existing
/// on-disk database on every startup.
const SCHEMA_SQL: &str = "
    CREATE TABLE IF NOT EXISTS submissions (
        token_sha256      BLOB PRIMARY KEY,
        queue_seq         INTEGER NOT NULL,
        state             TEXT NOT NULL,
        record_count      INTEGER NOT NULL,
        processed_records INTEGER NOT NULL DEFAULT 0,
        match_count       INTEGER,
        error             TEXT,
        input_path        TEXT NOT NULL,
        result_path       TEXT,
        attempts          INTEGER NOT NULL DEFAULT 0,
        created_at        INTEGER NOT NULL,
        started_at        INTEGER,
        finished_at       INTEGER,
        result_expires_at INTEGER,
        input_cleaned     INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_submissions_queue_seq
        ON submissions(queue_seq);
    CREATE INDEX IF NOT EXISTS idx_submissions_state
        ON submissions(state);
    CREATE TABLE IF NOT EXISTS queue_counter (
        id INTEGER PRIMARY KEY CHECK (id = 0),
        next_seq INTEGER NOT NULL
    );
    INSERT OR IGNORE INTO queue_counter (id, next_seq) VALUES (0, 1);
";

#[derive(Debug, Error)]
pub enum QueueError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, QueueError>;

/// Lifecycle state of a submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionState {
    Queued,
    Running,
    Ready,
    Failed,
    Cancelled,
}

impl SubmissionState {
    fn as_str(self) -> &'static str {
        match self {
            SubmissionState::Queued => "queued",
            SubmissionState::Running => "running",
            SubmissionState::Ready => "ready",
            SubmissionState::Failed => "failed",
            SubmissionState::Cancelled => "cancelled",
        }
    }

    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "queued" => Ok(SubmissionState::Queued),
            "running" => Ok(SubmissionState::Running),
            "ready" => Ok(SubmissionState::Ready),
            "failed" => Ok(SubmissionState::Failed),
            "cancelled" => Ok(SubmissionState::Cancelled),
            other => Err(format!("unknown submission state {other:?}")),
        }
    }
}

impl ToSql for SubmissionState {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.as_str()))
    }
}

impl FromSql for SubmissionState {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let text = value.as_str()?;
        SubmissionState::parse(text).map_err(|e| FromSqlError::Other(e.into()))
    }
}

/// A helper newtype so `TokenHash` ([u8; 32]) round-trips through
/// `rusqlite` as a fixed-size BLOB instead of a generic `Vec<u8>`.
struct TokenHashSql(TokenHash);

impl FromSql for TokenHashSql {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let blob = value.as_blob()?;
        let array: TokenHash = blob
            .try_into()
            .map_err(|_| FromSqlError::Other("token_sha256 blob is not 32 bytes".into()))?;
        Ok(TokenHashSql(array))
    }
}

/// Full row for the status/result/cancel handlers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionRow {
    pub token_sha256: TokenHash,
    pub queue_seq: i64,
    pub state: SubmissionState,
    pub record_count: i64,
    pub processed_records: i64,
    pub match_count: Option<i64>,
    pub error: Option<String>,
    pub input_path: String,
    pub result_path: Option<String>,
    pub attempts: i64,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
    pub result_expires_at: Option<i64>,
    pub input_cleaned: bool,
}

fn row_to_submission(row: &Row<'_>) -> rusqlite::Result<SubmissionRow> {
    let token = row.get::<_, TokenHashSql>("token_sha256")?;
    Ok(SubmissionRow {
        token_sha256: token.0,
        queue_seq: row.get("queue_seq")?,
        state: row.get("state")?,
        record_count: row.get("record_count")?,
        processed_records: row.get("processed_records")?,
        match_count: row.get("match_count")?,
        error: row.get("error")?,
        input_path: row.get("input_path")?,
        result_path: row.get("result_path")?,
        attempts: row.get("attempts")?,
        created_at: row.get("created_at")?,
        started_at: row.get("started_at")?,
        finished_at: row.get("finished_at")?,
        result_expires_at: row.get("result_expires_at")?,
        input_cleaned: row.get::<_, i64>("input_cleaned")? != 0,
    })
}

/// Result of [`SubmissionStore::insert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome {
    Inserted {
        queue_seq: i64,
    },
    /// `slots + max_queued` submissions are already `queued`/`running`.
    /// Caller maps this to an HTTP 503.
    QueueFull,
}

/// What a worker needs to run a lookup, returned by
/// [`SubmissionStore::claim_next_queued`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedJob {
    pub token_sha256: TokenHash,
    pub input_path: String,
    pub record_count: i64,
}

/// Filesystem cleanup work produced by [`SubmissionStore::reap`].
/// `queue.rs` never touches the filesystem itself; the caller deletes
/// these paths.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReapReport {
    /// `result_path`s of expired `ready` rows (whose DB rows were
    /// deleted).
    pub deleted_result_paths: Vec<String>,
    /// `input_path`s of rows now in a terminal state whose input hasn't
    /// been reported for cleanup before.
    pub cleanup_input_paths: Vec<String>,
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_secs() as i64
}

/// SQLite-backed durable submission queue. See the module docs for the
/// concurrency model.
pub struct SubmissionStore {
    conn: Mutex<Connection>,
}

impl SubmissionStore {
    /// Shared construction path for every way of getting a
    /// `SubmissionStore`: enables WAL mode and runs [`SCHEMA_SQL`] against
    /// `conn`. Both `open()` and `open_in_memory()` funnel through this so
    /// the schema can never drift between the on-disk and in-memory (test)
    /// paths.
    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(SubmissionStore {
            conn: Mutex::new(conn),
        })
    }

    /// Open (creating if necessary) the queue database at `path`, enable
    /// WAL mode, and ensure the schema exists.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())?;
        Self::init(conn)
    }

    /// Open an in-memory database. Only used by this module's own tests
    /// (`tempfile::tempdir()` is used for the "on disk" behavior instead
    /// where the test cares about the file existing). WAL mode is a no-op
    /// on `:memory:` databases but harmless to request.
    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    /// Insert a new `queued` submission, unless the number of
    /// `queued`/`running` rows has already reached `slots + max_queued`.
    pub fn insert(
        &self,
        token_sha256: TokenHash,
        record_count: u64,
        input_path: &str,
        max_queued: usize,
        slots: usize,
    ) -> Result<InsertOutcome> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction()?;

        let in_flight: i64 = tx.query_row(
            "SELECT COUNT(*) FROM submissions WHERE state IN ('queued', 'running')",
            [],
            |r| r.get(0),
        )?;
        if in_flight as usize >= slots + max_queued {
            tx.commit()?;
            return Ok(InsertOutcome::QueueFull);
        }

        let next_seq: i64 =
            tx.query_row("SELECT next_seq FROM queue_counter WHERE id = 0", [], |r| {
                r.get(0)
            })?;
        tx.execute(
            "UPDATE queue_counter SET next_seq = next_seq + 1 WHERE id = 0",
            [],
        )?;

        tx.execute(
            "INSERT INTO submissions
                (token_sha256, queue_seq, state, record_count, processed_records,
                 input_path, attempts, created_at)
             VALUES (?1, ?2, 'queued', ?3, 0, ?4, 0, ?5)",
            params![
                token_sha256.as_slice(),
                next_seq,
                record_count as i64,
                input_path,
                now_unix(),
            ],
        )?;
        tx.commit()?;
        Ok(InsertOutcome::Inserted {
            queue_seq: next_seq,
        })
    }

    /// Atomically claim the lowest-`queue_seq` `queued` row, transitioning
    /// it to `running`.
    pub fn claim_next_queued(&self) -> Result<Option<ClaimedJob>> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction()?;

        let candidate = tx
            .query_row(
                "SELECT token_sha256, input_path, record_count FROM submissions
                 WHERE state = 'queued' ORDER BY queue_seq ASC LIMIT 1",
                [],
                |row| {
                    let token = row.get::<_, TokenHashSql>(0)?;
                    let input_path: String = row.get(1)?;
                    let record_count: i64 = row.get(2)?;
                    Ok((token.0, input_path, record_count))
                },
            )
            .optional()?;

        let Some((token_sha256, input_path, record_count)) = candidate else {
            tx.commit()?;
            return Ok(None);
        };

        tx.execute(
            "UPDATE submissions SET state = 'running', started_at = ?2
             WHERE token_sha256 = ?1 AND state = 'queued'",
            params![token_sha256.as_slice(), now_unix()],
        )?;
        tx.commit()?;

        Ok(Some(ClaimedJob {
            token_sha256,
            input_path,
            record_count,
        }))
    }

    /// Cheap progress update, called by the worker heartbeat (~every 2s).
    pub fn checkpoint_progress(
        &self,
        token_sha256: TokenHash,
        processed_records: u64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "UPDATE submissions SET processed_records = ?2 WHERE token_sha256 = ?1",
            params![token_sha256.as_slice(), processed_records as i64],
        )?;
        Ok(())
    }

    /// Transition a `running` row to `ready`. Guarded to `state =
    /// 'running'` so a `mark_ready` racing after the row already left
    /// `running` (e.g. a concurrent `mark_cancelled`) is a no-op instead
    /// of clobbering whatever terminal state won the race. Returns `true`
    /// if a row was actually transitioned, `false` if the token is
    /// unknown or the row wasn't `running` anymore.
    pub fn mark_ready(
        &self,
        token_sha256: TokenHash,
        match_count: u64,
        result_path: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_unix();
        let affected = conn.execute(
            "UPDATE submissions
             SET state = 'ready', match_count = ?2, result_path = ?3,
                 finished_at = ?4, result_expires_at = ?5
             WHERE token_sha256 = ?1 AND state = 'running'",
            params![
                token_sha256.as_slice(),
                match_count as i64,
                result_path,
                now,
                now + RESULT_TTL_SECS,
            ],
        )?;
        Ok(affected > 0)
    }

    /// Transition a `queued`/`running` row to `failed`. Guarded so a
    /// `mark_failed` racing after the row already reached a terminal
    /// state (e.g. `mark_ready` already won, or a stale-queue `reap`
    /// already failed it) is a no-op. Returns `true` if a row was
    /// actually transitioned.
    pub fn mark_failed(&self, token_sha256: TokenHash, error: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let affected = conn.execute(
            "UPDATE submissions SET state = 'failed', error = ?2, finished_at = ?3
             WHERE token_sha256 = ?1 AND state IN ('queued', 'running')",
            params![token_sha256.as_slice(), error, now_unix()],
        )?;
        Ok(affected > 0)
    }

    /// Transition a `queued`/`running` row to `cancelled`. Guarded so a
    /// cancel landing after the worker already finished (a realistic
    /// race: the row went `ready` between the handler's `get` and its
    /// `mark_cancelled` call) is a no-op rather than overwriting a
    /// `ready` row's `result_path` with `cancelled` and orphaning the
    /// result file on disk. Returns `true` if a row was actually
    /// transitioned, `false` if the token is unknown or the row was
    /// already terminal (so the caller can distinguish "cancelled" from
    /// "too late" without a separate TOCTOU `get`).
    pub fn mark_cancelled(&self, token_sha256: TokenHash) -> Result<bool> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let affected = conn.execute(
            "UPDATE submissions SET state = 'cancelled', finished_at = ?2
             WHERE token_sha256 = ?1 AND state IN ('queued', 'running')",
            params![token_sha256.as_slice(), now_unix()],
        )?;
        Ok(affected > 0)
    }

    /// Full row for the status/result/cancel handlers.
    pub fn get(&self, token_sha256: TokenHash) -> Result<Option<SubmissionRow>> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let row = conn
            .query_row(
                "SELECT * FROM submissions WHERE token_sha256 = ?1",
                params![token_sha256.as_slice()],
                row_to_submission,
            )
            .optional()?;
        Ok(row)
    }

    /// Count of `queued` rows with a lower `queue_seq` than this row's.
    /// Only meaningful when the row itself is `queued`; callers decide
    /// when to use it (returns `Some` regardless of the row's current
    /// state, as long as the row exists).
    pub fn queue_position(&self, token_sha256: TokenHash) -> Result<Option<u64>> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let seq: Option<i64> = conn
            .query_row(
                "SELECT queue_seq FROM submissions WHERE token_sha256 = ?1",
                params![token_sha256.as_slice()],
                |r| r.get(0),
            )
            .optional()?;
        let Some(seq) = seq else {
            return Ok(None);
        };
        let position: i64 = conn.query_row(
            "SELECT COUNT(*) FROM submissions WHERE state = 'queued' AND queue_seq < ?1",
            params![seq],
            |r| r.get(0),
        )?;
        Ok(Some(position as u64))
    }

    /// Call once at startup. `running` rows with `attempts <= 2` go back
    /// to `queued` (progress reset, `attempts` incremented); rows beyond
    /// that are given up on and marked `failed`. Returns the rows that
    /// were requeued (post-transition state) so `main.rs` can re-enqueue
    /// their tokens with the worker pool's dispatch mechanism if needed.
    pub fn restart_sweep(&self) -> Result<Vec<SubmissionRow>> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction()?;
        let now = now_unix();

        let requeue_tokens: Vec<TokenHash> = {
            let mut stmt = tx.prepare(
                "SELECT token_sha256 FROM submissions
                 WHERE state = 'running' AND attempts <= ?1",
            )?;
            let rows = stmt.query_map(params![MAX_RESTART_ATTEMPTS], |row| {
                Ok(row.get::<_, TokenHashSql>(0)?.0)
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        tx.execute(
            "UPDATE submissions
             SET state = 'queued', processed_records = 0, attempts = attempts + 1,
                 started_at = NULL
             WHERE state = 'running' AND attempts <= ?1",
            params![MAX_RESTART_ATTEMPTS],
        )?;

        tx.execute(
            "UPDATE submissions
             SET state = 'failed', error = 'server restarted during processing',
                 finished_at = ?2
             WHERE state = 'running' AND attempts > ?1",
            params![MAX_RESTART_ATTEMPTS, now],
        )?;

        let mut requeued = Vec::with_capacity(requeue_tokens.len());
        for token in requeue_tokens {
            let row = tx.query_row(
                "SELECT * FROM submissions WHERE token_sha256 = ?1",
                params![token.as_slice()],
                row_to_submission,
            )?;
            requeued.push(row);
        }

        tx.commit()?;
        Ok(requeued)
    }

    /// Periodic maintenance: expire stale `ready` results, time out
    /// `queued` jobs that never got a worker, and report `input_path`s of
    /// newly-and-previously terminal rows that haven't been cleaned up.
    /// `queue.rs` never touches the filesystem itself; the caller acts on
    /// the returned paths.
    pub fn reap(&self, now: i64) -> Result<ReapReport> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction()?;

        // (b) Stale `queued` rows time out first, so they're picked up by
        // the terminal-input-cleanup pass below in the same sweep.
        tx.execute(
            "UPDATE submissions
             SET state = 'failed',
                 error = 'queue timeout: no worker available within 30 minutes',
                 finished_at = ?2
             WHERE state = 'queued' AND created_at < ?1",
            params![now - QUEUE_TIMEOUT_SECS, now],
        )?;

        // (c) Any terminal row whose input hasn't been reported yet.
        let cleanup_input_paths: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT input_path FROM submissions
                 WHERE state IN ('ready', 'failed', 'cancelled') AND input_cleaned = 0",
            )?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        tx.execute(
            "UPDATE submissions SET input_cleaned = 1
             WHERE state IN ('ready', 'failed', 'cancelled') AND input_cleaned = 0",
            [],
        )?;

        // (a) Expired `ready` rows: delete outright.
        let deleted_result_paths: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT result_path FROM submissions
                 WHERE state = 'ready' AND result_expires_at < ?1",
            )?;
            let rows = stmt.query_map(params![now], |row| row.get::<_, Option<String>>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect()
        };
        tx.execute(
            "DELETE FROM submissions WHERE state = 'ready' AND result_expires_at < ?1",
            params![now],
        )?;

        tx.commit()?;
        Ok(ReapReport {
            deleted_result_paths,
            cleanup_input_paths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(byte: u8) -> TokenHash {
        [byte; 32]
    }

    #[test]
    fn open_creates_schema_on_disk_with_wal_mode() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("queue.sqlite3");
        let store = SubmissionStore::open(&db_path).expect("open");
        let outcome = store
            .insert(token(1), 10, "/tmp/in1", 8, 2)
            .expect("insert");
        assert_eq!(outcome, InsertOutcome::Inserted { queue_seq: 1 });
        assert!(db_path.exists());
    }

    #[test]
    fn insert_respects_depth_cap_and_reports_queue_full() {
        let store = SubmissionStore::open_in_memory().expect("open");
        // slots=1, max_queued=1 => cap of 2 in-flight rows.
        assert_eq!(
            store.insert(token(1), 10, "/in1", 1, 1).unwrap(),
            InsertOutcome::Inserted { queue_seq: 1 }
        );
        assert_eq!(
            store.insert(token(2), 10, "/in2", 1, 1).unwrap(),
            InsertOutcome::Inserted { queue_seq: 2 }
        );
        assert_eq!(
            store.insert(token(3), 10, "/in3", 1, 1).unwrap(),
            InsertOutcome::QueueFull
        );
        // The rejected submission must not have been inserted.
        assert!(store.get(token(3)).unwrap().is_none());
    }

    #[test]
    fn insert_counts_running_rows_toward_the_cap_too() {
        let store = SubmissionStore::open_in_memory().expect("open");
        assert_eq!(
            store.insert(token(1), 10, "/in1", 0, 1).unwrap(),
            InsertOutcome::Inserted { queue_seq: 1 }
        );
        store.claim_next_queued().unwrap(); // now running
        assert_eq!(
            store.insert(token(2), 10, "/in2", 0, 1).unwrap(),
            InsertOutcome::QueueFull
        );
    }

    #[test]
    fn claim_next_queued_returns_rows_in_queue_seq_order_and_skips_non_queued() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 10, "/in1", 8, 8).unwrap();
        store.insert(token(2), 20, "/in2", 8, 8).unwrap();
        store.insert(token(3), 30, "/in3", 8, 8).unwrap();

        let first = store.claim_next_queued().unwrap().expect("job");
        assert_eq!(first.token_sha256, token(1));
        assert_eq!(first.input_path, "/in1");
        assert_eq!(first.record_count, 10);

        // token(1) is now running, so the next claim skips it.
        let second = store.claim_next_queued().unwrap().expect("job");
        assert_eq!(second.token_sha256, token(2));

        let third = store.claim_next_queued().unwrap().expect("job");
        assert_eq!(third.token_sha256, token(3));

        assert!(store.claim_next_queued().unwrap().is_none());

        let row1 = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row1.state, SubmissionState::Running);
        assert!(row1.started_at.is_some());
    }

    #[test]
    fn checkpoint_progress_then_get_round_trips() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 100, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        store.checkpoint_progress(token(1), 42).unwrap();
        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.processed_records, 42);
    }

    #[test]
    fn mark_ready_sets_terminal_fields_and_expiry() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 100, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        store.mark_ready(token(1), 7, "/out1").unwrap();
        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.state, SubmissionState::Ready);
        assert_eq!(row.match_count, Some(7));
        assert_eq!(row.result_path.as_deref(), Some("/out1"));
        assert!(row.finished_at.is_some());
        assert_eq!(
            row.result_expires_at,
            Some(row.finished_at.unwrap() + RESULT_TTL_SECS)
        );
    }

    #[test]
    fn mark_failed_sets_terminal_fields() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 100, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        store.mark_failed(token(1), "boom").unwrap();
        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.state, SubmissionState::Failed);
        assert_eq!(row.error.as_deref(), Some("boom"));
        assert!(row.finished_at.is_some());
        assert!(row.result_expires_at.is_none());
    }

    #[test]
    fn mark_cancelled_sets_terminal_fields() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 100, "/in1", 8, 8).unwrap();
        assert!(store.mark_cancelled(token(1)).unwrap());
        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.state, SubmissionState::Cancelled);
        assert!(row.finished_at.is_some());
    }

    #[test]
    fn mark_ready_mark_failed_mark_cancelled_report_whether_they_actually_transitioned() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        assert!(store.mark_ready(token(1), 1, "/out1").unwrap());

        // Row is already `ready` (terminal) now: further mark_* calls on
        // it must no-op and report false, not silently "succeed".
        assert!(!store.mark_ready(token(1), 2, "/out2").unwrap());
        assert!(!store.mark_failed(token(1), "too late").unwrap());
        assert!(!store.mark_cancelled(token(1)).unwrap());

        // An unknown token also reports false rather than a silent
        // no-op success.
        assert!(!store.mark_ready(token(99), 1, "/out99").unwrap());
        assert!(!store.mark_failed(token(99), "boom").unwrap());
        assert!(!store.mark_cancelled(token(99)).unwrap());
    }

    #[test]
    fn mark_cancelled_after_mark_ready_does_not_clobber_the_ready_row() {
        // Regression test for the race where a cancel request lands after
        // the worker already finished (between the handler's `get` and
        // its `mark_cancelled` call): the row must stay `ready` with its
        // `result_path` intact, not flip to `cancelled` and orphan the
        // result file on disk.
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        assert!(store.mark_ready(token(1), 3, "/out1").unwrap());

        let cancelled = store.mark_cancelled(token(1)).unwrap();
        assert!(!cancelled);

        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.state, SubmissionState::Ready);
        assert_eq!(row.match_count, Some(3));
        assert_eq!(row.result_path.as_deref(), Some("/out1"));
    }

    #[test]
    fn queue_position_is_correct_for_a_three_row_queue() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        store.insert(token(2), 1, "/in2", 8, 8).unwrap();
        store.insert(token(3), 1, "/in3", 8, 8).unwrap();

        assert_eq!(store.queue_position(token(1)).unwrap(), Some(0));
        assert_eq!(store.queue_position(token(2)).unwrap(), Some(1));
        assert_eq!(store.queue_position(token(3)).unwrap(), Some(2));
        assert_eq!(store.queue_position(token(99)).unwrap(), None);

        // Claiming token(1) removes it from 'queued', shifting the rest.
        store.claim_next_queued().unwrap();
        assert_eq!(store.queue_position(token(2)).unwrap(), Some(0));
        assert_eq!(store.queue_position(token(3)).unwrap(), Some(1));
    }

    #[test]
    fn restart_sweep_requeues_running_row_once_then_fails_it_on_second_sweep() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        store.checkpoint_progress(token(1), 5).unwrap();

        // Sweep 1: attempts 0 -> 1, requeued.
        let requeued = store.restart_sweep().unwrap();
        assert_eq!(requeued.len(), 1);
        assert_eq!(requeued[0].token_sha256, token(1));
        assert_eq!(requeued[0].state, SubmissionState::Queued);
        assert_eq!(requeued[0].attempts, 1);
        assert_eq!(requeued[0].processed_records, 0);
        assert!(requeued[0].started_at.is_none());

        store.claim_next_queued().unwrap();
        // Sweep 2: attempts 1 -> 2, requeued.
        let requeued = store.restart_sweep().unwrap();
        assert_eq!(requeued.len(), 1);
        assert_eq!(requeued[0].attempts, 2);

        store.claim_next_queued().unwrap();
        // Sweep 3: attempts 2 -> still <= 2, so requeued again to 3.
        let requeued = store.restart_sweep().unwrap();
        assert_eq!(requeued.len(), 1);
        assert_eq!(requeued[0].attempts, 3);

        store.claim_next_queued().unwrap();
        // Sweep 4: attempts is now 3 (> 2), so this run fails it instead.
        let requeued = store.restart_sweep().unwrap();
        assert!(requeued.is_empty());
        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.state, SubmissionState::Failed);
        assert_eq!(
            row.error.as_deref(),
            Some("server restarted during processing")
        );
        assert!(row.finished_at.is_some());
    }

    #[test]
    fn restart_sweep_leaves_queued_rows_alone() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        let requeued = store.restart_sweep().unwrap();
        assert!(requeued.is_empty());
        let row = store.get(token(1)).unwrap().expect("row");
        assert_eq!(row.state, SubmissionState::Queued);
        assert_eq!(row.attempts, 0);
    }

    #[test]
    fn reap_deletes_expired_ready_row_and_fails_stale_queued_row_leaving_fresh_queued_alone() {
        let store = SubmissionStore::open_in_memory().expect("open");

        // A ready row whose result already expired.
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        store.claim_next_queued().unwrap();
        store.mark_ready(token(1), 1, "/out1").unwrap();
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "UPDATE submissions SET result_expires_at = ?2 WHERE token_sha256 = ?1",
                params![token(1).as_slice(), now_unix() - 1],
            )
            .unwrap();
        }

        // A queued row that's been sitting for a long time.
        store.insert(token(2), 1, "/in2", 8, 8).unwrap();
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "UPDATE submissions SET created_at = ?2 WHERE token_sha256 = ?1",
                params![token(2).as_slice(), now_unix() - QUEUE_TIMEOUT_SECS - 1],
            )
            .unwrap();
        }

        // A fresh queued row that should be left alone.
        store.insert(token(3), 1, "/in3", 8, 8).unwrap();

        let now = now_unix();
        let report = store.reap(now).unwrap();

        assert_eq!(report.deleted_result_paths, vec!["/out1".to_string()]);
        assert!(store.get(token(1)).unwrap().is_none());

        let row2 = store.get(token(2)).unwrap().expect("row2");
        assert_eq!(row2.state, SubmissionState::Failed);
        assert_eq!(
            row2.error.as_deref(),
            Some("queue timeout: no worker available within 30 minutes")
        );

        let row3 = store.get(token(3)).unwrap().expect("row3");
        assert_eq!(row3.state, SubmissionState::Queued);

        // Terminal-input cleanup should include both token(1) (deleted
        // 'ready' row) and token(2) (just-failed stale row), but not
        // token(3) (still queued).
        let mut cleanup = report.cleanup_input_paths.clone();
        cleanup.sort();
        assert_eq!(cleanup, vec!["/in1".to_string(), "/in2".to_string()]);
    }

    #[test]
    fn reap_does_not_report_the_same_input_path_twice() {
        let store = SubmissionStore::open_in_memory().expect("open");
        store.insert(token(1), 1, "/in1", 8, 8).unwrap();
        store.mark_cancelled(token(1)).unwrap();

        let first = store.reap(now_unix()).unwrap();
        assert_eq!(first.cleanup_input_paths, vec!["/in1".to_string()]);

        let second = store.reap(now_unix()).unwrap();
        assert!(second.cleanup_input_paths.is_empty());
    }
}
