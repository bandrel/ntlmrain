//! Dedicated-thread worker pool that drains `SubmissionStore`'s queue
//! through a `LookupBackend` and reports progress/results back.
//!
//! # Why `std::thread`, not tokio tasks
//!
//! `LookupBackend::lookup` (in production, `LocalTableBackend` ->
//! `LocalTable::lookup_endpoint_file`) is a blocking call that can run for
//! minutes and internally spawns its own `read_workers` OS threads via
//! `std::thread::scope`. Running that on a tokio worker thread (even via
//! `spawn_blocking`, whose pool is meant for short blocking calls) would
//! starve the async runtime. So each pool "slot" is a dedicated
//! `std::thread::spawn` worker that polls `SubmissionStore` directly and
//! never touches tokio.
//!
//! # Heartbeat/timeout mechanism (thread-per-job)
//!
//! Each in-flight job gets two extra short-lived helper threads, spawned
//! and joined by the worker thread around the blocking `backend.lookup`
//! call:
//!
//! - a **heartbeat thread** that calls `store.checkpoint_progress` every
//!   ~2s by reading `control.processed()`;
//! - a **timer thread** that calls `control.cancel()` (recording "timed
//!   out", not "client cancelled", via `JobControl`) if the job's
//!   `job_timeout` deadline passes before it finishes.
//!
//! A single shared timer (e.g. one thread scanning all in-flight jobs'
//! deadlines) would also work and uses fewer OS threads overall, but
//! thread-per-job keeps each job's cancellation/heartbeat logic local and
//! trivially testable in isolation, and `lookup_slots` is small (a handful
//! of concurrent jobs, not thousands), so the extra threads (2 per
//! in-flight job, bounded by `lookup_slots`) are not a scaling concern.
//! Both helper threads are stopped (via a stop channel) and joined as soon
//! as `backend.lookup` returns, so they never outlive the job.
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ntlmrain::local_lookup::{LookupControl, validate_candidate_file};

use crate::backend::{LookupBackend, LookupBackendError};
use crate::queue::{SubmissionStore, TokenHash};

/// How often an idle worker thread polls `claim_next_queued` for new work.
const POLL_INTERVAL: Duration = Duration::from_millis(200);
/// How often the heartbeat thread checkpoints progress for a running job.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
/// How often the reaper sweeps the store for expired/stale rows.
const REAP_INTERVAL: Duration = Duration::from_secs(60);

/// `ntlmrain::local_lookup::LookupControl` only exposes
/// `cancel()`/`is_cancelled()`/`processed()` and can't be extended, but the
/// worker needs to know *why* a job was cancelled (client request vs. job
/// timeout) to report the right terminal state/error. This wraps a
/// `LookupControl` with that extra bit.
struct JobControl {
    control: LookupControl,
    timed_out: AtomicBool,
}

impl JobControl {
    fn new() -> Self {
        Self {
            control: LookupControl::default(),
            timed_out: AtomicBool::new(false),
        }
    }

    fn lookup_control(&self) -> &LookupControl {
        &self.control
    }

    fn processed(&self) -> u64 {
        self.control.processed()
    }

    /// Cancel because the job exceeded its wall-clock time limit.
    fn cancel_for_timeout(&self) {
        self.timed_out.store(true, Ordering::Release);
        self.control.cancel();
    }

    /// Cancel because the client asked us to.
    fn cancel_for_client(&self) {
        self.control.cancel();
    }

    fn is_timed_out(&self) -> bool {
        self.timed_out.load(Ordering::Acquire)
    }
}

type ControlMap = Mutex<HashMap<TokenHash, Arc<JobControl>>>;

/// Cheaply cloneable handle onto a `WorkerPool`'s shared control map.
///
/// # Public API split (for Task 4/`http.rs`)
///
/// `WorkerPool` intentionally does *not* expose a `submit` method: job
/// submission (validating the request, writing the input blob, and
/// calling `store.insert`) stays entirely in `http.rs`, which already owns
/// the `SubmissionStore` and knows the HTTP-layer validation rules
/// (exact-records policy, queue-full -> 503, etc). The worker pool has no
/// opinion about any of that — it only needs to know about a job once it's
/// `queued` in the store.
///
/// What `http.rs` *does* need from this module is a way to (a) read live
/// progress for a `running` job without going through the store's
/// (checkpointed, ~2s-stale) `processed_records` column, and (b) request
/// cancellation of a running job by token. Both are keyed off the same
/// shared `token_sha256 -> Arc<JobControl>` map, so they're exposed
/// together on this one `Clone`-able, `Send + Sync` handle that `http.rs`
/// can hold in its `axum` state alongside the `SubmissionStore`.
#[derive(Clone)]
pub struct WorkerPoolHandle {
    controls: Arc<ControlMap>,
}

impl WorkerPoolHandle {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<TokenHash, Arc<JobControl>>> {
        self.controls.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Live `processed` count for a job a worker is actively running, or
    /// `None` if no worker currently holds this token (not running, or
    /// already finished/removed from the map). Callers needing a value
    /// regardless of whether a worker is active should fall back to the
    /// store's checkpointed `processed_records` column.
    pub fn processed(&self, token_sha256: TokenHash) -> Option<u64> {
        self.lock().get(&token_sha256).map(|c| c.processed())
    }

    /// Request cancellation of a running/queued-and-then-claimed job by
    /// token. Marks the cancellation as client-requested (as opposed to a
    /// timeout), so the worker reports it as `cancelled`, not `failed`.
    /// Returns `true` if an active control entry existed for this token
    /// (a worker is currently processing it), `false` otherwise — the
    /// caller (`http.rs`) should treat `false` as "nothing to cancel" (the
    /// job may be merely `queued` with no worker yet, unknown, or already
    /// terminal) and is not itself an error.
    pub fn cancel(&self, token_sha256: TokenHash) -> bool {
        match self.lock().get(&token_sha256) {
            Some(control) => {
                control.cancel_for_client();
                true
            }
            None => false,
        }
    }
}

/// Owns the pool's dedicated worker threads. Dropping it signals all
/// worker threads to stop polling for *new* work (they finish whatever
/// job they're currently on, if any) but does not join them — shutdown is
/// fire-and-forget, matching the fact that a job already in flight when
/// the process is asked to stop should not be forcibly aborted mid-write.
pub struct WorkerPool {
    handle: WorkerPoolHandle,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    /// Spawn `slots` dedicated worker threads pulling from `store` and
    /// running jobs through `backend`. Before spawning any workers, runs
    /// `store.restart_sweep()` once so `running` rows left over from a
    /// previous crash (this process died mid-job, e.g. lost power) are
    /// requeued (or failed, past the retry cap) instead of stuck forever.
    ///
    /// `job_timeout` bounds a single job's wall-clock run time.
    /// `max_match_records` bounds the result's `match_count`, enforced
    /// after the backend call succeeds. `state_dir` is the service's state
    /// directory; results are written under `<state_dir>/jobs/`.
    pub fn spawn(
        slots: usize,
        job_timeout: Duration,
        max_match_records: u64,
        state_dir: impl Into<PathBuf>,
        store: Arc<SubmissionStore>,
        backend: Arc<dyn LookupBackend>,
    ) -> Self {
        let state_dir = state_dir.into();
        if let Err(error) = store.restart_sweep() {
            eprintln!("worker pool: startup restart_sweep failed: {error}");
        }

        let controls: Arc<ControlMap> = Arc::new(Mutex::new(HashMap::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let workers = (0..slots)
            .map(|_| {
                let store = Arc::clone(&store);
                let backend = Arc::clone(&backend);
                let controls = Arc::clone(&controls);
                let shutdown = Arc::clone(&shutdown);
                let state_dir = state_dir.clone();
                thread::spawn(move || {
                    worker_loop(
                        &store,
                        backend.as_ref(),
                        &controls,
                        &shutdown,
                        job_timeout,
                        max_match_records,
                        &state_dir,
                    )
                })
            })
            .collect();

        WorkerPool {
            handle: WorkerPoolHandle { controls },
            shutdown,
            workers,
        }
    }

    /// A cloneable handle for `http.rs` (progress reads + cancellation).
    pub fn handle(&self) -> WorkerPoolHandle {
        self.handle.clone()
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // Deliberately not joined: a job in flight when the pool is
        // dropped keeps running to whatever natural conclusion it
        // reaches, rather than blocking the dropper indefinitely.
        let _ = &self.workers;
    }
}

fn worker_loop(
    store: &Arc<SubmissionStore>,
    backend: &dyn LookupBackend,
    controls: &Arc<ControlMap>,
    shutdown: &AtomicBool,
    job_timeout: Duration,
    max_match_records: u64,
    state_dir: &Path,
) {
    while !shutdown.load(Ordering::Acquire) {
        let claimed = match store.claim_next_queued() {
            Ok(Some(job)) => job,
            Ok(None) => {
                thread::sleep(POLL_INTERVAL);
                continue;
            }
            Err(error) => {
                eprintln!("worker: claim_next_queued failed: {error}");
                thread::sleep(POLL_INTERVAL);
                continue;
            }
        };
        run_job(
            store,
            backend,
            controls,
            job_timeout,
            max_match_records,
            state_dir,
            claimed.token_sha256,
            &claimed.input_path,
            claimed.record_count,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_job(
    store: &Arc<SubmissionStore>,
    backend: &dyn LookupBackend,
    controls: &Arc<ControlMap>,
    job_timeout: Duration,
    max_match_records: u64,
    state_dir: &Path,
    token_sha256: TokenHash,
    input_path: &str,
    record_count: i64,
) {
    let input_blob = match fs::read(input_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = store.mark_failed(token_sha256, &format!("failed to read job input: {error}"));
            let _ = fs::remove_file(input_path);
            return;
        }
    };

    let job_control = Arc::new(JobControl::new());
    controls
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(token_sha256, Arc::clone(&job_control));

    // Heartbeat thread: checkpoints live progress into the store every
    // ~2s while the job runs. Stopped via a channel rather than a shared
    // atomic + sleep loop so it wakes immediately on shutdown instead of
    // waiting out its last sleep.
    let (heartbeat_stop_tx, heartbeat_stop_rx) = mpsc::channel::<()>();
    let heartbeat = {
        let store = Arc::clone(store);
        let job_control = Arc::clone(&job_control);
        thread::spawn(move || {
            loop {
                match heartbeat_stop_rx.recv_timeout(HEARTBEAT_INTERVAL) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {
                        let _ = store.checkpoint_progress(token_sha256, job_control.processed());
                    }
                }
            }
        })
    };

    // Timer thread: cancels the job (tagged "timed out") if it's still
    // running once `job_timeout` elapses.
    let (timer_stop_tx, timer_stop_rx) = mpsc::channel::<()>();
    let timer = {
        let job_control = Arc::clone(&job_control);
        thread::spawn(move || {
            if let Err(RecvTimeoutError::Timeout) = timer_stop_rx.recv_timeout(job_timeout) {
                job_control.cancel_for_timeout();
            }
        })
    };

    // Guard against a backend panic (malformed table page, an internal
    // `unwrap`, allocation failure, ...): without this, the panic would
    // unwind straight out of `worker_loop`, permanently retiring this
    // worker's thread (dropping the pool from N to N-1 slots for the rest
    // of the process's life), leaking this job's `controls` entry forever
    // (so `WorkerPoolHandle::cancel`/`processed` would keep reporting a
    // dead job as live), and leaving the row stuck `running` until a full
    // restart. Catching it here lets the worker report `failed` and keep
    // looping, same as any other backend error.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.lookup(&input_blob, job_control.lookup_control())
    }))
    .unwrap_or_else(|_| Err(LookupBackendError::Invalid("lookup panicked".to_string())));

    // Stop and join both helper threads before touching the store again,
    // so a slow heartbeat/timer thread can't race the terminal-state
    // write below.
    let _ = heartbeat_stop_tx.send(());
    let _ = heartbeat.join();
    let _ = timer_stop_tx.send(());
    let _ = timer.join();

    match result {
        Ok(result_bytes) => {
            match validate_candidate_file(&result_bytes, Some(record_count as u64)) {
                Ok(match_count) => {
                    if match_count > max_match_records {
                        let _ =
                            store.mark_failed(token_sha256, "result exceeds max-match-records cap");
                    } else {
                        match write_result_blob(state_dir, token_sha256, &result_bytes) {
                            Ok(result_path) => {
                                finalize_ready(store, token_sha256, match_count, &result_path);
                            }
                            Err(error) => {
                                let _ = store.mark_failed(
                                    token_sha256,
                                    &format!("failed to write lookup result: {error}"),
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    let _ =
                        store.mark_failed(token_sha256, &format!("invalid lookup result: {error}"));
                }
            }
        }
        Err(LookupBackendError::Cancelled) => {
            if job_control.is_timed_out() {
                let _ = store.mark_failed(token_sha256, "lookup exceeded the job time limit");
            } else {
                let _ = store.mark_cancelled(token_sha256);
            }
        }
        Err(error) => {
            let _ = store.mark_failed(token_sha256, &error.to_string());
        }
    }

    controls
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&token_sha256);
    let _ = fs::remove_file(input_path);
}

/// Marks a completed job `ready`, and cleans up after itself if it lost a
/// race: `store.mark_ready` is guarded to `state = 'running'` and returns
/// `false` if the row already left that state (e.g. a concurrent cancel
/// won). In that case the result file we just wrote to disk is now
/// unreferenced by any row — the reaper only ever cleans up `result_path`s
/// it finds on a `ready` row — so we delete it ourselves here rather than
/// leaking it forever.
fn finalize_ready(
    store: &SubmissionStore,
    token_sha256: TokenHash,
    match_count: u64,
    result_path: &str,
) {
    if let Ok(false) = store.mark_ready(token_sha256, match_count, result_path) {
        let _ = fs::remove_file(result_path);
    }
}

/// Write `bytes` to `<state_dir>/jobs/<hex[0:2]>/<hex>.can` via a
/// `.tmp`-then-rename in the same directory (atomic on the same
/// filesystem). Returns the final path as a `String` for
/// `SubmissionStore::mark_ready`.
fn write_result_blob(
    state_dir: &Path,
    token_sha256: TokenHash,
    bytes: &[u8],
) -> std::io::Result<String> {
    let hex = hex::encode(token_sha256);
    let dir = state_dir.join("jobs").join(&hex[0..2]);
    fs::create_dir_all(&dir)?;
    let final_path = dir.join(format!("{hex}.can"));
    let tmp_path = dir.join(format!("{hex}.can.tmp"));
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, &final_path)?;
    Ok(final_path.to_string_lossy().into_owned())
}

/// Background thread that periodically calls `store.reap` and deletes the
/// filesystem paths it reports. Like `WorkerPool`, dropping it signals the
/// thread to stop (via the same stop-channel pattern) but does not join.
pub struct Reaper {
    stop_tx: mpsc::Sender<()>,
}

impl Reaper {
    pub fn spawn(store: Arc<SubmissionStore>) -> Self {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        thread::spawn(move || {
            loop {
                match stop_rx.recv_timeout(REAP_INTERVAL) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {
                        reap_once(&store);
                    }
                }
            }
        });
        Reaper { stop_tx }
    }
}

impl Drop for Reaper {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
    }
}

/// One reap sweep: ask the store what to clean up, then delete those
/// filesystem paths. Split out from `Reaper::spawn`'s loop so tests can
/// call a single sweep directly without waiting on `REAP_INTERVAL`.
fn reap_once(store: &SubmissionStore) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_secs() as i64;
    let report = match store.reap(now) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("reaper: sweep failed: {error}");
            return;
        }
    };
    for path in report
        .deleted_result_paths
        .iter()
        .chain(report.cleanup_input_paths.iter())
    {
        if let Err(error) = fs::remove_file(path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("reaper: failed to delete {path}: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::FakeBackend;
    use crate::queue::SubmissionState;
    use ntlmrain::local_lookup::{CandidateRecord, encode_candidate_file};
    use std::io::Write;

    fn token(byte: u8) -> TokenHash {
        [byte; 32]
    }

    fn canned_response() -> Vec<u8> {
        encode_candidate_file(
            1,
            &[CandidateRecord {
                ordinal: 0,
                start: 42,
            }],
        )
        .expect("encode candidate file")
    }

    /// Writes an arbitrary input blob to `dir` and returns its path as a
    /// `String` (what `SubmissionStore::insert` expects). Content is
    /// irrelevant to `FakeBackend`, which ignores its `endpoint_file` arg.
    fn write_input(dir: &Path, name: &str) -> String {
        let path = dir.join(name);
        let mut file = fs::File::create(&path).expect("create input blob");
        file.write_all(b"fake endpoint blob")
            .expect("write input blob");
        path.to_string_lossy().into_owned()
    }

    fn wait_until<F: FnMut() -> bool>(mut predicate: F, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if predicate() {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn submitted_job_transitions_queued_running_ready_with_result_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap());
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_millis(50),
            Duration::from_millis(5),
            canned_response(),
        ));

        let input_path = write_input(dir.path(), "in1.end");
        store
            .insert(token(1), 1, &input_path, 8, 8)
            .expect("insert");
        assert_eq!(
            store.get(token(1)).unwrap().unwrap().state,
            SubmissionState::Queued
        );

        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Ready)
            },
            Duration::from_secs(5)
        ));

        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.match_count, Some(1));
        let result_path = row.result_path.expect("result_path set");
        assert!(Path::new(&result_path).exists());
        assert_eq!(fs::read(&result_path).unwrap(), canned_response());

        // Input blob deleted on terminal state.
        assert!(!Path::new(&input_path).exists());

        drop(pool);
    }

    #[test]
    fn queue_position_reflects_a_single_slot_serializing_two_jobs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap());
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_millis(300),
            Duration::from_millis(10),
            canned_response(),
        ));

        let input1 = write_input(dir.path(), "in1.end");
        let input2 = write_input(dir.path(), "in2.end");
        store.insert(token(1), 1, &input1, 8, 1).unwrap();
        store.insert(token(2), 1, &input2, 8, 1).unwrap();

        assert_eq!(store.queue_position(token(2)).unwrap(), Some(1));

        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        // Wait for the first job to start running.
        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Running)
            },
            Duration::from_secs(5)
        ));
        // With only one slot, token(2) is still queued at position 0
        // (nothing queued ahead of it) while token(1) runs.
        assert_eq!(store.queue_position(token(2)).unwrap(), Some(0));
        assert_eq!(
            store.get(token(2)).unwrap().unwrap().state,
            SubmissionState::Queued
        );

        // Once both finish, token(2) is no longer queued at all.
        assert!(wait_until(
            || {
                store
                    .get(token(2))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Ready)
            },
            Duration::from_secs(5)
        ));
        assert_eq!(store.queue_position(token(2)).unwrap(), Some(0));

        drop(pool);
    }

    #[test]
    fn cancelling_a_running_job_lands_it_in_cancelled_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap());
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_secs(5),
            Duration::from_millis(10),
            canned_response(),
        ));

        let input_path = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path, 8, 8).unwrap();

        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Running)
            },
            Duration::from_secs(5)
        ));

        assert!(pool.handle().cancel(token(1)));

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Cancelled)
            },
            Duration::from_secs(5)
        ));

        // Store-level state proves the transition; Task 4 maps
        // state=='cancelled' to an HTTP 404 (not this task's concern).
        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.state, SubmissionState::Cancelled);

        drop(pool);
    }

    #[test]
    fn job_exceeding_timeout_ends_in_failed_with_timeout_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap());
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_secs(5),
            Duration::from_millis(10),
            canned_response(),
        ));

        let input_path = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path, 8, 8).unwrap();

        let pool = WorkerPool::spawn(
            1,
            Duration::from_millis(50),
            1_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Failed)
            },
            Duration::from_secs(5)
        ));

        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.state, SubmissionState::Failed);
        assert_eq!(
            row.error.as_deref(),
            Some("lookup exceeded the job time limit")
        );

        drop(pool);
    }

    #[test]
    fn result_exceeding_max_match_records_fails_the_job_without_writing_a_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap());
        // Two candidate records in the canned response, cap of 1.
        let response = encode_candidate_file(
            1,
            &[
                CandidateRecord {
                    ordinal: 0,
                    start: 1,
                },
                CandidateRecord {
                    ordinal: 0,
                    start: 2,
                },
            ],
        )
        .unwrap();
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_millis(20),
            Duration::from_millis(5),
            response,
        ));

        let input_path = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path, 8, 8).unwrap();

        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Failed)
            },
            Duration::from_secs(5)
        ));

        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(
            row.error.as_deref(),
            Some("result exceeds max-match-records cap")
        );
        assert!(row.result_path.is_none());
        assert!(!dir.path().join("jobs").exists());

        drop(pool);
    }

    #[test]
    fn restarting_the_pool_over_the_same_db_picks_up_a_crashed_job_via_restart_sweep() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("queue.sqlite3");
        let input_path = write_input(dir.path(), "in1.end");

        {
            let store = Arc::new(SubmissionStore::open(&db_path).unwrap());
            store.insert(token(1), 1, &input_path, 8, 8).unwrap();
            // A backend that sleeps far longer than this test waits, so the
            // job is still "running" (never reaches mark_ready) when the
            // pool below is dropped, simulating a mid-job crash.
            let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
                Duration::from_secs(30),
                Duration::from_millis(10),
                canned_response(),
            ));
            let pool = WorkerPool::spawn(
                1,
                Duration::from_secs(120),
                1_000,
                dir.path().to_path_buf(),
                Arc::clone(&store),
                backend,
            );
            assert!(wait_until(
                || {
                    store
                        .get(token(1))
                        .unwrap()
                        .is_some_and(|row| row.state == SubmissionState::Running)
                },
                Duration::from_secs(5)
            ));
            // Drop without graceful shutdown: the worker thread is left
            // running detached (see `WorkerPool`'s Drop doc), but since its
            // FakeBackend sleeps for 30s and this test doesn't wait for it,
            // it never gets to call back into the store before the process
            // (or here, the test binary) is done with it.
            drop(pool);
        }

        // Re-open the same on-disk store and start a fresh pool with a fast
        // backend. `WorkerPool::spawn` runs `restart_sweep()` first, which
        // requeues the still-`running` row (attempts <= 2) left over from
        // the "crash" above.
        let store = Arc::new(SubmissionStore::open(&db_path).unwrap());
        let row_before = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row_before.state, SubmissionState::Running);

        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_millis(20),
            Duration::from_millis(5),
            canned_response(),
        ));
        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Ready)
            },
            Duration::from_secs(5)
        ));
        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.attempts, 1);

        drop(pool);
    }

    #[test]
    fn reap_once_deletes_reported_filesystem_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap();

        let input_path = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path, 8, 8).unwrap();
        store.mark_cancelled(token(1)).unwrap();
        assert!(Path::new(&input_path).exists());

        reap_once(&store);

        assert!(!Path::new(&input_path).exists());
        let row = store.get(token(1)).unwrap().unwrap();
        assert!(row.input_cleaned);
    }

    #[test]
    fn finalize_ready_deletes_the_result_file_when_mark_ready_loses_the_race() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap();
        let input_path = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path, 8, 8).unwrap();
        store.claim_next_queued().unwrap();

        // Simulate a cancel landing between the worker writing the result
        // file and calling mark_ready: the row leaves 'running' first.
        assert!(store.mark_cancelled(token(1)).unwrap());

        let result_path = dir.path().join("orphan.can");
        fs::write(&result_path, b"result bytes").unwrap();
        assert!(result_path.exists());

        finalize_ready(&store, token(1), 1, result_path.to_str().unwrap());

        // mark_ready lost the race (row is 'cancelled', not 'running'), so
        // the file we just wrote must not be left behind on disk forever.
        assert!(!result_path.exists());
        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.state, SubmissionState::Cancelled);
    }

    #[test]
    fn finalize_ready_leaves_the_result_file_in_place_when_mark_ready_wins() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap();
        let input_path = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path, 8, 8).unwrap();
        store.claim_next_queued().unwrap();

        let result_path = dir.path().join("result.can");
        fs::write(&result_path, b"result bytes").unwrap();

        finalize_ready(&store, token(1), 3, result_path.to_str().unwrap());

        assert!(result_path.exists());
        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.state, SubmissionState::Ready);
        assert_eq!(row.match_count, Some(3));
    }

    /// Always panics; used to prove a backend panic doesn't retire a
    /// worker slot or leak a `controls` map entry.
    struct PanicBackend;

    impl LookupBackend for PanicBackend {
        fn lookup(
            &self,
            _endpoint_file: &[u8],
            _control: &LookupControl,
        ) -> std::result::Result<Vec<u8>, LookupBackendError> {
            panic!("boom: simulated backend panic");
        }
    }

    #[test]
    fn backend_panic_fails_the_job_and_the_worker_keeps_looping() {
        // This test intentionally triggers a caught panic; expect to see
        // "thread '...' panicked at ...: boom: simulated backend panic"
        // on stderr even though the test passes — that's the default
        // panic hook, not a test failure.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).unwrap());
        let backend: Arc<dyn LookupBackend> = Arc::new(PanicBackend);

        let input_path1 = write_input(dir.path(), "in1.end");
        store.insert(token(1), 1, &input_path1, 8, 8).unwrap();

        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );

        assert!(wait_until(
            || {
                store
                    .get(token(1))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Failed)
            },
            Duration::from_secs(5)
        ));
        let row = store.get(token(1)).unwrap().unwrap();
        assert_eq!(row.error.as_deref(), Some("lookup panicked"));

        // The panic was caught and the controls-map entry was cleaned up
        // like any other terminal job: nothing left to cancel.
        assert!(!pool.handle().cancel(token(1)));

        // Prove the worker's thread survived the panic and is still
        // looping (not a permanently retired slot): a second job gets
        // picked up and driven to a terminal state too.
        let input_path2 = write_input(dir.path(), "in2.end");
        store.insert(token(2), 1, &input_path2, 8, 8).unwrap();
        assert!(wait_until(
            || {
                store
                    .get(token(2))
                    .unwrap()
                    .is_some_and(|row| row.state == SubmissionState::Failed)
            },
            Duration::from_secs(5)
        ));

        drop(pool);
    }
}
