//! `ntlmrain-server` library crate: exposes every module `main.rs` (and
//! `tests/e2e.rs`, which links against this crate like any other
//! downstream consumer) needs, plus [`build_app`], the single wiring
//! function that assembles a real table + queue + worker pool + router
//! from a [`Config`]. Factored out of `main.rs` (Task 5) specifically so
//! integration tests can drive the exact same startup path the binary
//! uses instead of re-implementing it.

pub mod auth;
pub mod backend;
pub mod config;
pub mod http;
pub mod queue;
pub mod worker;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Context;
use axum::Router;

use backend::{LocalTableBackend, LookupBackend};
use config::Config;
use ntlmrain::local_lookup::{LocalLookupOptions, LocalTable};
use queue::SubmissionStore;
use worker::{Reaper, WorkerPool};

/// Everything [`build_app`] assembled: the router `main.rs` (or a test)
/// serves, plus the background workers it must keep alive for the
/// process/test's lifetime. Dropping `pool`/`reaper` stops their threads
/// (see their own `Drop` impls), so callers must hold onto this for as
/// long as the router is being served.
pub struct App {
    pub router: Router,
    pub store: Arc<SubmissionStore>,
    pub pool: WorkerPool,
    pub reaper: Reaper,
}

/// The real `main.rs` wiring path: open the table, build the queue store,
/// spin up the worker pool and reaper, and build the router. Returns an
/// error (rather than exiting the process) so both `main.rs` and
/// integration tests can decide how to react to a startup failure.
///
/// Order matches the plan: table open first (so a bad table path fails
/// fast, before any listener could ever answer `/health/ready` with a
/// misleading "ready"), then the queue store, then the worker pool (whose
/// `spawn` runs the startup `restart_sweep` itself -- see its doc comment)
/// and reaper, then the router.
pub fn build_app(config: Config) -> anyhow::Result<App> {
    let config = Arc::new(config);

    std::fs::create_dir_all(&config.state_dir).with_context(|| {
        format!(
            "failed to create --state-dir {}",
            config.state_dir.display()
        )
    })?;

    let table = LocalTable::open(LocalLookupOptions {
        data_base: config.data_base.clone(),
        index_path: config.index.clone(),
        read_workers: config.read_workers,
        preload_index: config.preload_index,
        lock_index: config.lock_index,
    })
    .with_context(|| {
        format!(
            "failed to open local lookup table (--data-base {}, --index {})",
            config.data_base.display(),
            config.index.display()
        )
    })?;
    let table_info = Arc::new(table.info().clone());
    let backend: Arc<dyn LookupBackend> = Arc::new(LocalTableBackend(Arc::new(table)));

    let store = Arc::new(
        SubmissionStore::open(config.state_dir.join("queue.sqlite3"))
            .context("failed to open submission queue store")?,
    );
    // No explicit startup sweep here: `WorkerPool::spawn` below runs
    // `store.restart_sweep()` itself before spawning any worker threads
    // (see its doc comment), and this function creates exactly one
    // `WorkerPool` over this `store`, so that single internal sweep is the
    // only one this process ever needs. (An earlier version of this
    // function also called `store.restart_sweep()` directly here first --
    // redundant with, and always running strictly before, the sweep inside
    // `WorkerPool::spawn`, so it was always a no-op. Removed rather than
    // kept "for clarity": a duplicate call site is exactly the kind of
    // thing that silently stops being a no-op if `WorkerPool::spawn`'s
    // internals ever change, and `WorkerPool` already owns this
    // responsibility.)
    let pool = WorkerPool::spawn(
        config.lookup_slots,
        Duration::from_secs(config.job_timeout_secs),
        config.max_match_records,
        config.state_dir.clone(),
        Arc::clone(&store),
        backend,
    );
    let reaper = Reaper::spawn(Arc::clone(&store), config.state_dir.clone());

    // The table opened successfully above, so the service is ready the
    // moment the router starts serving -- there is no separate async
    // warm-up phase in this implementation.
    let ready = Arc::new(AtomicBool::new(true));
    let router = http::build_router(
        Arc::clone(&config),
        Arc::clone(&store),
        pool.handle(),
        ready,
        Some(table_info),
    )
    .context("failed to build HTTP router")?;

    Ok(App {
        router,
        store,
        pool,
        reaper,
    })
}
