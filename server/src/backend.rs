//! The seam between the HTTP/queue layer and the table reader.
//!
//! `LookupBackend` decouples the rest of the service from
//! `ntlmrain::local_lookup::LocalTable`: production code runs against
//! `LocalTableBackend`, while queue/worker tests run against `FakeBackend`
//! without needing a real GRTB table on disk.
//!
//! Wired into `main.rs` via `lib.rs::build_app` (Task 5): production runs
//! `LocalTableBackend`.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use ntlmrain::local_lookup::{LocalLookupError, LocalTable, LookupControl};
use thiserror::Error;

/// Errors a `LookupBackend` implementation can return.
#[derive(Debug, Error)]
pub enum LookupBackendError {
    /// The input (endpoint file, request shape, etc.) was invalid.
    /// Maps to an HTTP 400 in the request layer.
    #[error("{0}")]
    Invalid(String),

    /// The lookup failed for an internal reason (I/O error, table error,
    /// etc). Maps to a `failed` job state in the queue layer.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    /// The lookup was cancelled before it completed.
    #[error("lookup was cancelled")]
    Cancelled,
}

impl From<LocalLookupError> for LookupBackendError {
    fn from(error: LocalLookupError) -> Self {
        match error {
            LocalLookupError::Invalid(message) => LookupBackendError::Invalid(message),
            LocalLookupError::Io { context, source } => LookupBackendError::Io { context, source },
        }
    }
}

/// The seam between HTTP/queue code and the table reader.
pub trait LookupBackend: Send + Sync {
    fn lookup(
        &self,
        endpoint_file: &[u8],
        control: &LookupControl,
    ) -> Result<Vec<u8>, LookupBackendError>;
}

/// Production backend: looks up against a real on-disk GRTB/GIDX table.
pub struct LocalTableBackend(pub Arc<LocalTable>);

impl LookupBackend for LocalTableBackend {
    fn lookup(
        &self,
        endpoint_file: &[u8],
        control: &LookupControl,
    ) -> Result<Vec<u8>, LookupBackendError> {
        self.0
            .lookup_endpoint_file(endpoint_file, Some(control))
            .map_err(Into::into)
    }
}

/// Test backend used by Tasks 2/3/5 to exercise queue/worker behavior
/// without a real GRTB table on disk.
///
/// During its simulated sleep it ticks in `tick`-sized increments,
/// checking `control.is_cancelled()` on each tick (returning
/// `LookupBackendError::Cancelled` immediately if set) and advancing
/// `control.processed()` via `LookupControl::set_processed` so callers
/// polling progress see it move, the same way a real lookup's page reads
/// would.
pub struct FakeBackend {
    /// Total simulated work duration before returning success.
    sleep: Duration,
    /// Granularity of the cancellation-check/progress-tick loop.
    tick: Duration,
    /// Canned response bytes to return on success (a `NTLMCAN1` blob).
    response: Vec<u8>,
}

impl FakeBackend {
    pub fn new(sleep: Duration, tick: Duration, response: Vec<u8>) -> Self {
        Self {
            sleep,
            tick,
            response,
        }
    }
}

impl LookupBackend for FakeBackend {
    fn lookup(
        &self,
        _endpoint_file: &[u8],
        control: &LookupControl,
    ) -> Result<Vec<u8>, LookupBackendError> {
        let deadline = Instant::now() + self.sleep;
        let mut ticks: u64 = 0;
        loop {
            if control.is_cancelled() {
                return Err(LookupBackendError::Cancelled);
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline - now;
            std::thread::sleep(self.tick.min(remaining));
            ticks += 1;
            control.set_processed(ticks);
        }
        if control.is_cancelled() {
            return Err(LookupBackendError::Cancelled);
        }
        Ok(self.response.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntlmrain::local_lookup::{CandidateRecord, encode_candidate_file};
    use std::sync::Arc;
    use std::thread;

    fn canned_response() -> Vec<u8> {
        encode_candidate_file(
            1,
            &[CandidateRecord {
                ordinal: 0,
                start: 0x0088_46f7_eaee_8fb1,
            }],
        )
        .expect("encode candidate file")
    }

    #[test]
    fn fake_backend_sleeps_then_returns_canned_response() {
        let backend = FakeBackend::new(
            Duration::from_millis(30),
            Duration::from_millis(5),
            canned_response(),
        );
        let control = LookupControl::default();
        let started = Instant::now();
        let result = backend.lookup(&[], &control).expect("lookup succeeds");
        assert!(started.elapsed() >= Duration::from_millis(25));
        assert_eq!(result, canned_response());
    }

    #[test]
    fn fake_backend_advances_processed_during_simulated_sleep() {
        let backend = Arc::new(FakeBackend::new(
            Duration::from_millis(200),
            Duration::from_millis(5),
            canned_response(),
        ));
        let control = Arc::new(LookupControl::default());
        assert_eq!(control.processed(), 0);

        let worker_backend = Arc::clone(&backend);
        let worker_control = Arc::clone(&control);
        let handle = thread::spawn(move || worker_backend.lookup(&[], &worker_control));

        // Give the fake a chance to tick a few times before it finishes.
        thread::sleep(Duration::from_millis(60));
        let mid_flight = control.processed();
        assert!(
            mid_flight > 0,
            "expected processed() to have advanced mid-sleep, got {mid_flight}"
        );

        let result = handle.join().expect("worker thread panicked");
        assert!(result.is_ok());
        assert!(control.processed() >= mid_flight);
    }

    #[test]
    fn fake_backend_returns_cancelled_error_when_control_is_cancelled_up_front() {
        let backend = FakeBackend::new(
            Duration::from_millis(500),
            Duration::from_millis(5),
            canned_response(),
        );
        let control = LookupControl::default();
        control.cancel();
        let started = Instant::now();
        let result = backend.lookup(&[], &control);
        assert!(matches!(result, Err(LookupBackendError::Cancelled)));
        // Cancellation should short-circuit long before the full sleep.
        assert!(started.elapsed() < Duration::from_millis(200));
    }

    #[test]
    fn fake_backend_notices_cancellation_mid_sleep() {
        let backend = Arc::new(FakeBackend::new(
            Duration::from_secs(5),
            Duration::from_millis(5),
            canned_response(),
        ));
        let control = Arc::new(LookupControl::default());

        let worker_backend = Arc::clone(&backend);
        let worker_control = Arc::clone(&control);
        let handle = thread::spawn(move || worker_backend.lookup(&[], &worker_control));

        thread::sleep(Duration::from_millis(20));
        control.cancel();

        let started = Instant::now();
        let result = handle.join().expect("worker thread panicked");
        assert!(matches!(result, Err(LookupBackendError::Cancelled)));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
