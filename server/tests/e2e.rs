//! End-to-end tests: real `TcpListener`, real `ntlmrain_server::build_app`
//! wiring (or, for the cancellation/protocol tests, the same
//! store+`FakeBackend`+router assembly `http.rs`'s own unit tests use,
//! per the task brief's "the cancel behavior doesn't depend on the
//! table" note), and -- for the actual lookup round trip -- the real
//! `ntlmrain::remote_lookup::RemoteLookupClient` the CLI itself uses.

mod fixtures;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ntlmrain::local_lookup::{
    CANDIDATE_HEADER_BYTES, CANDIDATE_RECORD_BYTES, validate_candidate_file,
};
use ntlmrain::remote_lookup::{RemoteLookupClient, RemoteLookupConfig};
use ntlmrain_server::backend::{FakeBackend, LookupBackend};
use ntlmrain_server::config::Config;
use ntlmrain_server::queue::SubmissionStore;
use ntlmrain_server::worker::{Reaper, WorkerPool};
use ntlmrain_server::{App, build_app};

/// Keeps a spawned server's background threads (worker pool, reaper) and
/// store alive for as long as a test holds this, and reports the address
/// it's listening on. The `axum::serve` task itself runs detached
/// (`tokio::spawn`) and is abandoned at the end of the test process --
/// acceptable for a test binary.
struct ServerGuard {
    addr: SocketAddr,
    _pool: WorkerPool,
    _reaper: Reaper,
    _store: Arc<SubmissionStore>,
}

async fn bind(router: axum::Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    (addr, handle)
}

/// Spawns a real server via `ntlmrain_server::build_app` (the exact
/// `main.rs` wiring path) against the given `Config`, which must point at
/// a real GIDX/GRTB table (this test module's `fixtures::Fixture`).
async fn spawn_real_app(config: Config) -> ServerGuard {
    let App {
        router,
        store,
        pool,
        reaper,
    } = build_app(config).expect("build_app");
    let (addr, _serve_task) = bind(router).await;
    ServerGuard {
        addr,
        _pool: pool,
        _reaper: reaper,
        _store: store,
    }
}

/// Spawns a server wired directly against a `FakeBackend` instead of a
/// real table -- mirrors `http.rs`'s own `Harness` test helper, minus the
/// `oneshot` in-process calls (this binds a real listener instead). Used
/// by the cancellation and manual-protocol tests, which don't need (and,
/// for the cancellation test, actively don't want -- it wants a
/// controllable "slow lookup") a real table.
async fn spawn_fake_backend_app(
    dir: &std::path::Path,
    mut mutate_config: impl FnMut(&mut Config),
    backend_sleep: Duration,
    backend_response: Vec<u8>,
) -> ServerGuard {
    let mut config = base_fake_config(dir.to_path_buf());
    mutate_config(&mut config);
    let config = Arc::new(config);

    let store =
        Arc::new(SubmissionStore::open(dir.join("queue.sqlite3")).expect("open submission store"));
    let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
        backend_sleep,
        Duration::from_millis(5),
        backend_response,
    ));
    let pool = WorkerPool::spawn(
        config.lookup_slots,
        Duration::from_secs(config.job_timeout_secs),
        config.max_match_records,
        dir.to_path_buf(),
        Arc::clone(&store),
        backend,
    );
    let reaper = Reaper::spawn(Arc::clone(&store));
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let router = ntlmrain_server::http::build_router(
        Arc::clone(&config),
        Arc::clone(&store),
        pool.handle(),
        ready,
    )
    .expect("build router");

    let (addr, _serve_task) = bind(router).await;
    ServerGuard {
        addr,
        _pool: pool,
        _reaper: reaper,
        _store: store,
    }
}

fn base_fake_config(state_dir: PathBuf) -> Config {
    Config {
        data_base: PathBuf::from("/unused/table"),
        index: PathBuf::from("/unused/table.gidx"),
        read_workers: 1,
        preload_index: false,
        lock_index: false,
        listen: "127.0.0.1:0".to_string(),
        lookup_slots: 1,
        state_dir,
        max_queued: 8,
        max_concurrent_uploads: 16,
        exact_records: 2,
        any_record_count: false,
        max_match_records: 8_000_000,
        job_timeout_secs: 900,
        auth_user: None,
        auth_password_file: None,
        auth_password: None,
        static_dir: None,
    }
}

fn real_table_config(fixture: &fixtures::Fixture, state_dir: PathBuf) -> Config {
    Config {
        data_base: fixture.data_base.clone(),
        index: fixture.index_path.clone(),
        read_workers: 1,
        preload_index: false,
        lock_index: false,
        listen: "127.0.0.1:0".to_string(),
        lookup_slots: 1,
        state_dir,
        max_queued: 8,
        max_concurrent_uploads: 16,
        // Matches the fixture's fixed 8-endpoint page exactly.
        exact_records: fixture.endpoints.len() as u64,
        any_record_count: false,
        max_match_records: 8_000_000,
        job_timeout_secs: 900,
        auth_user: None,
        auth_password_file: None,
        auth_password: None,
        static_dir: None,
    }
}

/// Decodes an `NTLMCAN1` blob's `(ordinal, start)` records, bypassing
/// `RemoteLookupClient`'s own (already-tested) parsing so the assertion
/// doesn't just check that the client's internal validation passed.
fn decode_candidate_records(bytes: &[u8]) -> Vec<(u64, u64)> {
    let mut records = Vec::new();
    let mut offset = CANDIDATE_HEADER_BYTES;
    while offset + CANDIDATE_RECORD_BYTES <= bytes.len() {
        let ordinal = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        let start = u64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
        records.push((ordinal, start));
        offset += CANDIDATE_RECORD_BYTES;
    }
    records
}

/// Covers the brief's item 1: drives the real `RemoteLookupClient`
/// against a real `build_app`-wired server backed by the synthetic
/// fixture, once anonymously (with a guaranteed match, and separately a
/// guaranteed non-match) and once with Basic auth configured end to end.
#[tokio::test]
async fn real_client_lookup_round_trip_anonymous_and_authenticated() {
    let fixture = fixtures::build_fixture();
    eprintln!(
        "fixture prefix directory build took {:?}",
        fixture.prefix_build_time
    );

    // --- Anonymous server: a guaranteed match, then a guaranteed non-match. ---
    let anon_dir = tempfile::tempdir().expect("tempdir");
    let anon_config = real_table_config(&fixture, anon_dir.path().to_path_buf());
    let anon = spawn_real_app(anon_config).await;

    let matching_file = fixtures::encode_endpoint_file(&fixture.endpoints);
    let out = run_remote_lookup(anon.addr, None, None, matching_file).await;
    let match_count = validate_candidate_file(&out, Some(fixture.endpoints.len() as u64))
        .expect("valid NTLMCAN1");
    assert_eq!(match_count, fixture.endpoints.len() as u64);
    let records = decode_candidate_records(&out);
    assert_eq!(records.len(), fixture.endpoints.len());
    for (ordinal, start) in &records {
        assert_eq!(
            *start, fixture.starts[*ordinal as usize],
            "candidate start for ordinal {ordinal} should be the fixture's matching start"
        );
    }

    let out_of_range = fixture.out_of_range_endpoint();
    let non_matching_file =
        fixtures::encode_endpoint_file(&vec![out_of_range; fixture.endpoints.len()]);
    let out = run_remote_lookup(anon.addr, None, None, non_matching_file).await;
    let match_count = validate_candidate_file(&out, Some(fixture.endpoints.len() as u64))
        .expect("valid NTLMCAN1");
    assert_eq!(
        match_count, 0,
        "an out-of-range endpoint must produce zero candidates"
    );

    // --- Authenticated server, same fixture, fresh port. ---
    // Uses `--auth-password-file` (the intended production credential
    // mechanism per config.rs's own comments), not `auth_password` (the
    // env-var fallback) -- with a trailing newline, the way a
    // `echo hunter2 > pw` file would actually look, so this also
    // exercises `auth.rs`'s `.trim_end_matches(['\n', '\r'])`.
    let auth_dir = tempfile::tempdir().expect("tempdir");
    let password_path = auth_dir.path().join("password");
    std::fs::write(&password_path, "hunter2\n").expect("write password file");
    let mut auth_config = real_table_config(&fixture, auth_dir.path().to_path_buf());
    auth_config.auth_user = Some("alice".to_string());
    auth_config.auth_password_file = Some(password_path);
    let auth = spawn_real_app(auth_config).await;

    let matching_file = fixtures::encode_endpoint_file(&fixture.endpoints);
    let out = run_remote_lookup(
        auth.addr,
        Some("alice".to_string()),
        Some("hunter2".to_string()),
        matching_file,
    )
    .await;
    let match_count = validate_candidate_file(&out, Some(fixture.endpoints.len() as u64))
        .expect("valid NTLMCAN1");
    assert_eq!(match_count, fixture.endpoints.len() as u64);
}

/// Runs one full submit -> poll -> download cycle through the real
/// `RemoteLookupClient`, off the tokio runtime (`spawn_blocking`, since
/// the client is a blocking `reqwest` client).
async fn run_remote_lookup(
    addr: SocketAddr,
    username: Option<String>,
    password: Option<String>,
    endpoint_file: Vec<u8>,
) -> Vec<u8> {
    // `RemoteLookupClient` wraps a `reqwest::blocking::Client`, which owns
    // its own background tokio runtime: both constructing *and* dropping
    // it must happen off this test's own tokio runtime thread, or it
    // panics ("cannot drop a runtime in a context where blocking is not
    // allowed"). So the whole thing -- construction, call, and the
    // client's eventual drop at the end of this closure -- runs inside
    // `spawn_blocking`, not just the `lookup` call.
    tokio::task::spawn_blocking(move || {
        let client = RemoteLookupClient::new(RemoteLookupConfig {
            base_url: format!("http://{addr}"),
            username,
            password,
            poll_interval: Duration::from_millis(10),
            request_timeout: Duration::from_secs(30),
        })
        .expect("construct RemoteLookupClient");
        client.lookup(&endpoint_file, None, |_progress| {})
    })
    .await
    .expect("spawn_blocking join")
    .expect("remote lookup succeeds")
}

/// Covers the brief's item 2: a real cancellation round trip against a
/// deliberately slow `FakeBackend`-wired server (per the brief, the
/// cancel path doesn't depend on the table, so a real one isn't needed
/// here).
#[tokio::test]
async fn cancelling_a_submission_makes_status_404_afterward() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = spawn_fake_backend_app(
        dir.path(),
        |_| {},
        Duration::from_secs(30), // far longer than this test waits
        canned_candidate_response(),
    )
    .await;

    let http = reqwest::Client::new();
    let base = format!("http://{}", server.addr);

    let endpoint_file = fixtures::encode_endpoint_file(&[1, 2]);
    let submit = http
        .post(format!("{base}/api/v1/submissions"))
        .header("content-type", "application/vnd.netntlmv1.endpoints")
        .body(endpoint_file)
        .send()
        .await
        .expect("submit request");
    assert_eq!(submit.status(), reqwest::StatusCode::ACCEPTED);
    let body: serde_json::Value = submit.json().await.expect("submit response json");
    let token = body["submission_token"].as_str().unwrap().to_string();

    // Give the (single-slot) worker a moment to actually claim the job,
    // so the cancel below exercises the "worker control" path rather
    // than racing a job that's still merely queued.
    wait_until(
        || async {
            let status = http
                .post(format!("{base}/api/v1/submissions/status"))
                .json(&serde_json::json!({"submission_token": token}))
                .send()
                .await
                .expect("status request");
            let body: serde_json::Value = status.json().await.expect("status json");
            body["state"] == "running"
        },
        Duration::from_secs(5),
    )
    .await;

    let cancel = http
        .post(format!("{base}/api/v1/submissions/cancel"))
        .json(&serde_json::json!({"submission_token": token}))
        .send()
        .await
        .expect("cancel request");
    assert_eq!(cancel.status(), reqwest::StatusCode::NO_CONTENT);

    // Cancelled is an internal-only state -- the wire contract 404s it
    // just like an unknown token (no oracle).
    let status = http
        .post(format!("{base}/api/v1/submissions/status"))
        .json(&serde_json::json!({"submission_token": token}))
        .send()
        .await
        .expect("status request");
    assert_eq!(status.status(), reqwest::StatusCode::NOT_FOUND);
}

/// Covers the brief's item 3: manual (non-`RemoteLookupClient`) protocol
/// checks against a real running server -- malformed requests the CLI's
/// own client would never construct.
#[tokio::test]
async fn unknown_token_and_malformed_submission_bodies_get_the_documented_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = spawn_fake_backend_app(
        dir.path(),
        |_| {},
        Duration::from_millis(20),
        canned_candidate_response(),
    )
    .await;
    let http = reqwest::Client::new();
    let base = format!("http://{}", server.addr);

    // Unknown token on /status -> 404 with the exact wire-contract detail
    // string (see http.rs's UNKNOWN_TOKEN_DETAIL).
    let unknown = "a".repeat(64);
    let response = http
        .post(format!("{base}/api/v1/submissions/status"))
        .json(&serde_json::json!({"submission_token": unknown}))
        .send()
        .await
        .expect("status request");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    let body: serde_json::Value = response.json().await.expect("status error json");
    assert_eq!(body["detail"], "unknown or expired submission token");

    // Undersized: header declares 2 records but the body is truncated
    // before the second one.
    let mut undersized = fixtures::encode_endpoint_file(&[1, 2]);
    undersized.truncate(undersized.len() - 4);
    let response = http
        .post(format!("{base}/api/v1/submissions"))
        .header("content-type", "application/vnd.netntlmv1.endpoints")
        .body(undersized)
        .send()
        .await
        .expect("submit request");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);

    // Oversized: header declares 2 records but the body has trailing
    // bytes beyond the declared record count.
    let mut oversized = fixtures::encode_endpoint_file(&[1, 2]);
    oversized.extend_from_slice(&9u64.to_le_bytes());
    let response = http
        .post(format!("{base}/api/v1/submissions"))
        .header("content-type", "application/vnd.netntlmv1.endpoints")
        .body(oversized)
        .send()
        .await
        .expect("submit request");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
}

/// Covers the brief's item 3's queue-depth case, in its own server
/// instance since it needs `Config.max_queued = 0` (a cap of `slots`
/// in-flight submissions -- 1 here).
#[tokio::test]
async fn queue_depth_exceeded_gets_503_with_retry_after_30() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = spawn_fake_backend_app(
        dir.path(),
        |config| {
            config.lookup_slots = 1;
            config.max_queued = 0;
        },
        Duration::from_secs(5),
        canned_candidate_response(),
    )
    .await;
    let http = reqwest::Client::new();
    let base = format!("http://{}", server.addr);

    let first = http
        .post(format!("{base}/api/v1/submissions"))
        .header("content-type", "application/vnd.netntlmv1.endpoints")
        .body(fixtures::encode_endpoint_file(&[1, 2]))
        .send()
        .await
        .expect("first submit");
    assert_eq!(first.status(), reqwest::StatusCode::ACCEPTED);

    // Give the worker a moment to claim the first job so the second
    // submission's queue-depth check sees it as 'running' (still counted
    // toward the cap either way).
    tokio::time::sleep(Duration::from_millis(50)).await;

    let second = http
        .post(format!("{base}/api/v1/submissions"))
        .header("content-type", "application/vnd.netntlmv1.endpoints")
        .body(fixtures::encode_endpoint_file(&[1, 2]))
        .send()
        .await
        .expect("second submit");
    assert_eq!(second.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        second
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .expect("Retry-After header"),
        "30"
    );
}

fn canned_candidate_response() -> Vec<u8> {
    ntlmrain::local_lookup::encode_candidate_file(
        2,
        &[ntlmrain::local_lookup::CandidateRecord {
            ordinal: 0,
            start: 0x0088_46f7_eaee_8fb1,
        }],
    )
    .expect("encode candidate file")
}

/// Polls `predicate` (an async closure) every 10ms until it returns
/// `true` or `timeout` elapses, panicking in the latter case.
async fn wait_until<F, Fut>(mut predicate: F, timeout: Duration)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if predicate().await {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for predicate"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
