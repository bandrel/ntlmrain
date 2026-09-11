//! The `axum::Router` for `ntlmrain-server`: the four `/api/v1/submissions*`
//! routes, health checks, and the hand-written OpenAPI document.
//!
//! Wire protocol (routes, status field names, state enum, error shape) is
//! fixed by the plan and must match it exactly -- see the field-name list
//! in this crate's Task 4 report.
//!
//! Wired into `main.rs` via `lib.rs::build_app` (Task 5), which also adds
//! a 120s read timeout on the submit route -- see `SUBMIT_READ_TIMEOUT`.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use tower::util::ServiceExt;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::timeout::TimeoutLayer;

use ntlmrain::local_lookup::{TableInfo, parse_endpoint_file};

use crate::auth::{self, AuthState};
use crate::config::Config;
use crate::queue::{InsertOutcome, SubmissionState, SubmissionStore, TokenHash};
use crate::worker::WorkerPoolHandle;

/// Pinned per the plan's "poll_within_seconds is a footgun in the
/// load-increasing direction" note: the client polls at
/// `min(2s, poll_within_seconds/2)`, so this value keeps polling at the 2s
/// cap regardless of what it's set to, as long as it stays >= 4.0.
const POLL_WITHIN_SECONDS: f64 = 4.0;
const SUBMIT_BODY_LIMIT: usize = 8 * 1024 * 1024;
/// Body limit for the three token-only JSON routes (status/result/cancel).
/// axum's own default (2 MiB) would technically also work for these small
/// bodies, but the plan pins it explicitly small (section 5) so an
/// oversized/malformed body is rejected before any JSON parsing work.
const JSON_BODY_LIMIT: usize = 4096;
const UNKNOWN_TOKEN_DETAIL: &str = "unknown or expired submission token";
/// Read timeout on `POST /api/v1/submissions` only: the up-to-8-MiB
/// submission body may arrive slowly over a poor connection, and this
/// bounds how long the server waits on it before giving up. Task 4 (which
/// built this router) deliberately left this out; added here in Task 5
/// alongside the rest of the wiring (see `build_router`'s doc comment).
/// Applied via `tower_http::timeout::TimeoutLayer` rather than a bespoke
/// deadline check so it's enforced uniformly for the whole
/// request/response cycle (including the slow-body case) the same way
/// `tower_http` enforces it elsewhere in this router (`CatchPanicLayer`).
const SUBMIT_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Central error type: every handler returns `Result<_, ApiError>` so the
/// FastAPI-shaped `{"detail": ...}` error body is constructed in exactly
/// one place.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub detail: String,
}

impl ApiError {
    fn bad_request(detail: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
        }
    }

    fn not_found() -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            detail: UNKNOWN_TOKEN_DETAIL.to_string(),
        }
    }

    fn internal(detail: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "detail": self.detail })),
        )
            .into_response()
    }
}

fn queue_full_response() -> Response {
    let mut response = ApiError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        detail: "lookup queue is full; retry later".to_string(),
    }
    .into_response();
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("30"));
    response
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_secs() as i64
}

/// Shared axum state for every handler in this module.
#[derive(Clone)]
pub struct AppState {
    config: Arc<Config>,
    store: Arc<SubmissionStore>,
    workers: WorkerPoolHandle,
    upload_semaphore: Arc<Semaphore>,
    ready: Arc<AtomicBool>,
    table_info: Option<Arc<TableInfo>>,
}

/// Build the full `axum::Router`.
///
/// `ready` gates `/health/ready`: `200` once it's `true`, `503` otherwise.
/// Task 5 flips it to `true` once `LocalTable::open` succeeds; this task's
/// own tests default it to `true` (an `Arc::new(AtomicBool::new(true))`).
///
/// `table_info` is a snapshot of `LocalTable::info()` (plan section 8:
/// "`/health/ready` reports `TableInfo`"), taken once at startup right
/// after the table opens -- `TableInfo` is `Clone` and doesn't change over
/// the table's lifetime, so no interior mutability is needed here. `None`
/// when there's no real table backing this router (e.g. this module's own
/// tests, which run against `FakeBackend`); `health_ready` reports the
/// bare `{"status": "ready"}` body in that case.
///
/// A `--static-dir`-gated fallback and a `/shaders/*` route are Phase 2
/// hooks (plan section 9) reserved but not implemented: `Config` already
/// carries `static_dir`, and no route is registered for it here.
pub fn build_router(
    config: Arc<Config>,
    store: Arc<SubmissionStore>,
    workers: WorkerPoolHandle,
    ready: Arc<AtomicBool>,
    table_info: Option<Arc<TableInfo>>,
) -> anyhow::Result<Router> {
    let auth_state = AuthState::from_config(&config)?;
    let upload_semaphore = Arc::new(Semaphore::new(config.max_concurrent_uploads));
    let app_state = AppState {
        config,
        store,
        workers,
        upload_semaphore,
        ready,
        table_info,
    };

    // Built as its own `Router` (rather than a `MethodRouter::layer` chain
    // like the other three routes below) specifically so the outermost
    // `TimeoutLayer` can be added via `Router::layer`: stacking three
    // `MethodRouter::layer` calls left axum unable to infer the
    // handler-error type at the point `TimeoutLayer` was added, since
    // `MethodRouter::layer` is generic over it and nothing pins it until
    // the route is merged into a `Router` (which is `Infallible`-only).
    let submit_route = Router::new()
        .route("/api/v1/submissions", post(submit_handler))
        .layer(DefaultBodyLimit::max(SUBMIT_BODY_LIMIT))
        // Second-outermost: the admission semaphore must be acquired
        // before the handler's `Bytes` extractor runs (and thus before
        // the up-to-8-MiB body is buffered).
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            admission_middleware,
        ))
        // Outermost layer on this route: bounds the whole
        // request/response cycle, including however long the client
        // takes to finish sending its body, at `SUBMIT_READ_TIMEOUT`. See
        // that const's doc comment.
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            SUBMIT_READ_TIMEOUT,
        ));

    let submissions_router = submit_route
        .route(
            "/api/v1/submissions/status",
            post(status_handler).layer(DefaultBodyLimit::max(JSON_BODY_LIMIT)),
        )
        .route(
            "/api/v1/submissions/result",
            post(result_handler).layer(DefaultBodyLimit::max(JSON_BODY_LIMIT)),
        )
        .route(
            "/api/v1/submissions/cancel",
            post(cancel_handler).layer(DefaultBodyLimit::max(JSON_BODY_LIMIT)),
        )
        // Wraps all four routes above; health/openapi stay unauthenticated.
        .layer(middleware::from_fn_with_state(auth_state, auth::basic_auth))
        .with_state(app_state.clone());

    let public_router = Router::new()
        .route("/health/live", get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/openapi.json", get(openapi_json))
        .route("/docs", get(docs_page))
        .with_state(app_state.clone());

    let mut router = Router::new()
        .merge(submissions_router)
        .merge(public_router)
        .layer(CatchPanicLayer::custom(handle_panic));

    // Add static file serving fallback if configured
    if let Some(path) = app_state.config.static_dir.clone() {
        router = router.fallback(move |req: axum::extract::Request| {
            let path = path.clone();
            async move {
                match ServeDir::new(path).oneshot(req).await {
                    Ok(response) => response.map(Body::new),
                    Err(_) => Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .body(Body::empty())
                        .unwrap(),
                }
            }
        });
    }

    let router = router
        // Phase 2 stub (plan section 9): the browser UI isn't mounted yet,
        // but every response already carries a minimal same-origin CSP so
        // there's nothing to retrofit once it lands.
        .layer(SetResponseHeaderLayer::overriding(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:"),
        ));

    Ok(router)
}

/// Admission gate for `POST /api/v1/submissions`: `Config.max_concurrent_uploads`
/// permits, acquired (as middleware, so before the handler's body
/// extractor runs) via `try_acquire`. Held for the duration of the
/// request; released when the response is produced.
async fn admission_middleware(State(state): State<AppState>, req: Request, next: Next) -> Response {
    match Arc::clone(&state.upload_semaphore).try_acquire_owned() {
        Ok(permit) => {
            let response = next.run(req).await;
            drop(permit);
            response
        }
        Err(_) => queue_full_response(),
    }
}

fn handle_panic(err: Box<dyn std::any::Any + Send + 'static>) -> Response<Body> {
    // Deliberately don't include the panic payload in the response body --
    // it may contain internal details -- only in the server's own stderr
    // via the default panic hook, which still fires before this runs.
    let _ = err;
    let body = serde_json::json!({ "detail": "internal error" }).to_string();
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static panic response is well-formed")
}

#[derive(Deserialize)]
struct SubmissionTokenRequest {
    submission_token: String,
}

/// Mirrors the client's own validation at `src/remote_lookup.rs:341-353`
/// (`validate_token`): exactly 64 lowercase hex characters.
fn is_valid_token(token: &str) -> bool {
    token.len() == 64
        && token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_token(token: &str) -> TokenHash {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hasher.finalize().into()
}

/// Parse `{"submission_token": "..."}` out of a request body and hash it.
/// `400` on malformed JSON or a token that isn't 64 lowercase hex chars,
/// so a malformed token never reaches the store.
fn parse_token_body(body: &[u8]) -> Result<TokenHash, ApiError> {
    let request: SubmissionTokenRequest = serde_json::from_slice(body)
        .map_err(|_| ApiError::bad_request("malformed request body"))?;
    if !is_valid_token(&request.submission_token) {
        return Err(ApiError::bad_request(
            "submission token is not 256-bit lowercase hexadecimal",
        ));
    }
    Ok(hash_token(&request.submission_token))
}

async fn submit_handler(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let parse_input = body.to_vec();
    let endpoints = tokio::task::spawn_blocking(move || parse_endpoint_file(&parse_input))
        .await
        .map_err(|error| ApiError::internal(format!("submission parse task failed: {error}")))?
        .map_err(|error| ApiError::bad_request(error.to_string()))?;

    let record_count = endpoints.len() as u64;
    if !state.config.any_record_count && record_count != state.config.exact_records {
        return Err(ApiError::bad_request(format!(
            "expected exactly {} records, got {record_count}",
            state.config.exact_records
        )));
    }

    let token_bytes: [u8; 32] = rand::random();
    let token = hex::encode(token_bytes);
    let token_sha256 = hash_token(&token);
    let hash_hex = hex::encode(token_sha256);

    let dir = state.config.state_dir.join("jobs").join(&hash_hex[0..2]);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|error| ApiError::internal(format!("failed to create job directory: {error}")))?;
    let final_path = dir.join(format!("{hash_hex}.end"));
    let tmp_path = dir.join(format!("{hash_hex}.end.tmp"));
    tokio::fs::write(&tmp_path, &body)
        .await
        .map_err(|error| ApiError::internal(format!("failed to write submission blob: {error}")))?;
    tokio::fs::rename(&tmp_path, &final_path)
        .await
        .map_err(|error| {
            ApiError::internal(format!("failed to finalize submission blob: {error}"))
        })?;

    let input_path = final_path.to_string_lossy().into_owned();
    let outcome = match state.store.insert(
        token_sha256,
        record_count,
        &input_path,
        state.config.max_queued,
        state.config.lookup_slots,
    ) {
        Ok(outcome) => outcome,
        Err(error) => {
            // The blob is already renamed into place at this point, but no
            // row exists to reference it (the insert itself failed), so it
            // would otherwise leak on disk forever -- same reasoning as the
            // `QueueFull` arm below, just for a different failure mode.
            let _ = tokio::fs::remove_file(&final_path).await;
            return Err(ApiError::internal(format!("queue store error: {error}")));
        }
    };

    match outcome {
        InsertOutcome::Inserted { .. } => Ok((
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "submission_token": token,
                "poll_within_seconds": POLL_WITHIN_SECONDS,
            })),
        )
            .into_response()),
        InsertOutcome::QueueFull => {
            let _ = tokio::fs::remove_file(&final_path).await;
            Ok(queue_full_response())
        }
    }
}

/// Status response body. Field names are pinned exactly by the plan's Wire
/// protocol table -- see this task's report for the verbatim cross-check.
#[derive(Serialize)]
struct StatusResponse {
    state: &'static str,
    record_count: u64,
    processed_records: u64,
    progress: f64,
    queue_position: Option<u64>,
    match_count: Option<u64>,
    error: Option<String>,
    poll_within_seconds: f64,
    download_within_seconds: Option<f64>,
}

async fn status_handler(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let token_sha256 = parse_token_body(&body)?;
    let row = state
        .store
        .get(token_sha256)
        .map_err(|error| ApiError::internal(format!("queue store error: {error}")))?;
    let Some(row) = row else {
        return Err(ApiError::not_found());
    };
    // Cancelled is an internal-only state -- there is no `cancelled` wire
    // state, so cancelled tokens 404 just like unknown ones.
    if row.state == SubmissionState::Cancelled {
        return Err(ApiError::not_found());
    }

    let state_str = match row.state {
        SubmissionState::Queued => "queued",
        SubmissionState::Running => "running",
        SubmissionState::Ready => "ready",
        SubmissionState::Failed => "failed",
        SubmissionState::Cancelled => unreachable!("filtered above"),
    };

    // Prefer the live control's processed() while running (fresher than
    // the ~2s-stale checkpointed column); fall back to the row otherwise.
    let processed_records = if row.state == SubmissionState::Running {
        state
            .workers
            .processed(token_sha256)
            .unwrap_or(row.processed_records as u64)
    } else {
        row.processed_records as u64
    };

    let record_count = row.record_count as u64;
    // Never divide by zero (queued jobs report literal 0.0 -- processed/0
    // would be NaN and fail the client's is_finite() check) and always
    // clamp into 0.0..=1.0: `processed_records` can legitimately exceed
    // `record_count` (e.g. a backend's own progress counter isn't
    // strictly record-count-scoped, as with `FakeBackend`'s raw tick
    // counter), and the client hard-errors on anything outside that
    // range (src/remote_lookup.rs's status invariant check).
    let progress = if record_count == 0 {
        0.0
    } else {
        (processed_records as f64 / record_count as f64).clamp(0.0, 1.0)
    };

    let queue_position = if row.state == SubmissionState::Queued {
        state
            .store
            .queue_position(token_sha256)
            .map_err(|error| ApiError::internal(format!("queue store error: {error}")))?
    } else {
        None
    };

    let match_count = if row.state == SubmissionState::Ready {
        row.match_count.map(|value| value as u64)
    } else {
        None
    };

    let error = if row.state == SubmissionState::Failed {
        row.error.clone()
    } else {
        None
    };

    let download_within_seconds = if row.state == SubmissionState::Ready {
        row.result_expires_at
            .map(|expires_at| (expires_at - now_unix()).max(0) as f64)
    } else {
        None
    };

    Ok(Json(StatusResponse {
        state: state_str,
        record_count,
        processed_records,
        progress,
        queue_position,
        match_count,
        error,
        poll_within_seconds: POLL_WITHIN_SECONDS,
        download_within_seconds,
    })
    .into_response())
}

async fn result_handler(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let token_sha256 = parse_token_body(&body)?;
    let row = state
        .store
        .get(token_sha256)
        .map_err(|error| ApiError::internal(format!("queue store error: {error}")))?;
    let Some(row) = row else {
        return Err(ApiError::not_found());
    };
    if row.state != SubmissionState::Ready {
        return Err(ApiError::not_found());
    }
    let result_path = row
        .result_path
        .ok_or_else(|| ApiError::internal("ready submission is missing its result path"))?;

    // Read the whole result into memory rather than streaming it: results
    // are bounded by `--max-match-records` (default 8,000,000 records =~
    // 128 MiB), so this is both simple and acceptable, per the task brief.
    let bytes = tokio::fs::read(&result_path)
        .await
        .map_err(|error| ApiError::internal(format!("failed to read result blob: {error}")))?;
    let content_length = bytes.len();

    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = StatusCode::OK;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/vnd.netntlmv1.candidates"),
    );
    headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&content_length.to_string())
            .expect("a decimal length is always a valid header value"),
    );
    Ok(response)
}

async fn cancel_handler(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let token_sha256 = parse_token_body(&body)?;
    // Idempotent regardless of whether the token is known/still
    // cancellable, per the "no oracle" rule: always 204.
    if let Some(row) = state
        .store
        .get(token_sha256)
        .map_err(|error| ApiError::internal(format!("queue store error: {error}")))?
        && matches!(
            row.state,
            SubmissionState::Queued | SubmissionState::Running
        )
    {
        // Tell a live worker (if any is actively running this job) to
        // stop; also flip the store row directly so a merely-queued job
        // (no live worker yet) is reflected as cancelled immediately
        // instead of only once/if a worker eventually claims it.
        state.workers.cancel(token_sha256);
        let _ = state.store.mark_cancelled(token_sha256);
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn health_live() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({})))
}

async fn health_ready(State(state): State<AppState>) -> Response {
    if state.ready.load(Ordering::Acquire) {
        // Plan section 8: "/health/ready reports TableInfo (records,
        // blocks, parts, min/max endpoint)". `table_info` is `None` only
        // when there's no real table backing this router (this module's
        // own tests); production always has one by the time `ready` is
        // ever `true` (see `lib.rs::build_app`, which opens the table
        // before constructing the router at all).
        let mut body = serde_json::json!({ "status": "ready" });
        if let Some(info) = &state.table_info {
            body["records"] = serde_json::json!(info.records);
            body["blocks"] = serde_json::json!(info.blocks);
            body["parts"] = serde_json::json!(info.parts);
            body["min_endpoint"] = serde_json::json!(info.min_endpoint);
            body["max_endpoint"] = serde_json::json!(info.max_endpoint);
        }
        (StatusCode::OK, Json(body)).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "not ready" })),
        )
            .into_response()
    }
}

/// Minimal static page pointing at `/openapi.json`, for parity with the
/// public service's `/docs` route (plan's Wire protocol table). The real
/// CLI client never requests this; a full Swagger-UI-style page is out of
/// scope (no new dependency), so this is a hand-written link.
async fn docs_page() -> impl IntoResponse {
    let html = concat!(
        "<!doctype html><html><head><title>ntlmrain-server API docs</title></head>",
        "<body><h1>ntlmrain-server</h1>",
        "<p>See the OpenAPI document: <a href=\"/openapi.json\">/openapi.json</a></p>",
        "</body></html>",
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
}

async fn openapi_json() -> impl IntoResponse {
    Json(serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "ntlmrain-server",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "NetNTLMv1 rainbow-table lookup service."
        },
        "paths": {
            "/api/v1/submissions": {
                "post": {
                    "summary": "Submit an NTLMEND1 endpoint file for lookup.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/vnd.netntlmv1.endpoints": {
                                "schema": { "type": "string", "format": "binary" }
                            }
                        }
                    },
                    "responses": {
                        "202": {
                            "description": "Submission accepted.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "submission_token": { "type": "string" },
                                            "poll_within_seconds": { "type": "number" }
                                        }
                                    }
                                }
                            }
                        },
                        "400": { "description": "Malformed or invalid submission." },
                        "503": { "description": "Lookup queue is full." }
                    }
                }
            },
            "/api/v1/submissions/status": {
                "post": {
                    "summary": "Poll a submission's status.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": { "submission_token": { "type": "string" } }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Status.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "state": { "type": "string", "enum": ["queued", "running", "ready", "failed"] },
                                            "record_count": { "type": "integer" },
                                            "processed_records": { "type": "integer" },
                                            "progress": { "type": "number" },
                                            "queue_position": { "type": "integer", "nullable": true },
                                            "match_count": { "type": "integer", "nullable": true },
                                            "error": { "type": "string", "nullable": true },
                                            "poll_within_seconds": { "type": "number" },
                                            "download_within_seconds": { "type": "number", "nullable": true }
                                        }
                                    }
                                }
                            }
                        },
                        "404": { "description": "Unknown or expired submission token." }
                    }
                }
            },
            "/api/v1/submissions/result": {
                "post": {
                    "summary": "Download a ready submission's NTLMCAN1 result.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": { "submission_token": { "type": "string" } }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Result.",
                            "content": {
                                "application/vnd.netntlmv1.candidates": {
                                    "schema": { "type": "string", "format": "binary" }
                                }
                            }
                        },
                        "404": { "description": "Unknown, expired, or not-yet-ready submission token." }
                    }
                }
            },
            "/api/v1/submissions/cancel": {
                "post": {
                    "summary": "Cancel a submission.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": { "submission_token": { "type": "string" } }
                                }
                            }
                        }
                    },
                    "responses": {
                        "204": { "description": "Cancelled (idempotent; no oracle on token existence)." }
                    }
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{FakeBackend, LookupBackend};
    use crate::worker::WorkerPool;
    use axum::body::to_bytes;
    use axum::http::{HeaderMap, Request as HttpRequest};
    use ntlmrain::local_lookup::{CandidateRecord, encode_candidate_file};
    use std::time::Duration;
    use tower::ServiceExt;

    fn base_config(state_dir: std::path::PathBuf) -> Config {
        Config {
            data_base: std::path::PathBuf::from("/tmp/table"),
            index: std::path::PathBuf::from("/tmp/table.gidx"),
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

    fn encode_endpoint_file(endpoints: &[u64]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + endpoints.len() * 8);
        buf.extend_from_slice(b"NTLMEND1");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&8u32.to_le_bytes()); // record size
        buf.extend_from_slice(&(endpoints.len() as u64).to_le_bytes()); // count
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        for endpoint in endpoints {
            buf.extend_from_slice(&endpoint.to_le_bytes());
        }
        buf
    }

    fn canned_response() -> Vec<u8> {
        encode_candidate_file(
            2,
            &[CandidateRecord {
                ordinal: 0,
                start: 0x0088_46f7_eaee_8fb1,
            }],
        )
        .expect("encode candidate file")
    }

    fn sample_table_info() -> TableInfo {
        TableInfo {
            records: 881_688,
            blocks: 4096,
            data_bytes: 12_345,
            index_bytes: 6_789,
            index_locked: false,
            records_per_part: 1024,
            parts: 861,
            start_bits: 20,
            rice_k: 16,
            min_endpoint: 0x0000_0000_0000_0001,
            max_endpoint: 0xffff_ffff_ffff_fffe,
        }
    }

    /// Test harness: a real `SubmissionStore` + real `WorkerPool` (running
    /// `FakeBackend`) + the real router, all in a tempdir. `sleep` controls
    /// how long the fake lookup takes, so tests can observe
    /// queued/running/ready transitions.
    struct Harness {
        _dir: tempfile::TempDir,
        router: Router,
        _pool: WorkerPool,
    }

    fn harness_with(sleep: Duration, config_mutate: impl FnOnce(&mut Config)) -> Harness {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = base_config(dir.path().to_path_buf());
        config_mutate(&mut config);
        let config = Arc::new(config);

        let store =
            Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).expect("open store"));
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            sleep,
            Duration::from_millis(5),
            canned_response(),
        ));
        let pool = WorkerPool::spawn(
            config.lookup_slots,
            Duration::from_secs(config.job_timeout_secs),
            config.max_match_records,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );
        let ready = Arc::new(AtomicBool::new(true));
        let table_info = Some(Arc::new(sample_table_info()));
        let router = build_router(Arc::clone(&config), store, pool.handle(), ready, table_info)
            .expect("build router");

        Harness {
            _dir: dir,
            router,
            _pool: pool,
        }
    }

    fn harness() -> Harness {
        harness_with(Duration::from_millis(50), |_| {})
    }

    async fn call(
        router: &Router,
        method: &str,
        path: &str,
        body: Vec<u8>,
        headers: Vec<(&str, &str)>,
    ) -> Response {
        let mut builder = HttpRequest::builder().method(method).uri(path);
        for (name, value) in headers {
            builder = builder.header(name, value);
        }
        let request = builder.body(Body::from(body)).unwrap();
        router.clone().oneshot(request).await.unwrap()
    }

    async fn call_json(router: &Router, path: &str, body: serde_json::Value) -> Response {
        call(
            router,
            "POST",
            path,
            serde_json::to_vec(&body).unwrap(),
            vec![("content-type", "application/json")],
        )
        .await
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn headers_of(response: &Response) -> HeaderMap {
        response.headers().clone()
    }

    /// Asserts the client's hard invariant (`src/remote_lookup.rs`):
    /// `progress` must be finite and within `0.0..=1.0`. Called on every
    /// sampled status body during a polling loop, not just the terminal
    /// one, since the live-control path (used while `state=='running'`)
    /// and the checkpointed-column path (used otherwise) are both
    /// reachable and both must uphold it.
    fn assert_progress_invariant(body: &serde_json::Value) {
        let progress = body["progress"]
            .as_f64()
            .expect("progress field must be present and numeric");
        assert!(
            progress.is_finite() && (0.0..=1.0).contains(&progress),
            "progress {progress} outside 0.0..=1.0 (state={:?})",
            body["state"]
        );
    }

    #[tokio::test]
    async fn submit_then_status_transitions_queued_running_ready_with_progress_and_match_count() {
        let harness = harness_with(Duration::from_millis(150), |_| {});
        let endpoint_file = encode_endpoint_file(&[1, 2]);

        let submit_response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            endpoint_file,
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(submit_response.status(), StatusCode::ACCEPTED);
        let submit_body = body_json(submit_response).await;
        let token = submit_body["submission_token"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(token.len(), 64);
        assert_eq!(
            submit_body["poll_within_seconds"].as_f64().unwrap(),
            POLL_WITHIN_SECONDS
        );

        // First poll: should observe 'queued' or 'running' (racy but the
        // FakeBackend sleeps 150ms, so a poll shortly after submit is very
        // likely 'queued' -- either is acceptable, both are pre-'ready'.
        let status_response = call_json(
            &harness.router,
            "/api/v1/submissions/status",
            serde_json::json!({"submission_token": token}),
        )
        .await;
        assert_eq!(status_response.status(), StatusCode::OK);
        let status_body = body_json(status_response).await;
        let first_state = status_body["state"].as_str().unwrap().to_string();
        assert!(matches!(first_state.as_str(), "queued" | "running"));
        assert_progress_invariant(&status_body);
        assert!(status_body["match_count"].is_null());
        assert!(status_body["error"].is_null());

        // Poll until ready, watching for a 'running' sample with
        // processed_records > 0 along the way -- this only happens if the
        // status handler is preferring the worker pool's live control
        // (ticked every 5ms by FakeBackend) over the row's checkpointed
        // column (only updated every ~2s by the heartbeat, far slower
        // than this 150ms job).
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut final_body = status_body;
        let mut observed_live_progress = false;
        loop {
            assert_progress_invariant(&final_body);
            if final_body["state"] == "ready" {
                break;
            }
            if final_body["state"] == "running"
                && final_body["processed_records"].as_u64().unwrap_or(0) > 0
            {
                observed_live_progress = true;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for ready"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
            let response = call_json(
                &harness.router,
                "/api/v1/submissions/status",
                serde_json::json!({"submission_token": token}),
            )
            .await;
            final_body = body_json(response).await;
        }
        assert!(
            observed_live_progress,
            "expected to observe a 'running' status with processed_records > 0 \
             (proves the live worker-pool control is preferred over the \
             checkpointed column while running)"
        );

        assert_eq!(final_body["state"], "ready");
        assert_eq!(final_body["record_count"], 2);
        assert_eq!(final_body["match_count"], 1);
        // `progress` is `processed_records / record_count` from the row's
        // *checkpointed* column once a job leaves 'running' (the live
        // control is gone from the worker pool's map by then) -- the
        // worker's ~2s heartbeat interval (worker.rs's HEARTBEAT_INTERVAL)
        // is far longer than this test's fake lookup, so the checkpoint
        // may never have run before the job finished. The polling loop
        // above already asserted the wire invariant (finite, in
        // 0.0..=1.0 per src/remote_lookup.rs's client-side check) on
        // every sample including this one; not asserting a specific
        // value here since it depends on Task 3's heartbeat timing.
        assert!(final_body["queue_position"].is_null());
        assert!(final_body["download_within_seconds"].as_f64().unwrap() > 0.0);

        // Result download.
        let result_response = call_json(
            &harness.router,
            "/api/v1/submissions/result",
            serde_json::json!({"submission_token": token}),
        )
        .await;
        assert_eq!(result_response.status(), StatusCode::OK);
        let headers = headers_of(&result_response);
        assert_eq!(
            headers.get(header::CONTENT_TYPE).unwrap(),
            "application/vnd.netntlmv1.candidates"
        );
        let expected = canned_response();
        assert_eq!(
            headers.get(header::CONTENT_LENGTH).unwrap(),
            &expected.len().to_string()
        );
        let bytes = to_bytes(result_response.into_body(), 64 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), expected.as_slice());
    }

    #[tokio::test]
    async fn unknown_token_on_status_result_cancel_behaves_per_the_no_oracle_rule() {
        let harness = harness();
        let unknown = "a".repeat(64);

        let status_response = call_json(
            &harness.router,
            "/api/v1/submissions/status",
            serde_json::json!({"submission_token": unknown}),
        )
        .await;
        assert_eq!(status_response.status(), StatusCode::NOT_FOUND);
        let detail = body_json(status_response).await;
        assert_eq!(detail["detail"], UNKNOWN_TOKEN_DETAIL);

        let result_response = call_json(
            &harness.router,
            "/api/v1/submissions/result",
            serde_json::json!({"submission_token": unknown}),
        )
        .await;
        assert_eq!(result_response.status(), StatusCode::NOT_FOUND);

        let cancel_response = call_json(
            &harness.router,
            "/api/v1/submissions/cancel",
            serde_json::json!({"submission_token": unknown}),
        )
        .await;
        assert_eq!(cancel_response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn malformed_token_is_rejected_before_reaching_the_store() {
        let harness = harness();
        let response = call_json(
            &harness.router,
            "/api/v1/submissions/status",
            serde_json::json!({"submission_token": "not-hex"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn submission_over_the_exact_records_limit_gets_400() {
        let harness = harness();
        // exact_records is 2 in base_config; submit 3.
        let endpoint_file = encode_endpoint_file(&[1, 2, 3]);
        let response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            endpoint_file,
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let detail = body_json(response).await;
        assert!(
            detail["detail"]
                .as_str()
                .unwrap()
                .contains("expected exactly")
        );
    }

    #[tokio::test]
    async fn malformed_submission_body_gets_400() {
        let harness = harness();
        let response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            b"not an endpoint file".to_vec(),
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn queue_depth_exceeded_gets_503_with_retry_after() {
        // slots=1, max_queued=0 -> cap of 1 in-flight submission.
        let harness = harness_with(Duration::from_secs(5), |config| {
            config.lookup_slots = 1;
            config.max_queued = 0;
        });
        let endpoint_file = encode_endpoint_file(&[1, 2]);

        let first = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            endpoint_file.clone(),
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(first.status(), StatusCode::ACCEPTED);

        // Give the worker a moment to claim the first job so the second
        // submission's queue-depth check sees it as 'running' (still
        // counted toward the cap either way).
        tokio::time::sleep(Duration::from_millis(20)).await;

        let second = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            endpoint_file,
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
        let headers = headers_of(&second);
        assert_eq!(headers.get(header::RETRY_AFTER).unwrap(), "30");
        let detail = body_json(second).await;
        assert_eq!(detail["detail"], "lookup queue is full; retry later");
    }

    #[tokio::test]
    async fn health_live_and_ready_behave_per_the_ready_flag() {
        let harness = harness();
        let live = call(&harness.router, "GET", "/health/live", Vec::new(), vec![]).await;
        assert_eq!(live.status(), StatusCode::OK);

        let ready = call(&harness.router, "GET", "/health/ready", Vec::new(), vec![]).await;
        assert_eq!(ready.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_ready_reports_503_when_the_ready_flag_is_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = Arc::new(base_config(dir.path().to_path_buf()));
        let store =
            Arc::new(SubmissionStore::open(dir.path().join("queue.sqlite3")).expect("open store"));
        let backend: Arc<dyn LookupBackend> = Arc::new(FakeBackend::new(
            Duration::from_millis(10),
            Duration::from_millis(5),
            canned_response(),
        ));
        let pool = WorkerPool::spawn(
            1,
            Duration::from_secs(60),
            1_000_000,
            dir.path().to_path_buf(),
            Arc::clone(&store),
            backend,
        );
        let ready = Arc::new(AtomicBool::new(false));
        let router = build_router(config, store, pool.handle(), ready, None).unwrap();

        let response = call(&router, "GET", "/health/ready", Vec::new(), vec![]).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn health_ready_reports_table_info_fields_when_ready() {
        let harness = harness();
        let response = call(&harness.router, "GET", "/health/ready", Vec::new(), vec![]).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let info = sample_table_info();
        assert_eq!(body["status"], "ready");
        assert_eq!(body["records"], info.records);
        assert_eq!(body["blocks"], info.blocks);
        assert_eq!(body["parts"], info.parts);
        assert_eq!(body["min_endpoint"], info.min_endpoint);
        assert_eq!(body["max_endpoint"], info.max_endpoint);
    }

    #[tokio::test]
    async fn anonymous_request_to_submit_route_gets_401_with_www_authenticate() {
        // Regression test for finding #6: every other auth test hits
        // /status or a successful authenticated submit -- nothing
        // previously proved an anonymous request against the *submit*
        // route specifically was covered by the router-layer auth
        // middleware (the plan's whole reason for using a router-layer
        // middleware over a per-handler extractor: it's one forgotten
        // handler away from a bypass).
        let harness = harness_with(Duration::from_millis(50), |config| {
            config.auth_user = Some("alice".to_string());
            config.auth_password = Some("hunter2".to_string());
        });
        let response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            b"not a real endpoint file".to_vec(),
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let headers = headers_of(&response);
        assert_eq!(
            headers.get(header::WWW_AUTHENTICATE).unwrap(),
            "Basic realm=\"ntlmrain\""
        );
        let detail = body_json(response).await;
        assert_eq!(
            detail["detail"],
            "authentication required; set --remote-username/--remote-password"
        );
    }

    #[tokio::test]
    async fn openapi_json_returns_valid_json_describing_the_submission_routes() {
        let harness = harness();
        let response = call(&harness.router, "GET", "/openapi.json", Vec::new(), vec![]).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert!(body["paths"]["/api/v1/submissions"].is_object());
        assert!(body["paths"]["/api/v1/submissions/status"].is_object());
        assert!(body["paths"]["/api/v1/submissions/result"].is_object());
        assert!(body["paths"]["/api/v1/submissions/cancel"].is_object());
    }

    #[tokio::test]
    async fn docs_page_returns_html_linking_to_openapi_json() {
        let harness = harness();
        let response = call(&harness.router, "GET", "/docs", Vec::new(), vec![]).await;
        assert_eq!(response.status(), StatusCode::OK);
        let headers = headers_of(&response);
        assert!(
            headers
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("/openapi.json"));
    }

    fn harness_with_auth(user: &str, password: &str) -> Harness {
        harness_with(Duration::from_millis(50), |config| {
            config.auth_user = Some(user.to_string());
            config.auth_password = Some(password.to_string());
        })
    }

    #[tokio::test]
    async fn anonymous_request_against_an_auth_configured_router_gets_401_with_www_authenticate() {
        let harness = harness_with_auth("alice", "hunter2");
        let response = call_json(
            &harness.router,
            "/api/v1/submissions/status",
            serde_json::json!({"submission_token": "a".repeat(64)}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let headers = headers_of(&response);
        assert_eq!(
            headers.get(header::WWW_AUTHENTICATE).unwrap(),
            "Basic realm=\"ntlmrain\""
        );
        let detail = body_json(response).await;
        assert_eq!(
            detail["detail"],
            "authentication required; set --remote-username/--remote-password"
        );
    }

    #[tokio::test]
    async fn correct_basic_credentials_succeed() {
        let harness = harness_with_auth("alice", "hunter2");
        let credential =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, b"alice:hunter2");
        let response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions/status",
            serde_json::to_vec(&serde_json::json!({"submission_token": "a".repeat(64)})).unwrap(),
            vec![
                ("content-type", "application/json"),
                ("authorization", &format!("Basic {credential}")),
            ],
        )
        .await;
        // Auth passes; token is unknown, so 404 (not 401) proves the
        // request reached the handler.
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn wrong_credentials_get_401() {
        let harness = harness_with_auth("alice", "hunter2");
        let credential = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            b"alice:wrongpassword",
        );
        let response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions/status",
            serde_json::to_vec(&serde_json::json!({"submission_token": "a".repeat(64)})).unwrap(),
            vec![
                ("content-type", "application/json"),
                ("authorization", &format!("Basic {credential}")),
            ],
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn health_and_openapi_stay_unauthenticated_when_auth_is_configured() {
        let harness = harness_with_auth("alice", "hunter2");
        let live = call(&harness.router, "GET", "/health/live", Vec::new(), vec![]).await;
        assert_eq!(live.status(), StatusCode::OK);
        let openapi = call(&harness.router, "GET", "/openapi.json", Vec::new(), vec![]).await;
        assert_eq!(openapi.status(), StatusCode::OK);
        let docs = call(&harness.router, "GET", "/docs", Vec::new(), vec![]).await;
        assert_eq!(docs.status(), StatusCode::OK);
    }

    #[test]
    fn is_valid_token_matches_the_clients_validation() {
        assert!(is_valid_token(&"a".repeat(64)));
        assert!(is_valid_token(&"0123456789abcdef".repeat(4)));
        assert!(!is_valid_token(&"A".repeat(64))); // uppercase rejected
        assert!(!is_valid_token(&"a".repeat(63))); // too short
        assert!(!is_valid_token(&"g".repeat(64))); // out of hex range
    }

    #[tokio::test]
    async fn static_dir_serves_files_at_root_when_configured() {
        let static_dir = tempfile::tempdir().expect("tempdir for static files");
        let index_path = static_dir.path().join("index.html");
        let test_content = b"<!doctype html><html><body>Hello, World!</body></html>";
        tokio::fs::write(&index_path, test_content)
            .await
            .expect("write index.html");

        let harness = harness_with(Duration::from_millis(50), |config| {
            config.static_dir = Some(static_dir.path().to_path_buf());
        });

        // Test GET /
        let root_response = call(&harness.router, "GET", "/", Vec::new(), vec![]).await;
        assert_eq!(root_response.status(), StatusCode::OK);
        let root_bytes = to_bytes(root_response.into_body(), 64 * 1024)
            .await
            .unwrap();
        assert_eq!(root_bytes.as_ref(), test_content);

        // Test GET /index.html
        let index_response = call(&harness.router, "GET", "/index.html", Vec::new(), vec![]).await;
        assert_eq!(index_response.status(), StatusCode::OK);
        let index_bytes = to_bytes(index_response.into_body(), 64 * 1024)
            .await
            .unwrap();
        assert_eq!(index_bytes.as_ref(), test_content);
    }

    #[tokio::test]
    async fn static_dir_none_does_not_affect_existing_routes() {
        let harness = harness();

        // All existing routes should work exactly as before
        // Health checks
        let live = call(&harness.router, "GET", "/health/live", Vec::new(), vec![]).await;
        assert_eq!(live.status(), StatusCode::OK);

        let ready = call(&harness.router, "GET", "/health/ready", Vec::new(), vec![]).await;
        assert_eq!(ready.status(), StatusCode::OK);

        // OpenAPI and docs
        let openapi = call(&harness.router, "GET", "/openapi.json", Vec::new(), vec![]).await;
        assert_eq!(openapi.status(), StatusCode::OK);

        let docs = call(&harness.router, "GET", "/docs", Vec::new(), vec![]).await;
        assert_eq!(docs.status(), StatusCode::OK);

        // API routes work as expected (submission test)
        let endpoint_file = encode_endpoint_file(&[1, 2]);
        let submit_response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            endpoint_file,
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(submit_response.status(), StatusCode::ACCEPTED);
    }

    #[tokio::test]
    async fn static_files_do_not_shadow_api_routes() {
        let static_dir = tempfile::tempdir().expect("tempdir for static files");

        // Create files that could shadow API routes if the fallback is applied incorrectly
        let api_path = static_dir.path().join("api");
        tokio::fs::create_dir(&api_path).await.expect("create api dir");
        let shadowing_file = api_path.join("v1");
        tokio::fs::create_dir(&shadowing_file)
            .await
            .expect("create api/v1 dir");

        let harness = harness_with(Duration::from_millis(50), |config| {
            config.static_dir = Some(static_dir.path().to_path_buf());
        });

        // API routes must still work and not be shadowed by the static files
        let endpoint_file = encode_endpoint_file(&[1, 2]);
        let submit_response = call(
            &harness.router,
            "POST",
            "/api/v1/submissions",
            endpoint_file.clone(),
            vec![("content-type", "application/vnd.netntlmv1.endpoints")],
        )
        .await;
        assert_eq!(submit_response.status(), StatusCode::ACCEPTED);

        // Health checks must not be shadowed
        let live = call(&harness.router, "GET", "/health/live", Vec::new(), vec![]).await;
        assert_eq!(live.status(), StatusCode::OK);

        let ready = call(&harness.router, "GET", "/health/ready", Vec::new(), vec![]).await;
        assert_eq!(ready.status(), StatusCode::OK);

        // OpenAPI and docs must not be shadowed
        let openapi = call(&harness.router, "GET", "/openapi.json", Vec::new(), vec![]).await;
        assert_eq!(openapi.status(), StatusCode::OK);

        let docs = call(&harness.router, "GET", "/docs", Vec::new(), vec![]).await;
        assert_eq!(docs.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn csp_header_is_present_on_api_responses() {
        let harness = harness();
        let response = call(&harness.router, "GET", "/health/live", Vec::new(), vec![]).await;
        assert_eq!(response.status(), StatusCode::OK);
        let headers = headers_of(&response);
        let csp = headers
            .get(header::CONTENT_SECURITY_POLICY)
            .expect("CSP header must be present")
            .to_str()
            .expect("CSP header must be valid UTF-8");
        assert!(csp.contains("default-src 'self'"));
        assert!(csp.contains("script-src 'self' 'wasm-unsafe-eval'"));
        assert!(csp.contains("worker-src 'self' blob:"));
    }

    #[tokio::test]
    async fn csp_header_is_present_on_static_file_responses() {
        let static_dir = tempfile::tempdir().expect("tempdir for static files");
        let index_path = static_dir.path().join("index.html");
        tokio::fs::write(&index_path, b"<html></html>")
            .await
            .expect("write index.html");

        let harness = harness_with(Duration::from_millis(50), |config| {
            config.static_dir = Some(static_dir.path().to_path_buf());
        });

        let response = call(&harness.router, "GET", "/index.html", Vec::new(), vec![]).await;
        assert_eq!(response.status(), StatusCode::OK);
        let headers = headers_of(&response);
        let csp = headers
            .get(header::CONTENT_SECURITY_POLICY)
            .expect("CSP header must be present on static files")
            .to_str()
            .expect("CSP header must be valid UTF-8");
        assert!(csp.contains("default-src 'self'"));
        assert!(csp.contains("script-src 'self' 'wasm-unsafe-eval'"));
        assert!(csp.contains("worker-src 'self' blob:"));
    }
}
