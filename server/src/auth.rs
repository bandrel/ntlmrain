//! Optional HTTP Basic auth middleware for the `/api/v1` sub-router.
//!
//! Wraps the whole `/api/v1` router (not a per-handler extractor — see the
//! plan's Auth section: a per-handler check is one forgotten handler away
//! from a bypass). Pass-through (no header ever checked, no
//! `WWW-Authenticate` ever emitted) when no credentials are configured;
//! otherwise every request must present valid HTTP Basic credentials.
//!
//! Wired into `main.rs` via `lib.rs::build_app` -> `http::build_router`
//! (Task 5).
#![allow(dead_code)]

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::http::ApiError;

/// Resolved Basic-auth credentials (username + plaintext password), or
/// `None` if auth is disabled. Built once at router-construction time from
/// `Config` (which only carries an `Option<PathBuf>`/`Option<String>` for
/// the password source).
#[derive(Clone)]
pub struct AuthState {
    credentials: Option<Arc<(String, String)>>,
}

impl AuthState {
    /// Pass-through auth state (no credentials configured). Used by
    /// callers that don't want the middleware to check anything.
    pub fn disabled() -> Self {
        AuthState { credentials: None }
    }

    /// Resolve the configured username/password from `Config`, reading
    /// `auth_password_file` from disk if that's the configured password
    /// source. `Config::validate` already guarantees at most one of
    /// `auth_password_file`/`auth_password` is set alongside `auth_user`;
    /// this only has to handle actually reading the file.
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let Some(user) = config.auth_user.clone() else {
            return Ok(AuthState::disabled());
        };
        let password = if let Some(path) = &config.auth_password_file {
            std::fs::read_to_string(path)
                .map_err(|error| {
                    anyhow::anyhow!("failed to read --auth-password-file {path:?}: {error}")
                })?
                .trim_end_matches(['\n', '\r'])
                .to_string()
        } else if let Some(password) = &config.auth_password {
            password.clone()
        } else {
            // Config::validate() rejects auth_user set with no password
            // source before this ever runs.
            return Err(anyhow::anyhow!(
                "--auth-user is set but no password source is configured"
            ));
        };
        Ok(AuthState {
            credentials: Some(Arc::new((user, password))),
        })
    }
}

/// SHA-256 of `user\0pass`, matching the plan's "compare using SHA-256 of
/// `user\0pass` on both sides folded with XOR" spec.
fn credential_hash(user: &str, pass: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(user.as_bytes());
    hasher.update(b"\0");
    hasher.update(pass.as_bytes());
    hasher.finalize().into()
}

/// Constant-time-*shaped* comparison of two 32-byte hashes: every byte
/// pair is XORed and folded into one accumulator regardless of where (or
/// whether) a mismatch occurs, so the number of operations executed does
/// not depend on the position of the first differing byte the way a
/// short-circuiting `==` would. This intentionally avoids pulling in
/// `subtle`/`constant_time_eq` per the plan, since `sha2` is already a
/// dependency and the fixed 32-byte comparison is small enough to write
/// by hand.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

fn unauthorized() -> Response {
    let mut response = ApiError {
        status: StatusCode::UNAUTHORIZED,
        detail: "authentication required; set --remote-username/--remote-password".to_string(),
    }
    .into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Basic realm=\"ntlmrain\""),
    );
    response
}

/// Parse an `Authorization: Basic <base64>` header value into
/// `(user, pass)`, splitting the decoded `user:pass` on the first `:`.
fn parse_basic_auth(header_value: &HeaderValue) -> Option<(String, String)> {
    let text = header_value.to_str().ok()?;
    let encoded = text.strip_prefix("Basic ")?;
    let decoded = BASE64.decode(encoded).ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let (user, pass) = decoded.split_once(':')?;
    Some((user.to_string(), pass.to_string()))
}

/// `middleware::from_fn_with_state` handler wrapping the `/api/v1`
/// sub-router. Pure pass-through (no header check, no `WWW-Authenticate`
/// ever emitted) when `state.credentials` is `None`.
pub async fn basic_auth(State(state): State<AuthState>, req: Request, next: Next) -> Response {
    let Some(configured) = &state.credentials else {
        return next.run(req).await;
    };

    let supplied = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(parse_basic_auth);

    let Some((user, pass)) = supplied else {
        return unauthorized();
    };

    let expected_hash = credential_hash(&configured.0, &configured.1);
    let got_hash = credential_hash(&user, &pass);
    if constant_time_eq(&expected_hash, &got_hash) {
        next.run(req).await
    } else {
        unauthorized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_identical_hashes() {
        let hash = credential_hash("alice", "hunter2");
        assert!(constant_time_eq(&hash, &hash));
    }

    #[test]
    fn constant_time_eq_rejects_any_mismatch_position() {
        // Prove the comparison's correctness is independent of *where* the
        // mismatch occurs (first byte, middle byte, last byte) -- each of
        // these takes the identical code path (the full 32-iteration fold),
        // which is the property the plan asks us to document/test rather
        // than a wall-clock timing measurement.
        let base = credential_hash("alice", "hunter2");
        for position in [0usize, 15, 31] {
            let mut other = base;
            other[position] ^= 0x01;
            assert!(
                !constant_time_eq(&base, &other),
                "mismatch at byte {position} should be detected"
            );
        }
    }

    #[test]
    fn credential_hash_is_sensitive_to_the_separator() {
        // "ab" + "\0" + "c" must not collide with "a" + "\0" + "bc": the
        // null-byte separator (not naive string concatenation) is what
        // makes the hash unambiguous.
        assert_ne!(credential_hash("ab", "c"), credential_hash("a", "bc"));
    }

    #[test]
    fn parse_basic_auth_splits_on_first_colon_only() {
        let value = HeaderValue::from_str(&format!(
            "Basic {}",
            BASE64.encode(b"user:pass:with:colons")
        ))
        .unwrap();
        let (user, pass) = parse_basic_auth(&value).expect("parses");
        assert_eq!(user, "user");
        assert_eq!(pass, "pass:with:colons");
    }

    #[test]
    fn parse_basic_auth_rejects_non_basic_scheme() {
        let value = HeaderValue::from_static("Bearer sometoken");
        assert!(parse_basic_auth(&value).is_none());
    }
}
