//! CLI configuration surface for `ntlmrain-server`.
//!
//! This defines the full set of flags/env vars the service accepts. Later
//! tasks implement the behavior behind these flags; Task 1 only defines the
//! surface so downstream tasks have a stable `Config` to build against.

use std::path::PathBuf;

use clap::Parser;
use thiserror::Error;

fn default_read_workers() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

/// `ntlmrain-server` — NetNTLMv1 rainbow-table lookup service.
#[derive(Debug, Clone, Parser)]
#[command(name = "ntlmrain-server", version)]
pub struct Config {
    /// Base name of the GRTB table shards (with or without `.grtb`).
    #[arg(long = "data-base", env = "NTLMRAIN_SERVER_DATA_BASE")]
    pub data_base: PathBuf,

    /// Path to the GIDX index file.
    #[arg(long = "index", env = "NTLMRAIN_SERVER_INDEX")]
    pub index: PathBuf,

    /// Number of concurrent table-read workers.
    #[arg(
        long = "read-workers",
        env = "NTLMRAIN_SERVER_READ_WORKERS",
        default_value_t = default_read_workers()
    )]
    pub read_workers: usize,

    /// Preload the index into memory at startup.
    #[arg(long = "preload-index", env = "NTLMRAIN_SERVER_PRELOAD_INDEX")]
    pub preload_index: bool,

    /// Lock the index into memory (mlock) at startup.
    #[arg(long = "lock-index", env = "NTLMRAIN_SERVER_LOCK_INDEX")]
    pub lock_index: bool,

    /// Address the HTTP listener binds to.
    #[arg(
        long = "listen",
        env = "NTLMRAIN_SERVER_LISTEN",
        default_value = "127.0.0.1:8080"
    )]
    pub listen: String,

    /// Number of concurrent lookup slots.
    #[arg(
        long = "lookup-slots",
        env = "NTLMRAIN_SERVER_LOOKUP_SLOTS",
        default_value_t = 2
    )]
    pub lookup_slots: usize,

    /// Directory holding the SQLite job database and the `jobs/` blob tree.
    #[arg(long = "state-dir", env = "NTLMRAIN_SERVER_STATE_DIR")]
    pub state_dir: PathBuf,

    /// Maximum number of jobs that may be queued (including in-flight).
    #[arg(
        long = "max-queued",
        env = "NTLMRAIN_SERVER_MAX_QUEUED",
        default_value_t = 8
    )]
    pub max_queued: usize,

    /// Maximum number of concurrent job upload streams.
    #[arg(
        long = "max-concurrent-uploads",
        env = "NTLMRAIN_SERVER_MAX_CONCURRENT_UPLOADS",
        default_value_t = 16
    )]
    pub max_concurrent_uploads: usize,

    /// Expected exact record count for the loaded table (rejected if it
    /// doesn't match, unless `--any-record-count` is set).
    #[arg(
        long = "exact-records",
        env = "NTLMRAIN_SERVER_EXACT_RECORDS",
        default_value_t = 881_688
    )]
    pub exact_records: u64,

    /// Accept any record count for the loaded table, overriding
    /// `--exact-records`.
    #[arg(long = "any-record-count", env = "NTLMRAIN_SERVER_ANY_RECORD_COUNT")]
    pub any_record_count: bool,

    /// Maximum number of candidate match records a single lookup may return.
    #[arg(
        long = "max-match-records",
        env = "NTLMRAIN_SERVER_MAX_MATCH_RECORDS",
        default_value_t = 8_000_000
    )]
    pub max_match_records: u64,

    /// Basic auth username. Requires `--auth-password-file` to also be set.
    #[arg(long = "auth-user", env = "NTLMRAIN_SERVER_AUTH_USER")]
    pub auth_user: Option<String>,

    /// Path to a file containing the basic auth password. In practice this
    /// is normally supplied via the `NTLMRAIN_SERVER_AUTH_PASSWORD`
    /// environment variable rather than the flag; there is deliberately no
    /// plaintext `--auth-password` flag.
    #[arg(long = "auth-password-file", env = "NTLMRAIN_SERVER_AUTH_PASSWORD")]
    pub auth_password_file: Option<PathBuf>,

    /// Directory to serve static assets from (phase 2; unused in this
    /// task).
    #[arg(long = "static-dir", env = "NTLMRAIN_SERVER_STATIC_DIR")]
    pub static_dir: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error(
        "--auth-user and --auth-password-file (or NTLMRAIN_SERVER_AUTH_PASSWORD) must both be set, or neither"
    )]
    IncompleteCredentials,
}

impl Config {
    /// Validate cross-field invariants that `clap` can't express directly.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match (&self.auth_user, &self.auth_password_file) {
            (None, None) | (Some(_), Some(_)) => Ok(()),
            _ => Err(ConfigError::IncompleteCredentials),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> Config {
        Config {
            data_base: PathBuf::from("/tmp/table"),
            index: PathBuf::from("/tmp/table.gidx"),
            read_workers: 1,
            preload_index: false,
            lock_index: false,
            listen: "127.0.0.1:8080".to_string(),
            lookup_slots: 2,
            state_dir: PathBuf::from("/tmp/state"),
            max_queued: 8,
            max_concurrent_uploads: 16,
            exact_records: 881_688,
            any_record_count: false,
            max_match_records: 8_000_000,
            auth_user: None,
            auth_password_file: None,
            static_dir: None,
        }
    }

    #[test]
    fn validate_allows_no_auth() {
        assert!(base_config().validate().is_ok());
    }

    #[test]
    fn validate_allows_complete_auth() {
        let mut config = base_config();
        config.auth_user = Some("alice".to_string());
        config.auth_password_file = Some(PathBuf::from("/tmp/pw"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_user_without_password() {
        let mut config = base_config();
        config.auth_user = Some("alice".to_string());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompleteCredentials)
        ));
    }

    #[test]
    fn validate_rejects_password_without_user() {
        let mut config = base_config();
        config.auth_password_file = Some(PathBuf::from("/tmp/pw"));
        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompleteCredentials)
        ));
    }

    #[test]
    fn parses_required_flags() {
        let config = Config::parse_from([
            "ntlmrain-server",
            "--data-base",
            "/tmp/table",
            "--index",
            "/tmp/table.gidx",
            "--state-dir",
            "/tmp/state",
        ]);
        assert_eq!(config.data_base, PathBuf::from("/tmp/table"));
        assert_eq!(config.index, PathBuf::from("/tmp/table.gidx"));
        assert_eq!(config.state_dir, PathBuf::from("/tmp/state"));
        assert_eq!(config.listen, "127.0.0.1:8080");
        assert_eq!(config.lookup_slots, 2);
        assert_eq!(config.max_queued, 8);
        assert_eq!(config.max_concurrent_uploads, 16);
        assert_eq!(config.exact_records, 881_688);
        assert_eq!(config.max_match_records, 8_000_000);
        assert!(!config.any_record_count);
        assert!(!config.preload_index);
        assert!(!config.lock_index);
        assert!(config.auth_user.is_none());
        assert!(config.auth_password_file.is_none());
        assert!(config.static_dir.is_none());
    }

    #[test]
    fn missing_required_flag_fails_to_parse() {
        let result = Config::try_parse_from(["ntlmrain-server", "--index", "/tmp/table.gidx"]);
        assert!(result.is_err());
    }
}
