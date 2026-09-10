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

    /// Wall-clock time limit (in seconds) a single lookup job may run
    /// before the worker pool cancels it as a timeout. Added by Task 3
    /// (worker pool): Task 1 didn't name a field for this, so this is a
    /// small additive change to the surface it defined.
    #[arg(
        long = "job-timeout-secs",
        env = "NTLMRAIN_SERVER_JOB_TIMEOUT_SECS",
        default_value_t = 900
    )]
    pub job_timeout_secs: u64,

    /// Basic auth username. Requires exactly one of `--auth-password-file`
    /// or `NTLMRAIN_SERVER_AUTH_PASSWORD` to also be set.
    #[arg(long = "auth-user", env = "NTLMRAIN_SERVER_AUTH_USER")]
    pub auth_user: Option<String>,

    /// Path to a file containing the basic auth password.
    #[arg(
        long = "auth-password-file",
        env = "NTLMRAIN_SERVER_AUTH_PASSWORD_FILE"
    )]
    pub auth_password_file: Option<PathBuf>,

    /// The basic auth password itself. Env-only (no CLI flag or
    /// positional argument, since argv is world-readable on most
    /// platforms); set via `NTLMRAIN_SERVER_AUTH_PASSWORD`. Deliberately
    /// excluded from clap's argument parsing (`skip`) — a bare `env`
    /// attribute without `long`/`short` would make clap treat this as a
    /// positional argument instead of hiding it, which is the opposite of
    /// what we want. `read_auth_password_env` populates it after parsing.
    #[arg(skip)]
    pub auth_password: Option<String>,

    /// Directory to serve static assets from (phase 2; unused in this
    /// task).
    #[arg(long = "static-dir", env = "NTLMRAIN_SERVER_STATIC_DIR")]
    pub static_dir: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error(
        "--auth-user requires exactly one of --auth-password-file or NTLMRAIN_SERVER_AUTH_PASSWORD to be set"
    )]
    IncompleteCredentials,

    #[error(
        "--auth-password-file and NTLMRAIN_SERVER_AUTH_PASSWORD are mutually exclusive; set at most one"
    )]
    ConflictingPasswordSources,
}

const AUTH_PASSWORD_ENV: &str = "NTLMRAIN_SERVER_AUTH_PASSWORD";

impl Config {
    /// Populate `auth_password` from its env var. `auth_password` is
    /// `#[arg(skip)]`ed so clap never turns it into a CLI flag or
    /// positional argument (see its field doc comment); this is the
    /// manual equivalent of clap's `env` handling for that one field.
    /// Called by `parse()`/`try_parse_from()` below so every construction
    /// path picks it up the same way `Config::parse()` in `main.rs` does.
    fn read_auth_password_env(&mut self) {
        if self.auth_password.is_none() {
            self.auth_password = std::env::var(AUTH_PASSWORD_ENV).ok();
        }
    }

    /// Parse from `std::env::args_os()`, then fill in `auth_password` from
    /// its environment variable. Shadows (does not override)
    /// `clap::Parser::parse`.
    pub fn parse() -> Self {
        let mut config = <Self as clap::Parser>::parse();
        config.read_auth_password_env();
        config
    }

    /// Parse from an explicit argument iterator, then fill in
    /// `auth_password` from its environment variable. Shadows (does not
    /// override) `clap::Parser::parse_from`. Only used by tests today;
    /// kept `pub` since it mirrors `clap::Parser`'s surface and later
    /// tasks' tests will likely want it too.
    #[allow(dead_code)]
    pub fn parse_from<I, T>(itr: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let mut config = <Self as clap::Parser>::parse_from(itr);
        config.read_auth_password_env();
        config
    }

    /// Fallible parse from an explicit argument iterator, then fill in
    /// `auth_password` from its environment variable. Shadows (does not
    /// override) `clap::Parser::try_parse_from`. Only used by tests today;
    /// kept `pub` for the same reason as `parse_from` above.
    #[allow(dead_code)]
    pub fn try_parse_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let mut config = <Self as clap::Parser>::try_parse_from(itr)?;
        config.read_auth_password_env();
        Ok(config)
    }

    /// Validate cross-field invariants that `clap` can't express directly.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.auth_password_file.is_some() && self.auth_password.is_some() {
            return Err(ConfigError::ConflictingPasswordSources);
        }
        let password_set = self.auth_password_file.is_some() || self.auth_password.is_some();
        match (self.auth_user.is_some(), password_set) {
            (true, true) | (false, false) => Ok(()),
            _ => Err(ConfigError::IncompleteCredentials),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Guards `NTLMRAIN_SERVER_AUTH_PASSWORD`, which is process-global
    /// state. Any test that parses a `Config` (reading it, even to assert
    /// it's absent) or mutates it must hold this lock, since `cargo test`
    /// runs tests in parallel within one process.
    static AUTH_PASSWORD_ENV_LOCK: Mutex<()> = Mutex::new(());

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
            job_timeout_secs: 900,
            auth_user: None,
            auth_password_file: None,
            auth_password: None,
            static_dir: None,
        }
    }

    #[test]
    fn validate_allows_no_auth() {
        assert!(base_config().validate().is_ok());
    }

    #[test]
    fn validate_allows_complete_auth_with_password_file() {
        let mut config = base_config();
        config.auth_user = Some("alice".to_string());
        config.auth_password_file = Some(PathBuf::from("/tmp/pw"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_allows_complete_auth_with_env_password() {
        let mut config = base_config();
        config.auth_user = Some("alice".to_string());
        config.auth_password = Some("hunter2".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_user_without_any_password() {
        let mut config = base_config();
        config.auth_user = Some("alice".to_string());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompleteCredentials)
        ));
    }

    #[test]
    fn validate_rejects_password_file_without_user() {
        let mut config = base_config();
        config.auth_password_file = Some(PathBuf::from("/tmp/pw"));
        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompleteCredentials)
        ));
    }

    #[test]
    fn validate_rejects_env_password_without_user() {
        let mut config = base_config();
        config.auth_password = Some("hunter2".to_string());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompleteCredentials)
        ));
    }

    #[test]
    fn validate_rejects_both_password_sources_set() {
        let mut config = base_config();
        config.auth_user = Some("alice".to_string());
        config.auth_password_file = Some(PathBuf::from("/tmp/pw"));
        config.auth_password = Some("hunter2".to_string());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ConflictingPasswordSources)
        ));
    }

    #[test]
    fn validate_rejects_both_password_sources_set_even_without_user() {
        // The mutual-exclusion check fires regardless of auth_user, since
        // it's a config hygiene error independent of whether auth is on.
        let mut config = base_config();
        config.auth_password_file = Some(PathBuf::from("/tmp/pw"));
        config.auth_password = Some("hunter2".to_string());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ConflictingPasswordSources)
        ));
    }

    #[test]
    fn parses_required_flags() {
        let _guard = AUTH_PASSWORD_ENV_LOCK.lock().unwrap();
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
        assert_eq!(config.job_timeout_secs, 900);
        assert!(!config.any_record_count);
        assert!(!config.preload_index);
        assert!(!config.lock_index);
        assert!(config.auth_user.is_none());
        assert!(config.auth_password_file.is_none());
        assert!(config.auth_password.is_none());
        assert!(config.static_dir.is_none());
    }

    #[test]
    fn missing_required_flag_fails_to_parse() {
        let result = Config::try_parse_from(["ntlmrain-server", "--index", "/tmp/table.gidx"]);
        assert!(result.is_err());
    }

    #[test]
    fn auth_password_is_not_a_cli_argument() {
        // auth_password is #[arg(skip)]ed specifically so it can't be
        // supplied via argv (flag or positional) — only via its env var.
        // A stray positional value after the required flags should be
        // rejected as an unexpected argument, not silently captured.
        let result = Config::try_parse_from([
            "ntlmrain-server",
            "--data-base",
            "/tmp/table",
            "--index",
            "/tmp/table.gidx",
            "--state-dir",
            "/tmp/state",
            "hunter2",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn auth_password_is_populated_from_its_env_var() {
        let _guard = AUTH_PASSWORD_ENV_LOCK.lock().unwrap();
        let key = "NTLMRAIN_SERVER_AUTH_PASSWORD";
        // SAFETY: the lock above serializes this against every other test
        // in this module that parses a Config (and thus reads this var).
        unsafe {
            std::env::set_var(key, "hunter2");
        }
        let config = Config::parse_from([
            "ntlmrain-server",
            "--data-base",
            "/tmp/table",
            "--index",
            "/tmp/table.gidx",
            "--state-dir",
            "/tmp/state",
        ]);
        unsafe {
            std::env::remove_var(key);
        }
        assert_eq!(config.auth_password.as_deref(), Some("hunter2"));
    }
}
