//! `ntlmrain-server` binary entry point.
//!
//! # Building for production
//!
//! Build with the `release-server` profile (defined in the workspace root
//! `Cargo.toml`: `inherits = "release"`, `panic = "unwind"` -- this crate
//! needs unwinding so `worker.rs`'s `catch_unwind` around a backend panic
//! can actually catch it, unlike the root `ntlmrain` CLI's own
//! `release` profile, which uses `panic = "abort"`):
//!
//! ```sh
//! cargo build --profile release-server -p ntlmrain-server
//! ```

// Build-time guard against building this crate with `panic = "abort"`.
//
// Cargo profiles are workspace-wide, and the root `Cargo.toml`'s
// `[profile.release]` sets `panic = "abort"` for the CLI binary. Typing the
// wrong-but-easy command `cargo build --release -p ntlmrain-server` would
// silently inherit that: `tower_http::catch_panic::CatchPanicLayer`
// (`http.rs`) becomes inert (there's nothing to unwind into), and so does
// `worker.rs`'s `catch_unwind` around the backend call -- a single
// handler/backend panic would then kill the whole daemon instead of
// degrading to a `500`/failed job. A doc comment nobody reads is not a
// guard, so this fails the build instead.
//
// This deliberately does NOT use a `build.rs` reading `CARGO_CFG_PANIC`:
// measured against this toolchain, that build-script env var reports the
// *target's default* panic strategy, not the panic strategy the profile
// actually resolves to for this compilation -- it read "unwind" even when
// rustc was invoked with `-C panic=abort` under `cargo build --release`.
// `#[cfg(panic = "...")]` used directly in source, by contrast, is filled
// in by rustc from the real `-C panic` flag it was invoked with for *this*
// compilation unit, so it can't drift from what's actually happening.
// `cargo test` always forces `unwind` regardless of profile, so this never
// fires there.
#[cfg(not(panic = "unwind"))]
compile_error!(
    "ntlmrain-server must be built with panic=\"unwind\", but this build's \
     panic strategy is not \"unwind\". Cargo profiles are workspace-wide, so \
     the root crate's [profile.release] (panic=\"abort\") applies to `cargo \
     build --release -p ntlmrain-server` too, which would silently disable \
     tower_http::catch_panic::CatchPanicLayer and worker.rs's catch_unwind \
     guard around backend panics. Build this crate with:\n\n    cargo build \
     --profile release-server -p ntlmrain-server\n"
);

use ntlmrain_server::config::Config;

#[tokio::main]
async fn main() {
    // clap exits the process itself (with a usage message) if a required
    // flag/env var -- e.g. --data-base, --index, --state-dir -- is
    // missing, so there's nothing more to do for that case here.
    let config = Config::parse();
    if let Err(error) = config.validate() {
        eprintln!("ntlmrain-server: {error}");
        std::process::exit(1);
    }

    let listen = config.listen.clone();
    let app = match ntlmrain_server::build_app(config) {
        Ok(app) => app,
        Err(error) => {
            // Covers a table that won't open, a state dir that can't be
            // created, etc. -- deliberately exits before ever binding the
            // listener, since a running service that can't answer
            // /health/ready truthfully is worse than not starting.
            eprintln!("ntlmrain-server: startup failed: {error:#}");
            std::process::exit(1);
        }
    };
    // Held for the process's lifetime: dropping either stops its
    // background threads (see their own `Drop` impls).
    let _pool = app.pool;
    let _reaper = app.reaper;
    let _store = app.store;

    let listener = match tokio::net::TcpListener::bind(&listen).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("ntlmrain-server: failed to bind --listen {listen}: {error}");
            std::process::exit(1);
        }
    };
    println!("ntlmrain-server: listening on {listen}");

    if let Err(error) = axum::serve(listener, app.router).await {
        eprintln!("ntlmrain-server: server error: {error}");
        std::process::exit(1);
    }
}
