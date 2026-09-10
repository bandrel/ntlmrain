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
