# ntlmrain-server

A queue-backed HTTP service that recovers NT hashes from `NetNTLMv1`
endpoint files against a local GRTB/GIDX rainbow table, exposing the same
wire protocol as the public `https://lookup.ntlmrain.com` service so the
`ntlmrain` CLI needs no changes beyond `--remote-url` to point at it.

## Building

```sh
cargo build --profile release-server -p ntlmrain-server
```

Do **not** use `cargo build --release -p ntlmrain-server`: Cargo profiles
are workspace-wide, and the root `Cargo.toml`'s `[profile.release]` (used
by the `ntlmrain` CLI) sets `panic = "abort"`, which silently disables both
`tower_http::catch_panic::CatchPanicLayer` and `worker.rs`'s
`catch_unwind` guard around backend panics -- a single handler/backend
panic would then kill the whole daemon instead of degrading to a `500` or
a `failed` job. The `release-server` profile (`inherits = "release"`,
`panic = "unwind"`) exists specifically to avoid that. A build-time guard
in `main.rs` (`#[cfg(not(panic = "unwind"))] compile_error!(...)`) refuses
to compile the wrong way, so getting this wrong fails loudly rather than
silently.

## Running

Required flags: `--data-base` and `--index` (the GRTB/GIDX table to serve
lookups from) and `--state-dir` (holds the SQLite job queue and the
`jobs/` input/result blob tree). Every flag also has an
`NTLMRAIN_SERVER_*` environment variable equivalent.

```sh
cargo run --profile release-server -p ntlmrain-server -- \
    --data-base /data/tables/ntlmv1 \
    --index /data/tables/ntlmv1.gidx \
    --state-dir /var/lib/ntlmrain-server \
    --listen 0.0.0.0:8080
```

See `--help` for the full flag surface (lookup concurrency, queue depth,
validation limits, job timeout, and the phase-2 `--static-dir` stub).

## Auth

Anonymous access is allowed unless both `--auth-user` and a password
source are set. The password is never accepted as a plaintext CLI flag
(argv is world-readable); set it one of two ways:

- `--auth-password-file <path>` -- a file containing the password. Its
  contents are trimmed of a trailing newline (so a file created with a
  plain text editor or `echo` works as expected) but otherwise used
  verbatim.
- `NTLMRAIN_SERVER_AUTH_PASSWORD` -- the password directly, via
  environment variable only (no `--auth-password` flag exists).

Setting both, or setting `--auth-user` without either, is a startup error.
When auth is configured, HTTP Basic credentials are required on every
`/api/v1/submissions*` route; `/health/*`, `/openapi.json`, and `/docs`
stay unauthenticated.

## Health checks

- `GET /health/live` -- `200` once the process is up. Does not reflect
  table or queue state; use this for a liveness probe.
- `GET /health/ready` -- `200` once the GRTB/GIDX table has opened
  successfully, with a body reporting a snapshot of the table's
  `TableInfo` (`records`, `blocks`, `parts`, `min_endpoint`,
  `max_endpoint`); `503` otherwise. Use this for a readiness probe --
  a service that's live but not yet ready cannot serve lookups.

## Running in Docker

A multi-stage Dockerfile and `docker-compose.yml` are provided at the
repository root. Both files are designed for self-hosted deployments and
assume you will integrate this service with your own reverse-proxy setup
(Traefik, Caddy, nginx, etc.) — no proxy or TLS is baked in.

### Building the image

```sh
docker compose build
# or manually:
docker build -t ntlmrain-server .
```

**Important:** The service must be built with the `release-server` profile
to enable panic recovery (see the ["Building"](#building) section above).
The Dockerfile enforces this via `cargo build --profile release-server`.
Do not attempt to bypass this or build with plain `--release`.

### Running the container

The easiest way is to use the compose file:

```sh
docker compose up
```

Before running, edit `docker-compose.yml` to point the bind-mount at your
actual table directory (replace `/path/to/your/table` with the real path)
and set the `NTLMRAIN_SERVER_DATA_BASE` and `NTLMRAIN_SERVER_INDEX` paths
if you mount the table at a different in-container location.

For example, if your table is at `/mnt/tables/ntlmv1-rainbow`:

```yaml
volumes:
  - /mnt/tables/ntlmv1-rainbow:/data:ro

environment:
  NTLMRAIN_SERVER_DATA_BASE: /data/ntlmv1
  NTLMRAIN_SERVER_INDEX: /data/ntlmv1.gidx
```

The container exposes port 8080 and binds all interfaces
(`NTLMRAIN_SERVER_LISTEN=0.0.0.0:8080`) to allow a reverse proxy on
another container or host to reach it.

### Reverse-proxy setup

In a multi-service Docker deployment, attach this service to a shared
external network rather than publishing the port on the host:

```yaml
services:
  ntlmrain-server:
    # ... (rest of config)
    networks:
      - proxy-network

networks:
  proxy-network:
    external: true
```

Then configure your reverse proxy to reach this service at
`ntlmrain-server:8080` on the `proxy-network` network.

### Authentication

To enable HTTP Basic auth, set both `NTLMRAIN_SERVER_AUTH_USER` and one
of:
- `NTLMRAIN_SERVER_AUTH_PASSWORD_FILE` (path to a file with the password)
- `NTLMRAIN_SERVER_AUTH_PASSWORD` (password directly, via environment only)

See the `docker-compose.yml` comments for examples.
