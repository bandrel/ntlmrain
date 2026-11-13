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

`--data-base` is a filename **prefix**, not a path that exists on disk.
The server appends a shard suffix to it, so `--data-base /tables/mytable`
is opened as `/tables/mytable.0000.grtb`, `/tables/mytable.0001.grtb`, and
so on. Take the prefix your shard files share and drop the `.NNNN.grtb`
tail. `--index`, by contrast, is a real file: that same prefix plus
`.gidx`.

```sh
cargo run --profile release-server -p ntlmrain-server -- \
    --data-base '/tables/netntlmv1_byte#7-7_0_881689x134217668' \
    --index '/tables/netntlmv1_byte#7-7_0_881689x134217668.gidx' \
    --state-dir /var/lib/ntlmrain-server \
    --listen 0.0.0.0:8080
```

To check a prefix before starting the server, glob it:

```sh
ls '/tables/netntlmv1_byte#7-7_0_881689x134217668'.*.grtb | head
```

See `--help` for the full flag surface (lookup concurrency, queue depth,
validation limits, job timeout, and `--static-dir` for serving static web assets).

## Static assets

To serve a web UI or other static assets, pass `--static-dir /path/to/web/dist`. Files in that directory are served at `/` alongside the API routes (`/api/v1/*`), health checks, and OpenAPI documentation. See `web/README.md` for how to build the bundled browser WebGPU UI into a `web/dist` directory suitable for this flag.

The Docker image does this for you: it builds `crypto-wasm` with `wasm-pack`
and the UI with Vite, ships the result at `/usr/share/ntlmrain/web`, and sets
`NTLMRAIN_SERVER_STATIC_DIR` to point there. A container started from it
serves the UI at `/` with no extra configuration.

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
cp .env.example .env
$EDITOR .env
docker compose up
```

All configuration lives in `.env`; `docker-compose.yml` needs no edits for
a normal deployment. Three settings are required, and the rest of
`.env.example` documents the optional ones.

Point `TABLES_DIR` at the directory holding your shards, then set
`NTLMRAIN_SERVER_DATA_BASE` and `NTLMRAIN_SERVER_INDEX` to match, as paths
*inside* the container — that directory is bind-mounted read-only at
`/data`. The host side of the mount is the **directory containing** the
shards, not the table itself; as described under [Running](#running),
there is no single table file to point at.

For example, for shards named
`/srv/tables/netntlmv1_byte#7-7_0_881689x134217668.0000.grtb` and up:

```sh
TABLES_DIR=/srv/tables
NTLMRAIN_SERVER_DATA_BASE=/data/netntlmv1_byte#7-7_0_881689x134217668
NTLMRAIN_SERVER_INDEX=/data/netntlmv1_byte#7-7_0_881689x134217668.gidx
```

A `#` only starts a comment in this file when it follows whitespace, so a
table name containing one needs no quoting or escaping. Quote only when
the name contains a space.

Leave an optional setting commented out to accept its default rather than
setting it to an empty value. The server parses its config with clap,
which treats an empty string as a value that *was* supplied, so
`NTLMRAIN_SERVER_READ_WORKERS=` is a parse error rather than "unset".

Forgetting `TABLES_DIR` fails immediately, at compose-parse time, with a
message naming the variable — the bind-mount guards on it.

The container drops to uid 1000, so the shards and the `.gidx` must be
readable by that uid or world-readable. A root-owned `0600` table
bind-mounts cleanly and then fails at open time, which reads as a config
error but is a permissions one.

The container exposes port 8080 and binds all interfaces
(`NTLMRAIN_SERVER_LISTEN=0.0.0.0:8080`) to allow a reverse proxy on
another container or host to reach it.

### TLS

`docker compose up` also starts a Caddy reverse proxy on 443, because the
browser UI needs it. Browsers expose `navigator.gpu` only in a secure
context, and plain HTTP on a LAN address is not one, so over HTTP the UI
loads and then reports WebGPU as unavailable. `http://localhost` *is* a
secure context, so an SSH tunnel works too if you would rather not run the
proxy.

The server itself speaks plain HTTP only; it has no TLS listener. All TLS
terminates at the proxy.

Two modes, both set in `.env` (see `.env.example`):

- **Self-signed (default).** A `cert-init` service generates the certificate
  before Caddy starts, and does not overwrite an existing one, so restarts
  keep the same identity. Set `CERT_HOSTS` to the addresses you actually
  browse to; browsers match on `subjectAltName`, so an address missing from
  that list will mismatch. The browser warns once, and clicking through is
  enough for WebGPU.

  The `ntlmrain` CLI has no such click-through: it verifies against a
  built-in root store and ignores the platform trust store, so a self-signed
  certificate fails with `lookup connection failed: ... UnknownIssuer`. Copy
  the certificate out of the `certs` volume and point the client at it:

  ```console
  $ docker compose cp cert-init:/certs/tls.crt ntlmrain-ca.pem
  $ ntlmrain crack --remote-url https://lookup.example.internal \
      --remote-ca-cert ntlmrain-ca.pem ...
  ```

  `--remote-insecure` skips verification instead, which is fine for a
  throwaway lab server but leaves the connection open to interception.
- **Let's Encrypt.** Set `SITE_ADDRESS` to a domain resolving to this host
  and `TLS_ARGS` to your email address. Requires port 80 reachable for the
  HTTP-01 challenge. The generated self-signed files are ignored.

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

### Troubleshooting: 502 Bad Gateway from the proxy

A `502` from Caddy means the proxy is up but nothing answered on
`ntlmrain-server:8080`. The Caddy log shows it as:

```text
"msg":"dial tcp 172.18.0.2:8080: connect: connection refused" ... "status":502
```

The usual cause is a server that is still starting, not one that crashed.
`ntlmrain-server` opens the whole table (every GRTB shard, plus the mmapped
GIDX index) *before* it binds the listener, so a service that can't answer
`/health/ready` truthfully never accepts connections. It prints nothing
while it does this. A full-size table means thousands of shards and a
multi-gigabyte `.gidx`, and opening that takes long enough to look like a
hang. Until the server prints `ntlmrain-server: listening on 0.0.0.0:8080`,
every request through the proxy gets a `502`.

The trap is stopping it during that silent window. In one real incident,
the stack was started in the foreground with `docker compose up` and
interrupted with Ctrl-C about 18 seconds in, twice. The table was still
loading both times, so the browser only ever saw a `502`. Left alone, that
same server (4,096 shards, a 3.9 GB `.gidx`, local ext4 disk) printed its
listening line 21 seconds after the container started, so both stops came
only seconds early. Expect the time to vary with storage speed and with
how much of the table is already in the page cache. `docker ps -a`
then showed:

```text
ntlmrain-ntlmrain-server-1   Exited (137)
ntlmrain-caddy-1             Exited (0)
```

Exit `137` is SIGKILL. Here it came from the stop, not from the kernel's OOM
killer. Check before assuming a crash:

```sh
docker inspect -f '{{.State.OOMKilled}} {{.State.StartedAt}} {{.State.FinishedAt}}' ntlmrain-ntlmrain-server-1
docker events --since 30m --until 0s \
  --filter label=com.docker.compose.project=ntlmrain \
  --format '{{.Time}} {{.Action}} {{.Actor.Attributes.name}}'
```

`OOMKilled=false` together with `kill` events just before the `die` events
means something stopped the stack on purpose (Ctrl-C, `docker compose stop`,
or `down`). `attach` events at startup point to a foreground `docker compose up`.

#### Fix

Start the stack detached, then wait for the listening line instead of
judging it by the browser:

```sh
docker compose up -d --build
docker compose logs -f ntlmrain-server   # wait for "listening on 0.0.0.0:8080"
curl -sk https://localhost/health/ready  # 200 with TableInfo once the table is open
```

Notes:

- `--build` matters after a `git pull`. Plain `docker compose up` reuses
  whatever image was built last, which can be months behind the checked-out
  source. Check the image's age with
  `docker image inspect -f '{{.Created}}' ntlmrain-ntlmrain-server`.
- Docker reports the container as `unhealthy` during a long table load.
  That's expected: the image's healthcheck (`/health/live`, 5-second start
  period) can't pass until the listener is bound. `unhealthy` by itself
  doesn't restart the container, so leave it alone.
- On a host where your user isn't in the `docker` group, prefix every
  command with `sudo`.

If the server exits on its own instead (status `Exited (1)`), the reason is
in its log, since startup failures are printed before exit:

```sh
docker compose logs ntlmrain-server
```

Common ones are `startup failed: failed to open local lookup table`, caused
by a wrong `NTLMRAIN_SERVER_DATA_BASE`/`NTLMRAIN_SERVER_INDEX`, shards not
readable by uid 1000 (see [Running the container](#running-the-container)),
or a `nofile` limit below the shard count.
