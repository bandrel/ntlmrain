# Multi-stage build for ntlmrain-server
#
# Environment variables required at runtime:
#   NTLMRAIN_SERVER_DATA_BASE - path to GRTB table shards base
#   NTLMRAIN_SERVER_INDEX - path to GIDX index file
#   NTLMRAIN_SERVER_STATE_DIR - directory for SQLite queue DB and jobs
# Set by this image (override only to serve a different static bundle):
#   NTLMRAIN_SERVER_STATIC_DIR - directory of static web assets served at /
# Optional auth env vars (both must be set together):
#   NTLMRAIN_SERVER_AUTH_USER - HTTP Basic auth username
#   NTLMRAIN_SERVER_AUTH_PASSWORD or NTLMRAIN_SERVER_AUTH_PASSWORD_FILE - password source

# ===== Builder Stage =====
# Pinned above the workspace's own edition-2024 floor (rustc >=1.85): a
# transitive dependency (wide/safe_arch, pulled in via bytemuck) currently
# requires rustc >=1.89, so this pin tracks that higher floor instead.
FROM rust:1.90-bookworm AS builder

# Install C compiler required by rusqlite's bundled SQLite feature
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source and build the server binary with the release-server profile
COPY . .

# Build ntlmrain-server with the release-server profile
# (panic = "unwind" is required; plain --release fails the compile_error guard)
RUN cargo build --profile release-server -p ntlmrain-server

# ===== WASM Builder Stage =====
# The browser UI verifies candidates locally through a wasm-bindgen build of
# the workspace's Rust crypto code. web/src/crypto/index.ts imports it by
# relative path (../../../crypto-wasm/pkg/crypto_wasm.js), so crypto-wasm/pkg
# must exist before the Vite build runs.
FROM rust:1.90-bookworm AS wasm-builder

ARG TARGETARCH
ARG WASM_PACK_VERSION=0.15.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Prefer the prebuilt release binary over `cargo install wasm-pack`, which
# would compile the tool itself on every cache miss. The release assets are
# per-architecture, so map Docker's TARGETARCH onto the target triple rather
# than hardcoding x86_64 and breaking arm64 builds.
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
        amd64) arch=x86_64 ;; \
        arm64) arch=aarch64 ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -sSfL "https://github.com/rustwasm/wasm-pack/releases/download/v${WASM_PACK_VERSION}/wasm-pack-v${WASM_PACK_VERSION}-${arch}-unknown-linux-musl.tar.gz" \
      | tar -xzf - -C /usr/local/bin --strip-components=1 --wildcards '*/wasm-pack'; \
    wasm-pack --version

WORKDIR /build

# crypto-wasm depends on the root crate by path, so it needs the full source
# tree rather than just its own directory.
COPY . .

RUN wasm-pack build crypto-wasm --target web

# ===== Web UI Builder Stage =====
FROM node:22-bookworm-slim AS web-builder

WORKDIR /build

# Install dependencies before copying sources so edits to web/src don't
# invalidate the npm layer.
COPY web/package.json web/package-lock.json web/
RUN npm --prefix web ci

# The prebuild script copies shaders/ (WGSL sources + des_lut.bin) into
# web/public/shaders/, and vite.config.ts allow-lists the repo root so the
# ?raw shader imports can reach outside web/.
COPY shaders shaders
COPY web web
COPY --from=wasm-builder /build/crypto-wasm/pkg crypto-wasm/pkg

RUN npm --prefix web run build

# ===== Runtime Stage =====
FROM debian:bookworm-slim

# Install minimal runtime dependencies
# ca-certificates for HTTPS support (pure Rust TLS via rustls)
# curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with fixed uid/gid for predictable ownership
# in bind-mounts (operators can align host permissions via `chown 1000:1000`)
RUN groupadd -g 1000 ntlmrain && \
    useradd -u 1000 -g 1000 -s /sbin/nologin -M ntlmrain

# Create necessary directories and set ownership
RUN mkdir -p /data /state && \
    chown -R ntlmrain:ntlmrain /data /state

WORKDIR /home/ntlmrain

# Copy only the built binary from builder stage
COPY --from=builder --chown=ntlmrain:ntlmrain /build/target/release-server/ntlmrain-server /usr/local/bin/

# Copy the built browser UI and serve it at / by default. The UI is
# same-origin by design (the server has no CORS layer), so it has to be
# served by the very instance it submits lookups to.
COPY --from=web-builder --chown=ntlmrain:ntlmrain /build/web/dist /usr/share/ntlmrain/web

# Set default listen address to bind all interfaces (required for reverse-proxy reach)
ENV NTLMRAIN_SERVER_LISTEN=0.0.0.0:8080

# Serve the bundled UI. Override to point at a different bundle; there is no
# supported way to disable static serving other than pointing this elsewhere.
ENV NTLMRAIN_SERVER_STATIC_DIR=/usr/share/ntlmrain/web

# Switch to non-root user
USER ntlmrain

# Expose the default listen port
EXPOSE 8080

# Healthcheck: use /health/live (fast liveness check)
# Don't use /health/ready (blocks until table loads, fails on initial healthcheck for large tables)
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Run the binary directly; no wrapper shell script needed since all flags have env-var equivalents
ENTRYPOINT ["ntlmrain-server"]
