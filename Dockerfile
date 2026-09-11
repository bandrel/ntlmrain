# Multi-stage build for ntlmrain-server
#
# Environment variables required at runtime:
#   NTLMRAIN_SERVER_DATA_BASE - path to GRTB table shards base
#   NTLMRAIN_SERVER_INDEX - path to GIDX index file
#   NTLMRAIN_SERVER_STATE_DIR - directory for SQLite queue DB and jobs
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

# Set default listen address to bind all interfaces (required for reverse-proxy reach)
ENV NTLMRAIN_SERVER_LISTEN=0.0.0.0:8080

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
