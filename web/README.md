# ntlmrain web UI

A browser-based, WebGPU-driven NetNTLMv1 recovery client, served as a
static site by `ntlmrain-server`'s `--static-dir` flag. It runs GPU
precompute directly in the tab (via WebGPU compute shaders shared with the
native CLI), submits/polls/downloads against the same server's `/api/v1`
routes, and verifies candidates locally via a WASM build of the project's
Rust crypto code (`crypto-wasm/`).

## Prerequisites

- Rust + [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) (`cargo install wasm-pack`), to build `crypto-wasm/`.
- Node.js (a current LTS) + npm, to build `web/`.

## Build sequence

Run these in order, from the repository root:

```sh
# 1. Build crypto-wasm's browser package (produces crypto-wasm/pkg/, which
#    web/src/crypto/index.ts imports by relative path).
cd crypto-wasm
wasm-pack build --target web
cd ..

# 2. Build the web UI (also copies shaders/*.wgsl + shaders/des_lut.bin into
#    web/public/shaders/ via the predev/prebuild npm scripts).
cd web
npm ci
npm run build
cd ..

# 3. Point the server at the built output.
cargo run -p server -- --static-dir web/dist [other server flags...]
```

`web/dist/` is a plain static site — any `ntlmrain-server` invocation with
`--static-dir web/dist` will serve it at `/`, alongside the existing
`/api/v1/*` routes, health checks, and OpenAPI docs (see `server/README.md`'s
"Static assets" section).

## Browser requirement

You need a real WebGPU-capable browser — a current version of Chrome or
Edge. There is no CPU/WASM fallback for the GPU precompute step; if
`navigator.gpu` is unavailable, the page reports that plainly rather than
silently falling back.

## Same-origin constraint

The server has no CORS layer, by design. The UI must be loaded from, and
will only ever talk to, its own origin (`window.location.origin`) — do not
try to serve `web/dist/` from a separate host/port than the
`ntlmrain-server` instance it submits lookups to.

## Development

`npm run dev` (from `web/`) starts a Vite dev server for iterating on the UI
without a full production build; `npm test` runs the Vitest unit suite,
`npm run typecheck` runs `tsc --noEmit`. Both `crypto-wasm/pkg/` (step 1
above) and `web/public/shaders/` (Vite's `predev` script) must exist first.
