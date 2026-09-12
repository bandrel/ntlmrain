import { fileURLToPath } from "node:url";
import path from "node:path";
import { defineConfig } from "vitest/config";

const here = path.dirname(fileURLToPath(import.meta.url));
// The repo root, one level up from web/. src/crypto/index.ts imports
// crypto-wasm's wasm-pack output via a relative path that reaches outside
// web/'s own root (../../../crypto-wasm/pkg/...), and webgpu/precompute.ts
// bundles the WGSL sources from ../../../shaders/*.wgsl via `?raw` imports.
// Vite's dev server refuses to serve files outside root unless allow-listed
// here; this only affects `vite dev`, not `vite build` (Rollup reads
// arbitrary paths at build time regardless).
const repoRoot = path.resolve(here, "..");

export default defineConfig({
  root: here,
  server: {
    fs: {
      allow: [repoRoot],
    },
  },
  build: {
    target: "es2022",
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
