// Copies the root Rust crate's shaders/ directory (WGSL sources + the
// des_lut.bin binary LUT asset) into web/public/shaders/ so:
//   - des_lut.bin is fetchable at runtime from `/shaders/des_lut.bin`
//     (webgpu/precompute.ts fetches it directly; there's no other way to
//     get a binary asset in front of the browser).
//   - the WGSL sources are available under the same documented convention
//     even though webgpu/precompute.ts actually bundles their *text* via
//     Vite `?raw` imports straight from ../shaders/ (no extra network
//     round-trip, no dependency on this copy step being fresh at dev time).
//
// Run via the `predev`/`prebuild` npm scripts, before `vite`/`vite build`.
import { existsSync, mkdirSync, readdirSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = join(here, "..");
const repoRoot = join(webRoot, "..");
const sourceDir = join(repoRoot, "shaders");
const destDir = join(webRoot, "public", "shaders");

if (!existsSync(sourceDir)) {
  console.error(`copy-shaders: source directory not found: ${sourceDir}`);
  process.exit(1);
}

mkdirSync(destDir, { recursive: true });

let copied = 0;
for (const entry of readdirSync(sourceDir, { withFileTypes: true })) {
  if (!entry.isFile()) continue;
  copyFileSync(join(sourceDir, entry.name), join(destDir, entry.name));
  copied += 1;
}

console.log(`copy-shaders: copied ${copied} file(s) from ${sourceDir} to ${destDir}`);
