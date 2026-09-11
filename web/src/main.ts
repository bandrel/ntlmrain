// Minimal entry point. There is no real UI yet (Task 6 builds that); this
// wires up just enough to let a human manually exercise the WebGPU
// precompute pipeline from the devtools console via index.html's debug
// harness page, per this task's brief.
import { requestGpuDevice, supportsExpandedShader } from "./webgpu/device";
import { createPrecomputePipeline, loadDesLut, runPrecompute } from "./webgpu/precompute";
import { initCrypto, encode_endpoint_file } from "./crypto";

function log(message: string): void {
  const el = document.getElementById("log");
  console.log(message);
  if (el) el.textContent += `${message}\n`;
}

function hexToBytes(hex: string): Uint8Array {
  if (!/^[0-9a-fA-F]{16}$/.test(hex)) {
    throw new Error("expected a 16-character (8-byte) hex string");
  }
  const bytes = new Uint8Array(8);
  for (let i = 0; i < 8; i += 1) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

/**
 * Debug entry point: precompute the full endpoint set for `targetHex` (a
 * 16-hex-char, 8-byte target) and log the resulting NTLMEND1 file's byte
 * length. Exposed on `window` for manual console-driven verification —
 * there is no automated real-GPU test harness in this environment (see
 * Task 3's report for why).
 */
async function debugPrecompute(targetHex: string): Promise<void> {
  log(`requesting GPU device...`);
  const { device } = await requestGpuDevice({ powerPreference: "high-performance" });
  log(`device acquired; expanded-shader support: ${supportsExpandedShader(device)}`);

  const workgroupSize = 64;
  const pipeline = createPrecomputePipeline(device, "compact", workgroupSize);
  const lut = await loadDesLut(device);

  await initCrypto();

  const target = hexToBytes(targetHex);
  log("running precompute (this can take a while)...");
  const endpoints = await runPrecompute(
    device,
    pipeline,
    lut,
    {
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      minStorageBufferOffsetAlignment: device.limits.minStorageBufferOffsetAlignment,
    },
    {
      target,
      workgroupSize,
      onProgress: (progress) => {
        log(
          `progress: ${progress.stepsDone}/${progress.stepsTotal} steps ` +
            `(batch ${progress.batchSteps} steps in ${progress.elapsedMs.toFixed(1)}ms)`,
        );
      },
    },
  );

  const encoded = encode_endpoint_file(endpoints);
  log(`done: ${endpoints.length} endpoints, encoded NTLMEND1 file is ${encoded.length} bytes`);
}

declare global {
  interface Window {
    ntlmrainDebugPrecompute: typeof debugPrecompute;
  }
}

window.ntlmrainDebugPrecompute = debugPrecompute;
log("ready. call window.ntlmrainDebugPrecompute(hashHex) from the console.");
