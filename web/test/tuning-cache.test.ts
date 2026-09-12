// Regression test for the final whole-branch review's finding: the tuning
// cache key must actually digest the shipped WGSL sources + LUT bytes (so a
// changed shader/LUT invalidates a stale cached selection), not empty
// placeholders. `main.ts` previously passed `""`/`new Uint8Array(0)` for
// these fields, making cache invalidation on a shader/LUT change inert.

import { describe, expect, it } from "vitest";
import { computeTuningCacheKey, type TuningCacheKeyInputs } from "../src/webgpu/tuning-cache";

function fakeDevice(): GPUDevice {
  return {
    limits: {
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 30,
      maxComputeWorkgroupStorageSize: 16_384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
      minStorageBufferOffsetAlignment: 256,
    },
  } as unknown as GPUDevice;
}

function baseInputs(overrides: Partial<TuningCacheKeyInputs> = {}): TuningCacheKeyInputs {
  return {
    adapterInfo: { vendor: "vendorX", architecture: "archX", description: "descX" },
    device: fakeDevice(),
    precomputeCompactSource: "compact wgsl source",
    precomputeExpandedSource: "expanded wgsl source",
    falseAlarmCompactSource: "",
    falseAlarmExpandedSource: "",
    desLutBytes: new Uint8Array([1, 2, 3, 4]),
    ...overrides,
  };
}

describe("computeTuningCacheKey", () => {
  it("produces a real, non-trivial digest for non-empty shader/LUT inputs", async () => {
    const key = await computeTuningCacheKey(baseInputs());
    // SHA-256 hex digest: 64 hex chars.
    expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("changes when the precompute shader source text changes (cache invalidation is not inert)", async () => {
    const original = await computeTuningCacheKey(baseInputs());
    const changed = await computeTuningCacheKey(
      baseInputs({ precomputeCompactSource: "a completely different compact wgsl source" }),
    );
    expect(changed).not.toBe(original);
  });

  it("changes when the expanded shader source text changes", async () => {
    const original = await computeTuningCacheKey(baseInputs());
    const changed = await computeTuningCacheKey(
      baseInputs({ precomputeExpandedSource: "a completely different expanded wgsl source" }),
    );
    expect(changed).not.toBe(original);
  });

  it("changes when the DES LUT bytes change", async () => {
    const original = await computeTuningCacheKey(baseInputs());
    const changed = await computeTuningCacheKey(baseInputs({ desLutBytes: new Uint8Array([9, 9, 9, 9]) }));
    expect(changed).not.toBe(original);
  });

  it("stays stable for byte-identical inputs (same key on every run, not just non-empty)", async () => {
    const first = await computeTuningCacheKey(baseInputs());
    const second = await computeTuningCacheKey(baseInputs());
    expect(first).toBe(second);
  });
});
