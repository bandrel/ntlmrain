// Regression tests for the final whole-branch review's finding: the
// tie-break hysteresis in `webgpu/tuning.ts::stableTuningWinner` (keep the
// cached incumbent if it's within noise of a new leader) was dead code in
// production, because `createDefaultPorts`'s `defaultGetTuning` only ever
// called `autoTuneDevice` on a cache MISS (where, by construction, no
// incumbent can exist) and never on a cache HIT.
//
// `autoTuneDevice` needs a real `GPUDevice`/`GPUBuffer` to run, so this file
// mocks `webgpu/tuning` and `webgpu/tuning-cache` rather than exercising a
// real GPU, and asserts purely on how `createDefaultPorts`'s `getTuning` port
// wires the cache read/write and the `incumbent` argument together.

import { beforeEach, describe, expect, it, vi } from "vitest";
import type { TuningSelection } from "../src/webgpu/tuning";

const getCachedTuningMock = vi.fn<(key: string) => Promise<TuningSelection | null>>();
const putCachedTuningMock = vi.fn<(key: string, selection: TuningSelection) => Promise<void>>();
const computeTuningCacheKeyMock = vi.fn(async (_inputs: unknown) => "fixed-cache-key");
const autoTuneDeviceMock = vi.fn();

vi.mock("../src/webgpu/tuning-cache", () => ({
  getCachedTuning: (key: string) => getCachedTuningMock(key),
  putCachedTuning: (key: string, selection: TuningSelection) => putCachedTuningMock(key, selection),
  computeTuningCacheKey: (inputs: unknown) => computeTuningCacheKeyMock(inputs),
}));

vi.mock("../src/webgpu/tuning", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/webgpu/tuning")>();
  return {
    ...actual,
    autoTuneDevice: (options: unknown) => autoTuneDeviceMock(options),
  };
});

// Imported after the mocks above so `createDefaultPorts` picks up the mocked
// `webgpu/tuning` and `webgpu/tuning-cache` modules.
const { createDefaultPorts } = await import("../src/pipeline/orchestrator");

function cachedSelection(): TuningSelection {
  return {
    shader: "expanded",
    workgroupSize: 128,
    minimumWorkgroups: 128,
    estimatedStepsPerSecond: 42,
    source: "auto-tune",
    selectionReason: "clear-winner",
    tuningElapsedMs: 100,
    deadlineReached: false,
    measurements: [],
  };
}

function freshSelection(): TuningSelection {
  return {
    shader: "compact",
    workgroupSize: 64,
    minimumWorkgroups: 128,
    estimatedStepsPerSecond: 99,
    source: "auto-tune",
    selectionReason: "cached-winner-retained",
    tuningElapsedMs: 200,
    deadlineReached: false,
    measurements: [],
  };
}

function baseOptions(overrides: Partial<Parameters<typeof createDefaultPorts>[0]> = {}) {
  return {
    device: {} as GPUDevice,
    lutBuffer: {} as GPUBuffer,
    precomputeDeviceLimits: {
      maxComputeWorkgroupsPerDimension: 65_535,
      minStorageBufferOffsetAlignment: 256,
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 30,
    },
    tuningDeviceLimits: { maxComputeInvocationsPerWorkgroup: 1_024, maxComputeWorkgroupSizeX: 1_024 },
    supportedShaders: ["compact", "expanded"] as ("compact" | "expanded")[],
    tuningCacheKeyInputs: {
      adapterInfo: { vendor: "", architecture: "", description: "" },
      device: {} as GPUDevice,
      precomputeCompactSource: "compact source",
      precomputeExpandedSource: "expanded source",
      falseAlarmCompactSource: "",
      falseAlarmExpandedSource: "",
      desLutBytes: new Uint8Array([1, 2, 3]),
    },
    lookupConfig: { baseUrl: "http://localhost" },
    ...overrides,
  };
}

describe("createDefaultPorts's getTuning: cache + forced re-tune + hysteresis wiring", () => {
  beforeEach(() => {
    getCachedTuningMock.mockReset();
    putCachedTuningMock.mockReset();
    autoTuneDeviceMock.mockReset();
  });

  it("on a cache hit with no forceRetune, returns the cached selection directly without calling autoTuneDevice (default fast path unchanged)", async () => {
    getCachedTuningMock.mockResolvedValue(cachedSelection());
    const ports = createDefaultPorts(baseOptions());

    const result = await ports.getTuning();

    expect(autoTuneDeviceMock).not.toHaveBeenCalled();
    expect(result.source).toBe("cache");
    expect(result.selectionReason).toBe("cached-winner");
    expect(result.shader).toBe("expanded");
    expect(result.workgroupSize).toBe(128);
  });

  it("on a cache hit with forceRetune, calls autoTuneDevice with the cached selection as `incumbent` (exercising the hysteresis path)", async () => {
    getCachedTuningMock.mockResolvedValue(cachedSelection());
    autoTuneDeviceMock.mockResolvedValue(freshSelection());
    const ports = createDefaultPorts(baseOptions({ forceRetune: true }));

    const result = await ports.getTuning();

    expect(autoTuneDeviceMock).toHaveBeenCalledTimes(1);
    const passedOptions = autoTuneDeviceMock.mock.calls[0][0] as { incumbent?: unknown };
    expect(passedOptions.incumbent).toEqual({ shader: "expanded", workgroupSize: 128 });
    expect(result).toEqual(freshSelection());
  });

  it("on a cache miss, calls autoTuneDevice with no incumbent (nothing to prefer)", async () => {
    getCachedTuningMock.mockResolvedValue(null);
    autoTuneDeviceMock.mockResolvedValue(freshSelection());
    const ports = createDefaultPorts(baseOptions());

    await ports.getTuning();

    expect(autoTuneDeviceMock).toHaveBeenCalledTimes(1);
    const passedOptions = autoTuneDeviceMock.mock.calls[0][0] as { incumbent?: unknown };
    expect(passedOptions.incumbent).toBeUndefined();
  });

  it("persists a genuine fresh auto-tune result to the cache after a forced re-tune", async () => {
    getCachedTuningMock.mockResolvedValue(cachedSelection());
    autoTuneDeviceMock.mockResolvedValue(freshSelection());
    const ports = createDefaultPorts(baseOptions({ forceRetune: true }));

    await ports.getTuning();

    expect(putCachedTuningMock).toHaveBeenCalledWith("fixed-cache-key", freshSelection());
  });
});
