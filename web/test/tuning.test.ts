import { describe, expect, it } from "vitest";
import {
  aggregateTuningSamples,
  alternatingTuningOrder,
  autoTuneDevice,
  calibratedSyntheticSteps,
  coefficientOfVariation,
  confirmedSlowTuningRates,
  medianF64,
  productionTuningMeasurements,
  productionTuningShape,
  stableTuningWinner,
  tuningUncertainty,
  validWorkgroups,
  type TuningKey,
  type TuningMeasurement,
} from "../src/webgpu/tuning";

function row(
  shader: TuningMeasurement["shader"],
  workgroupSize: number,
  rate: number,
  rateCv: number,
): TuningMeasurement {
  return {
    shader,
    workgroupSize,
    stage: "test",
    elapsedMs: 50,
    steps: Math.floor(rate / 20),
    stepsPerSecond: rate,
    valid: true,
    note: undefined,
    sampleCount: 3,
    rateCv,
  };
}

describe("medianF64", () => {
  it("matches gpu.rs's median_f64 for odd and even counts", () => {
    expect(medianF64([3, 1, 2])).toBe(2);
    expect(medianF64([4, 1, 3, 2])).toBe(2.5);
  });

  it("ignores non-finite and non-positive values", () => {
    expect(medianF64([0, -1, NaN, Infinity, 5])).toBe(5);
  });

  it("returns 0 for an empty/invalid input", () => {
    expect(medianF64([])).toBe(0);
    expect(medianF64([0, -5])).toBe(0);
  });
});

describe("coefficientOfVariation", () => {
  it("returns 0 for fewer than two valid samples", () => {
    expect(coefficientOfVariation([])).toBe(0);
    expect(coefficientOfVariation([10])).toBe(0);
  });

  it("computes population stddev / mean", () => {
    // values [10, 20]: mean 15, variance ((5^2 + 5^2)/2) = 25, stddev 5, cv 5/15
    expect(coefficientOfVariation([10, 20])).toBeCloseTo(5 / 15, 12);
  });
});

describe("aggregateTuningSamples", () => {
  const key: TuningKey = { shader: "compact", workgroupSize: 64 };

  it("returns null when no sample is valid", () => {
    const invalid: TuningMeasurement = { ...row(key.shader, key.workgroupSize, 100, 0), valid: false };
    expect(aggregateTuningSamples(key, "family", [invalid])).toBeNull();
  });

  it("sums elapsed/steps and takes the median rate across valid samples", () => {
    const samples = [
      row(key.shader, key.workgroupSize, 100, 0),
      row(key.shader, key.workgroupSize, 200, 0),
      { ...row(key.shader, key.workgroupSize, 999, 0), valid: false },
    ];
    const aggregated = aggregateTuningSamples(key, "family", samples);
    expect(aggregated).not.toBeNull();
    expect(aggregated!.sampleCount).toBe(2);
    expect(aggregated!.stepsPerSecond).toBe(150);
    expect(aggregated!.elapsedMs).toBe(100);
    expect(aggregated!.steps).toBe(samples[0].steps + samples[1].steps);
  });
});

describe("tuningUncertainty", () => {
  it("clamps 2*rate_cv to [0.03, 0.10]", () => {
    expect(tuningUncertainty(row("compact", 64, 100, 0))).toBeCloseTo(0.03, 12);
    expect(tuningUncertainty(row("compact", 64, 100, 0.01))).toBeCloseTo(0.03, 12);
    expect(tuningUncertainty(row("compact", 64, 100, 0.02))).toBeCloseTo(0.04, 12);
    expect(tuningUncertainty(row("compact", 64, 100, 1))).toBeCloseTo(0.1, 12);
  });
});

describe("confirmedSlowTuningRates", () => {
  it("requires at least 2 valid samples strictly below the slow-rate threshold", () => {
    expect(confirmedSlowTuningRates([5_000_000])).toBe(false);
    expect(confirmedSlowTuningRates([9_000_000, 11_000_000])).toBe(false);
    expect(confirmedSlowTuningRates([8_000_000, 9_000_000])).toBe(true);
  });
});

describe("alternatingTuningOrder", () => {
  it("keeps even rounds as-is and reverses odd rounds", () => {
    const keys: TuningKey[] = [
      { shader: "compact", workgroupSize: 32 },
      { shader: "compact", workgroupSize: 64 },
      { shader: "expanded", workgroupSize: 64 },
    ];
    expect(alternatingTuningOrder(keys, 0)).toEqual(keys);
    expect(alternatingTuningOrder(keys, 1)).toEqual([...keys].reverse());
    expect(alternatingTuningOrder(keys, 2)).toEqual(keys);
  });
});

describe("calibratedSyntheticSteps", () => {
  it("returns 1 for non-finite, non-positive, or zero-invocation input", () => {
    expect(calibratedSyntheticSteps(NaN, 50, 100)).toBe(1);
    expect(calibratedSyntheticSteps(0, 50, 100)).toBe(1);
    expect(calibratedSyntheticSteps(-5, 50, 100)).toBe(1);
    expect(calibratedSyntheticSteps(100, 50, 0)).toBe(1);
  });

  it("ceils rate*targetMs/1000/invocations", () => {
    // rate=1000 steps/s, target 50ms -> 50 steps total / 4 invocations = 12.5 -> ceil 13
    expect(calibratedSyntheticSteps(1000, 50, 4)).toBe(13);
  });
});

describe("productionTuningShape", () => {
  it("truncates the endpoint start and clamps slice steps to DEFAULT_CHECKPOINT_STEPS", () => {
    const shape = productionTuningShape(1_000_000, 0.6, 150);
    // (881_689 - 1) * 0.6 = 529_012.8 -> trunc 529012
    expect(shape.endpointStart).toBe(529_012);
    expect(shape.invocations).toBeGreaterThan(0);
    expect(shape.sliceSteps).toBeGreaterThanOrEqual(1);
    expect(shape.sliceSteps).toBeLessThanOrEqual(65_536);
  });
});

describe("productionTuningMeasurements (weighted harmonic mean)", () => {
  it("matches gpu.rs's production_score_is_weighted_harmonic_rate test exactly", () => {
    const key: TuningKey = { shader: "compact", workgroupSize: 64 };
    const samples = new Map<string, TuningMeasurement[]>();
    [100, 50, 25].forEach((rate, index) => {
      samples.set(`${key.shader}:${key.workgroupSize}::${index}`, [row(key.shader, key.workgroupSize, rate, 0)]);
    });
    const [measurement] = productionTuningMeasurements([key], samples);
    const expected = 1 / (0.16 / 100 + 0.4 / 50 + 0.44 / 25);
    expect(measurement.stepsPerSecond).toBeCloseTo(expected, 9);
  });

  it("drops a finalist missing any of the 3 points", () => {
    const key: TuningKey = { shader: "compact", workgroupSize: 64 };
    const samples = new Map<string, TuningMeasurement[]>();
    samples.set(`${key.shader}:${key.workgroupSize}::0`, [row(key.shader, key.workgroupSize, 100, 0)]);
    samples.set(`${key.shader}:${key.workgroupSize}::1`, [row(key.shader, key.workgroupSize, 100, 0)]);
    // point index 2 missing entirely
    expect(productionTuningMeasurements([key], samples)).toEqual([]);
  });
});

describe("validWorkgroups", () => {
  it("filters the native candidate set by device limits", () => {
    expect(
      validWorkgroups({ maxComputeInvocationsPerWorkgroup: 1_024, maxComputeWorkgroupSizeX: 1_024 }),
    ).toEqual([32, 64, 128, 256, 512, 1_024]);
    expect(
      validWorkgroups({ maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 1_024 }),
    ).toEqual([32, 64, 128, 256]);
  });
});

describe("autoTuneDevice's forcedWorkgroup validation", () => {
  // Regression test for the final whole-branch review's finding: picking
  // 512/1024 in the UI's manual override used to bypass `validWorkgroups`
  // entirely (`forcedWorkgroup` was taken verbatim), so an unsupported
  // workgroup size only surfaced as an opaque WebGPU validation error deep
  // inside pipeline/dispatch creation. This must now be validated up front,
  // before any GPU call, with a clear and specific error message.

  it("rejects a forcedWorkgroup the device's negotiated limits don't support", async () => {
    await expect(
      autoTuneDevice({
        device: {} as GPUDevice,
        lutBuffer: {} as GPUBuffer,
        limits: { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 },
        supportedShaders: ["compact"],
        forcedWorkgroup: 512,
      }),
    ).rejects.toThrow(/workgroup size 512 is not supported by this device/i);
  });

  it("rejects a forcedWorkgroup of 1024 the same way", async () => {
    await expect(
      autoTuneDevice({
        device: {} as GPUDevice,
        lutBuffer: {} as GPUBuffer,
        limits: { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 },
        supportedShaders: ["compact"],
        forcedWorkgroup: 1_024,
      }),
    ).rejects.toThrow(/workgroup size 1024 is not supported by this device/i);
  });

  it("still rejects with 'no supported workgroup size' when the device supports none at all, even with no forcedWorkgroup", async () => {
    await expect(
      autoTuneDevice({
        device: {} as GPUDevice,
        lutBuffer: {} as GPUBuffer,
        limits: { maxComputeInvocationsPerWorkgroup: 0, maxComputeWorkgroupSizeX: 0 },
        supportedShaders: ["compact"],
      }),
    ).rejects.toThrow(/no supported workgroup size/i);
  });
});

describe("stableTuningWinner", () => {
  it("matches gpu.rs's stable_winner_prefers_compact_or_cached_incumbent_inside_tie_band test", () => {
    const rows = [row("compact", 64, 100, 0.01), row("expanded", 128, 102, 0.01)];

    const compact = stableTuningWinner(rows, null, false);
    expect(compact).not.toBeNull();
    expect(compact!.key).toEqual({ shader: "compact", workgroupSize: 64 });
    expect(compact!.reason).toBe("deterministic-tie-break");

    const incumbent = stableTuningWinner(rows, { shader: "expanded", workgroupSize: 128 }, false);
    expect(incumbent).not.toBeNull();
    expect(incumbent!.key).toEqual({ shader: "expanded", workgroupSize: 128 });
    expect(incumbent!.reason).toBe("cached-winner-retained");

    const clearRows = [row("compact", 64, 100, 0.01), row("expanded", 128, 120, 0.01)];
    const clear = stableTuningWinner(clearRows, { shader: "compact", workgroupSize: 64 }, false);
    expect(clear).not.toBeNull();
    expect(clear!.key).toEqual({ shader: "expanded", workgroupSize: 128 });
    expect(clear!.reason).toBe("clear-winner");
  });

  it("returns null when there are no valid, positive-rate candidates", () => {
    expect(stableTuningWinner([], null, false)).toBeNull();
    expect(stableTuningWinner([{ ...row("compact", 64, 100, 0), valid: false }], null, false)).toBeNull();
  });

  it("picks the tie-set member minimizing (workgroup_storage_bytes, |workgroup-64|) when neither hysteresis nor Compact+64 applies", () => {
    // Neither candidate is Compact+64, and there's no cached incumbent in the tie set.
    // Expanded/128 has larger workgroup_storage_bytes than Expanded/32, so /32 should win
    // even though it's further from 64 than /128 is (storage bytes dominate the tuple).
    const rows = [row("expanded", 128, 100, 0.005), row("expanded", 32, 99, 0.005)];
    const outcome = stableTuningWinner(rows, null, false);
    expect(outcome).not.toBeNull();
    expect(outcome!.key).toEqual({ shader: "expanded", workgroupSize: 32 });
    expect(outcome!.tied.length).toBe(2);
  });

  it("prefers the workgroup closer to 64 when storage bytes tie", () => {
    const rows = [row("expanded", 256, 100, 0.005), row("expanded", 128, 99, 0.005)];
    const outcome = stableTuningWinner(rows, null, false);
    expect(outcome).not.toBeNull();
    expect(outcome!.key).toEqual({ shader: "expanded", workgroupSize: 128 });
  });

  it("labels a genuine 3+ way tie as deadline-tie-break when the deadline is reached", () => {
    const rows = [
      row("compact", 128, 100, 0.005),
      row("compact", 256, 99.5, 0.005),
      row("expanded", 512, 99, 0.005),
    ];
    const outcome = stableTuningWinner(rows, null, true);
    expect(outcome).not.toBeNull();
    expect(outcome!.tied.length).toBe(3);
    expect(outcome!.reason).toBe("deadline-tie-break");
  });
});
