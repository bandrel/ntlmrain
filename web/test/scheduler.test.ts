import { describe, expect, it } from "vitest";
import {
  AdaptiveStepScheduler,
  DEFAULT_BUDGET_MS,
  INITIAL_TARGET_STEPS,
  MIN_TARGET_STEPS,
  MAX_TARGET_STEPS,
  stepsForRange,
  slicedStepsForRange,
  slicedDispatchWidth,
  divCeilU32,
  alignUpU32,
  alignTo,
  maxDispatchInvocations,
} from "../src/webgpu/precompute";

describe("AdaptiveStepScheduler", () => {
  it("starts at the clamped initial target before any measurement", () => {
    const scheduler = new AdaptiveStepScheduler(DEFAULT_BUDGET_MS);
    expect(scheduler.currentTargetSteps).toBe(INITIAL_TARGET_STEPS);
    expect(scheduler.currentSmoothedRate).toBeNull();
  });

  it("seeds the smoothed rate from the first measurement (no prior EWMA state)", () => {
    const scheduler = new AdaptiveStepScheduler(1000);
    // 100_000_000 steps in 1000ms => rate 100_000_000 steps/s.
    scheduler.recordBatch(100_000_000, 1000);
    expect(scheduler.currentSmoothedRate).toBe(100_000_000);
    // target = rate * budgetMs/1000 = 100_000_000 * 1 = 100_000_000
    expect(scheduler.currentTargetSteps).toBe(100_000_000);
  });

  it("applies alpha=0.25 smoothing to subsequent measurements", () => {
    const scheduler = new AdaptiveStepScheduler(1000);
    scheduler.recordBatch(100_000_000, 1000); // rate = 100_000_000
    scheduler.recordBatch(200_000_000, 1000); // measured = 200_000_000
    // rate = 100_000_000*0.75 + 200_000_000*0.25 = 125_000_000
    expect(scheduler.currentSmoothedRate).toBe(125_000_000);
    expect(scheduler.currentTargetSteps).toBe(125_000_000);
  });

  it("clamps the derived target to MIN_TARGET_STEPS", () => {
    const scheduler = new AdaptiveStepScheduler(800);
    // Tiny measured rate => desired target far below the floor.
    scheduler.recordBatch(1000, 1000);
    expect(scheduler.currentTargetSteps).toBe(MIN_TARGET_STEPS);
  });

  it("clamps the derived target to MAX_TARGET_STEPS", () => {
    const scheduler = new AdaptiveStepScheduler(800);
    // Enormous measured rate => desired target far above the ceiling.
    scheduler.recordBatch(1_000_000_000_000, 1);
    expect(scheduler.currentTargetSteps).toBe(MAX_TARGET_STEPS);
  });

  it("reproduces a full budget sequence for a known rate sequence", () => {
    const scheduler = new AdaptiveStepScheduler(800, 256_000_000);
    const measurements: Array<{ batchSteps: number; elapsedMs: number }> = [
      { batchSteps: 80_000_000, elapsedMs: 1000 }, // measured = 80,000,000
      { batchSteps: 120_000_000, elapsedMs: 1000 }, // measured = 120,000,000
      { batchSteps: 100_000_000, elapsedMs: 500 }, // measured = 200,000,000
    ];
    const expectedTargets: number[] = [];
    let rate: number | null = null;
    for (const { batchSteps, elapsedMs } of measurements) {
      const measured = batchSteps / (elapsedMs / 1000);
      rate = rate === null ? measured : rate * 0.75 + measured * 0.25;
      expectedTargets.push(
        Math.min(Math.max(Math.floor((rate * 800) / 1000), MIN_TARGET_STEPS), MAX_TARGET_STEPS),
      );
    }

    const actualTargets = measurements.map(({ batchSteps, elapsedMs }) =>
      scheduler.recordBatch(batchSteps, elapsedMs),
    );
    expect(actualTargets).toEqual(expectedTargets);
  });
});

describe("step-count pure math (ported from src/gpu.rs)", () => {
  it("stepsForRange matches the triangular-number formula", () => {
    expect(stepsForRange(0, 1)).toBe(0);
    expect(stepsForRange(0, 5)).toBe(0 + 1 + 2 + 3 + 4);
    expect(stepsForRange(10, 5)).toBe(10 + 11 + 12 + 13 + 14);
  });

  it("slicedStepsForRange over a checkpoint window covering the whole range equals stepsForRange", () => {
    for (const length of [1, 2, 65_536, 100_000]) {
      expect(slicedStepsForRange(0, length, 0, length)).toBe(stepsForRange(0, length));
    }
  });

  it("slicedStepsForRange caps per-thread cost at sliceSteps once past the ramp", () => {
    // sliceStart=0, sliceSteps=10: threads at position >= 10 each contribute
    // exactly 10 steps (fully saturated), not their full distance-from-end.
    const steps = slicedStepsForRange(10, 5, 0, 10);
    expect(steps).toBe(5 * 10);
  });

  it("divCeilU32/alignUpU32/alignTo round as expected", () => {
    expect(divCeilU32(10, 3)).toBe(4);
    expect(divCeilU32(9, 3)).toBe(3);
    expect(alignUpU32(65_535, 64)).toBe(65_536);
    expect(alignUpU32(65_536, 64)).toBe(65_536);
    expect(alignTo(9, 8)).toBe(16);
  });

  it("maxDispatchInvocations clamps to the host cap and the workgroup floor", () => {
    expect(maxDispatchInvocations(1_000_000, 64)).toBe(65_536); // host cap wins
    expect(maxDispatchInvocations(1, 128)).toBe(128); // workgroup floor wins
  });

  it("slicedDispatchWidth finds the widest aligned length within budget", () => {
    // sliceStart=0, sliceSteps=1000 (whole remaining range fits the ramp).
    // Total cost for length L starting at 0 is L*(L-1)/2 (triangular).
    // Find the largest L (aligned to 32) with L*(L-1)/2 <= 5000.
    const length = slicedDispatchWidth(0, 1000, 0, 1000, 5000, 1000, 32);
    const cost = slicedStepsForRange(0, length, 0, 1000);
    expect(cost).toBeLessThanOrEqual(5000);
    expect(length % 32).toBe(0);
    // The next aligned step up must exceed the budget (or hit the maximum).
    const next = length + 32;
    if (next <= 1000) {
      expect(slicedStepsForRange(0, next, 0, 1000)).toBeGreaterThan(5000);
    }
  });

  it("slicedDispatchWidth never exceeds remaining or maximumWidth", () => {
    const length = slicedDispatchWidth(500, 10, 0, 1000, 1_000_000, 200, 32);
    expect(length).toBeLessThanOrEqual(10);
  });
});
