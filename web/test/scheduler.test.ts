import { describe, expect, it } from "vitest";
import {
  AdaptiveStepScheduler,
  DEFAULT_BUDGET_MS,
  INITIAL_TARGET_STEPS,
  MIN_TARGET_STEPS,
  MAX_TARGET_STEPS,
  MAX_DISPATCH_BUDGET_MS,
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
    const scheduler = new AdaptiveStepScheduler(250);
    // 400_000_000 steps in 1000ms => rate 400_000_000 steps/s.
    scheduler.recordBatch(400_000_000, 1000);
    expect(scheduler.currentSmoothedRate).toBe(400_000_000);
    // target = rate * budgetMs/1000 = 400_000_000 * 0.25 = 100_000_000
    expect(scheduler.currentTargetSteps).toBe(100_000_000);
  });

  it("applies alpha=0.25 smoothing to subsequent measurements", () => {
    const scheduler = new AdaptiveStepScheduler(250);
    scheduler.recordBatch(400_000_000, 1000); // rate = 400_000_000
    scheduler.recordBatch(800_000_000, 1000); // measured = 800_000_000
    // rate = 400_000_000*0.75 + 800_000_000*0.25 = 500_000_000
    expect(scheduler.currentSmoothedRate).toBe(500_000_000);
    // target = 500_000_000 * 0.25 = 125_000_000
    expect(scheduler.currentTargetSteps).toBe(125_000_000);
  });

  it("does not floor the derived target at MIN_TARGET_STEPS", () => {
    const scheduler = new AdaptiveStepScheduler(250);
    // Tiny measured rate. Flooring at MIN_TARGET_STEPS here would mean a
    // dispatch far longer than the budget on slow hardware, which is the
    // dropped-dispatch failure this cap exists to prevent — so the
    // budget-derived value wins instead.
    scheduler.recordBatch(1000, 1000);
    expect(scheduler.currentTargetSteps).toBe(250);
    expect(scheduler.currentTargetSteps).toBeLessThan(MIN_TARGET_STEPS);
  });

  it("still seeds the pre-measurement target at MIN_TARGET_STEPS", () => {
    const scheduler = new AdaptiveStepScheduler(250, 1);
    expect(scheduler.currentTargetSteps).toBe(MIN_TARGET_STEPS);
  });

  it("clamps the derived target to MAX_TARGET_STEPS", () => {
    const scheduler = new AdaptiveStepScheduler(250);
    // Enormous measured rate => desired target far above the ceiling.
    scheduler.recordBatch(1_000_000_000_000, 1);
    expect(scheduler.currentTargetSteps).toBe(MAX_TARGET_STEPS);
  });

  it("reproduces a full budget sequence for a known rate sequence", () => {
    const scheduler = new AdaptiveStepScheduler(250, 256_000_000);
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
      expectedTargets.push(Math.min(Math.floor((rate * 250) / 1000), MAX_TARGET_STEPS));
    }

    const actualTargets = measurements.map(({ batchSteps, elapsedMs }) =>
      scheduler.recordBatch(batchSteps, elapsedMs),
    );
    expect(actualTargets).toEqual(expectedTargets);
  });
});

// A dispatch that runs too long gets dropped by the platform (observed on
// Apple M2 Max / Metal: the marker readback comes back cleared and the run
// dies mid-precompute). These pin the duration ceiling that prevents it.
describe("AdaptiveStepScheduler dispatch-duration ceiling", () => {
  it("caps an over-long requested budget at MAX_DISPATCH_BUDGET_MS", () => {
    const scheduler = new AdaptiveStepScheduler(800);
    expect(scheduler.effectiveBudgetMs).toBe(MAX_DISPATCH_BUDGET_MS);
  });

  it("leaves a budget under the ceiling untouched", () => {
    const scheduler = new AdaptiveStepScheduler(100);
    expect(scheduler.effectiveBudgetMs).toBe(100);
  });

  it("never derives a target implying more than MAX_DISPATCH_BUDGET_MS of work", () => {
    // The M2 Max case: ~1.71B DES steps/s measured, 800ms requested.
    // Pre-fix this produced ~1.37B steps (~800ms/dispatch) and the dispatch
    // was dropped; the ceiling must hold it to ~250ms of work.
    const rate = 1_710_000_000;
    const scheduler = new AdaptiveStepScheduler(800);
    scheduler.recordBatch(rate, 1000);

    const impliedMs = (scheduler.currentTargetSteps / rate) * 1000;
    expect(impliedMs).toBeLessThanOrEqual(MAX_DISPATCH_BUDGET_MS);
    expect(scheduler.currentTargetSteps).toBe(Math.floor(rate * (MAX_DISPATCH_BUDGET_MS / 1000)));
  });

  it("holds the ceiling as the EWMA ratchets up over a long run", () => {
    // The failure appeared 66% into a run, not at the start: the smoothed
    // rate climbs as the GPU warms into the work. Every step of that climb
    // must stay bounded.
    const scheduler = new AdaptiveStepScheduler(800);
    let rate = 200_000_000;
    for (let i = 0; i < 40; i += 1) {
      rate *= 1.2;
      scheduler.recordBatch(rate, 1000);
      const smoothed = scheduler.currentSmoothedRate ?? rate;
      const impliedMs = (scheduler.currentTargetSteps / smoothed) * 1000;
      expect(impliedMs).toBeLessThanOrEqual(MAX_DISPATCH_BUDGET_MS);
    }
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
