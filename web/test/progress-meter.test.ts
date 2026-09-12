import { describe, expect, it } from "vitest";
import { ProgressMeter } from "../src/ui/progress-meter";

describe("ProgressMeter", () => {
  it("starts its clock on the first update() and reports elapsed time relative to it", () => {
    let now = 1_000;
    const meter = new ProgressMeter(() => now);

    const first = meter.update(0, 100);
    expect(first.elapsedMs).toBe(0);
    expect(first.percent).toBe(0);
    expect(first.rate).toBeNull();
    expect(first.etaMs).toBeNull();

    now = 1_500;
    const second = meter.update(50, 100);
    expect(second.elapsedMs).toBe(500);
    expect(second.percent).toBe(50);
    expect(second.rate).toBeCloseTo(100, 5); // 50 done / 0.5s = 100/s
    expect(second.etaMs).toBeCloseTo(500, 5); // 50 remaining / 100 per s
  });

  it("reports 100% and a zero ETA once done reaches total", () => {
    let now = 0;
    const meter = new ProgressMeter(() => now);
    meter.update(0, 10);
    now = 1_000;
    const done = meter.update(10, 10);
    expect(done.percent).toBe(100);
    expect(done.etaMs).toBe(0);
  });

  it("treats total=0 as 0% rather than dividing by zero", () => {
    const meter = new ProgressMeter(() => 0);
    const snapshot = meter.update(0, 0);
    expect(snapshot.percent).toBe(0);
    expect(snapshot.rate).toBeNull();
  });

  it("reset() restarts the elapsed-time clock on the next update()", () => {
    let now = 0;
    const meter = new ProgressMeter(() => now);
    meter.update(0, 10);
    now = 5_000;
    meter.update(5, 10);

    meter.reset();
    now = 5_100;
    const afterReset = meter.update(0, 10);
    expect(afterReset.elapsedMs).toBe(0);
  });
});
