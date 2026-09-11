// Turns a running count-of-`total` into the percent/elapsed/rate/ETA numbers
// the three-stage progress display shows, given only `(done, total)` pairs
// arriving over time — neither `PrecomputeProgress` nor `VerifyProgress`
// carry a wall-clock start time or a cumulative elapsed duration, only a
// per-batch `elapsedMs`, so this tracks the stage's own start time itself.
// Pure (no DOM), so it's unit-testable with an injected clock.

export interface MeterSnapshot {
  done: number;
  total: number;
  /** `0..100`; `0` when `total` is `0` (nothing to measure yet). */
  percent: number;
  elapsedMs: number;
  /** Units of `done` per second, or `null` before enough time has elapsed to estimate one. */
  rate: number | null;
  /** Estimated remaining time in ms, or `null` when `rate` is unavailable or already at/over `total`. */
  etaMs: number | null;
}

export class ProgressMeter {
  private startedAt: number | null = null;
  private readonly now: () => number;

  constructor(now: () => number = () => Date.now()) {
    this.now = now;
  }

  /** Record a new `(done, total)` reading (the stage's cumulative progress so far) and return the derived snapshot. */
  update(done: number, total: number): MeterSnapshot {
    if (this.startedAt === null) this.startedAt = this.now();
    const elapsedMs = Math.max(this.now() - this.startedAt, 0);
    const elapsedSeconds = elapsedMs / 1000;
    const rate = elapsedSeconds > 0 && done > 0 ? done / elapsedSeconds : null;
    const remaining = Math.max(total - done, 0);
    const etaMs = rate !== null && rate > 0 && remaining > 0 ? (remaining / rate) * 1000 : rate !== null ? 0 : null;
    const percent = total > 0 ? Math.min((done / total) * 100, 100) : 0;
    return { done, total, percent, elapsedMs, rate, etaMs };
  }

  /** Forget the recorded start time, so the next `update()` restarts the clock (a new run of this stage). */
  reset(): void {
    this.startedAt = null;
  }
}
