// The "continue automatically through remote lookup and local verification"
// checkbox's plumbing. When checked, `runOrchestrator` (from
// `pipeline/orchestrator.ts`) runs straight through every stage. When
// unchecked, this wraps a real `OrchestratorPorts` so the pipeline pauses
// right before each of precompute/lookup/verify, per DES slot, until the UI
// calls `resume()`.
//
// `runOrchestrator`'s structure (after Task 7's fix wave and Task 8's GPU
// verify port) is two passes, not one per-slot loop:
//   pass 1: des1's precompute -> lookup, THEN des2's precompute -> lookup
//           (verify is NOT called here at all);
//   pass 2: des1's verify, THEN des2's verify (only after BOTH slots'
//           lookups in pass 1 have fully resolved).
// `precompute`, `lookup`, and (as of Task 8) `verifyCandidates` are now all
// async port methods `runOrchestrator` awaits, so each gets its own pause
// point below, gating entry to that stage — there is no need to piggyback
// the verify pause onto the end of `lookup` the way an earlier revision of
// this file did (stale once verify moved into its own pass: pausing inside
// `lookup` would have gated des1's verify immediately after des1's lookup,
// long before des1's verify pass actually runs, and would have said nothing
// about des2's verify pause at all). The pause point sits at the very top
// of the wrapped `verifyCandidates`, before any dispatch begins, so
// manual-stepping mode genuinely blocks dispatch rather than pausing
// somewhere mid-run or after the fact.
//
// `precompute`/`lookup`/`verifyCandidates` are each called exactly once per
// DES slot, strictly in des1-then-des2 order within their own pass
// (`runOrchestrator`'s documented sequencing), so a simple per-method call
// counter is enough to label which slot is currently gating — there is no
// need for the orchestrator to pass `which` into the ports themselves.

import type { DesSlot, OrchestratorPorts } from "../pipeline/orchestrator";

export type GateStage = "precompute" | "lookup" | "verify";

export interface GateWaitInfo {
  stage: GateStage;
  which: DesSlot;
}

/**
 * Holds the current "continue automatically" setting and lets callers wait
 * for (or trigger) each stage boundary. Changing the setting to `true` while
 * a wait is pending immediately releases it, matching the checkbox's obvious
 * meaning ("stop pausing from now on").
 */
export class StageGate {
  private continueAutomatically: boolean;
  private pending: { info: GateWaitInfo; resolve: () => void } | null = null;
  private readonly onWaitingChange?: (waiting: GateWaitInfo | null) => void;

  /**
   * `onWaitingChange`, if given, fires synchronously the instant a pause
   * starts or ends — the only way a caller can observe a pause immediately,
   * since (unlike `runOrchestrator`'s own progress events) nothing else
   * fires while a stage is gated: `wait()` is called from inside a wrapped
   * port, strictly before the real port function (and therefore before any
   * progress callback) runs.
   */
  constructor(continueAutomatically: boolean, onWaitingChange?: (waiting: GateWaitInfo | null) => void) {
    this.continueAutomatically = continueAutomatically;
    this.onWaitingChange = onWaitingChange;
  }

  get isContinuingAutomatically(): boolean {
    return this.continueAutomatically;
  }

  /** Current stage waiting for a resume, or `null` if nothing is paused. */
  get waiting(): GateWaitInfo | null {
    return this.pending?.info ?? null;
  }

  setContinueAutomatically(value: boolean): void {
    this.continueAutomatically = value;
    if (value) this.resume();
  }

  /** Resolve once immediately if continuing automatically, else block until `resume()`. */
  wait(stage: GateStage, which: DesSlot): Promise<void> {
    if (this.continueAutomatically) return Promise.resolve();
    return new Promise((resolve) => {
      this.pending = { info: { stage, which }, resolve };
      this.onWaitingChange?.(this.pending.info);
    });
  }

  /** Release whatever stage is currently paused, if any. A no-op if nothing is waiting. */
  resume(): void {
    const current = this.pending;
    if (!current) return;
    this.pending = null;
    current.resolve();
    this.onWaitingChange?.(null);
  }
}

/**
 * Wrap a real `OrchestratorPorts` so `precompute`, `lookup`, and (in its own
 * later pass) `verifyCandidates` each pause on `gate` before running.
 */
export function wrapPortsWithGate(ports: OrchestratorPorts, gate: StageGate): OrchestratorPorts {
  const slotOrder: DesSlot[] = ["des1", "des2"];
  let precomputeCalls = 0;
  let lookupCalls = 0;
  let verifyCalls = 0;

  const slotFor = (callIndex: number): DesSlot => slotOrder[Math.min(callIndex, slotOrder.length - 1)];

  return {
    ...ports,
    precompute: async (target, tuning, onProgress) => {
      const which = slotFor(precomputeCalls);
      precomputeCalls += 1;
      await gate.wait("precompute", which);
      return ports.precompute(target, tuning, onProgress);
    },
    lookup: async (endpointFile, expectedCount, onEvent) => {
      const which = slotFor(lookupCalls);
      lookupCalls += 1;
      await gate.wait("lookup", which);
      return ports.lookup(endpointFile, expectedCount, onEvent);
    },
    verifyCandidates: async (candidates, target, stopAtFirst, tuning, onProgress) => {
      const which = slotFor(verifyCalls);
      verifyCalls += 1;
      await gate.wait("verify", which);
      return ports.verifyCandidates(candidates, target, stopAtFirst, tuning, onProgress);
    },
  };
}
