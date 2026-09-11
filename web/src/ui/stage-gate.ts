// The "continue automatically through remote lookup and local verification"
// checkbox's plumbing. When checked, `runOrchestrator` (from
// `pipeline/orchestrator.ts`) runs straight through every stage. When
// unchecked, this wraps a real `OrchestratorPorts` so the pipeline pauses
// right before each of precompute/lookup/verify, per DES slot, until the UI
// calls `resume()`.
//
// `OrchestratorPorts.precompute`/`lookup` are the only two port methods
// `runOrchestrator` awaits before its next step (`decodeCandidateFile` and
// `verifyCandidates` are synchronous — see `pipeline/orchestrator.ts`), so
// this is the only place a pause can actually be inserted without changing
// that file: pausing at the top of `lookup` gates entry to the lookup
// stage, and pausing again right after `lookup`'s promise resolves (but
// before returning to the orchestrator, which immediately calls
// `decodeCandidateFile`/`verifyCandidates` next) gates entry to the verify
// stage.
//
// `precompute`/`lookup` are called exactly once per DES slot, strictly in
// des1-then-des2 order (`runOrchestrator`'s documented sequencing), so a
// simple call counter is enough to label which slot is currently gating —
// there is no need for the orchestrator to pass `which` into the ports
// themselves.

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
 * Wrap a real `OrchestratorPorts` so `precompute`/`lookup` pause on `gate`
 * before running, and the pipeline additionally pauses on `gate` again right
 * after a slot's lookup resolves (i.e. before verification starts).
 */
export function wrapPortsWithGate(ports: OrchestratorPorts, gate: StageGate): OrchestratorPorts {
  const slotOrder: DesSlot[] = ["des1", "des2"];
  let precomputeCalls = 0;
  let lookupCalls = 0;

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
      const bytes = await ports.lookup(endpointFile, expectedCount, onEvent);
      await gate.wait("verify", which);
      return bytes;
    },
  };
}
