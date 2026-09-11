// The UI-facing state machine: turns `runOrchestrator`'s event stream (Task
// 5) into a renderable snapshot, drives the "continue automatically"
// pause/resume gate (`stage-gate.ts`), and archives the finished run
// (`archive/db.ts`). No DOM here — `ui/dom/*.ts` renders a snapshot,
// `main.ts` wires DOM events back into this controller's public methods.

import {
  runOrchestrator,
  type DesSlot,
  type LookupEvent,
  type OrchestratorPorts,
  type OrchestratorResult,
} from "../pipeline/orchestrator";
import type { PrecomputeProgress } from "../webgpu/precompute";
import type { VerifyProgress } from "../pipeline/orchestrator";
import { StageGate, wrapPortsWithGate, type GateWaitInfo } from "./stage-gate";
import type { RecoveryMode } from "../archive/db";

export interface SlotSnapshot {
  precompute: PrecomputeProgress | null;
  lookupEvent: LookupEvent | null;
  verify: VerifyProgress | null;
}

export type RunStatus = "idle" | "running" | "paused" | "done" | "no-match" | "error";

export interface RunSnapshot {
  status: RunStatus;
  mode: RecoveryMode;
  waiting: GateWaitInfo | null;
  slots: { des1: SlotSnapshot; des2: SlotSnapshot | null };
  result: OrchestratorResult | null;
  errorMessage: string | null;
}

function emptySlot(): SlotSnapshot {
  return { precompute: null, lookupEvent: null, verify: null };
}

function idleSnapshot(mode: RecoveryMode): RunSnapshot {
  return {
    status: "idle",
    mode,
    waiting: null,
    slots: { des1: emptySlot(), des2: null },
    result: null,
    errorMessage: null,
  };
}

export interface RunControllerOptions {
  /** Injectable for tests; defaults to the real `runOrchestrator`. */
  run?: typeof runOrchestrator;
}

/**
 * Drives one recovery run at a time. `start` resolves once the run (and its
 * archive write, if `onFinished` is supplied) is fully done; `subscribe`
 * receives every intermediate snapshot along the way.
 */
export class RunController {
  private readonly runFn: typeof runOrchestrator;
  private readonly listeners = new Set<(snapshot: RunSnapshot) => void>();
  private gate: StageGate | null = null;
  private snapshot: RunSnapshot;

  constructor(mode: RecoveryMode, options: RunControllerOptions = {}) {
    this.runFn = options.run ?? runOrchestrator;
    this.snapshot = idleSnapshot(mode);
  }

  subscribe(listener: (snapshot: RunSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.snapshot);
    return () => this.listeners.delete(listener);
  }

  get current(): RunSnapshot {
    return this.snapshot;
  }

  /** Toggle "continue automatically". Releases a pending pause immediately when turned on. */
  setContinueAutomatically(value: boolean): void {
    this.gate?.setContinueAutomatically(value);
  }

  /** Resume from a manual pause (no-op if nothing is paused, or continuing automatically). */
  resume(): void {
    this.gate?.resume();
  }

  private emit(patch: Partial<RunSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const listener of this.listeners) listener(this.snapshot);
  }

  private slotPatch(which: DesSlot, patch: Partial<SlotSnapshot>): RunSnapshot["slots"] {
    const existing = which === "des1" ? this.snapshot.slots.des1 : (this.snapshot.slots.des2 ?? emptySlot());
    const updated = { ...existing, ...patch };
    return which === "des1"
      ? { ...this.snapshot.slots, des1: updated }
      : { ...this.snapshot.slots, des2: updated };
  }

  /**
   * Run `input` against `ports` (a real `createDefaultPorts(...)` in
   * production). `continueAutomatically` seeds the gate's initial setting;
   * flip it later via `setContinueAutomatically`.
   */
  async start(
    input: string,
    ports: OrchestratorPorts,
    continueAutomatically: boolean,
  ): Promise<OrchestratorResult | null> {
    this.gate = new StageGate(continueAutomatically, (waiting) => {
      // A pause/resume can happen mid-run (see `StageGate`'s docs: it fires
      // synchronously, independent of any orchestrator event), but never
      // after a terminal status (done/no-match/error) has already been set.
      if (this.snapshot.status === "done" || this.snapshot.status === "no-match" || this.snapshot.status === "error") {
        return;
      }
      this.emit({ status: waiting ? "paused" : "running", waiting });
    });
    const wrapped = wrapPortsWithGate(ports, this.gate);
    this.emit({
      status: "running",
      waiting: null,
      slots: { des1: emptySlot(), des2: null },
      result: null,
      errorMessage: null,
    });

    try {
      const result = await this.runFn(input, wrapped, {
        onEvent: (event) => {
          switch (event.type) {
            case "precompute-progress":
              this.emit({ slots: this.slotPatch(event.which, { precompute: event.progress }) });
              break;
            case "lookup":
              this.emit({ slots: this.slotPatch(event.which, { lookupEvent: event.event }) });
              break;
            case "verify-progress":
              this.emit({ slots: this.slotPatch(event.which, { verify: event.progress }) });
              break;
            case "parsed":
            case "tuning":
              break;
            case "result":
              this.emit({ status: "done", waiting: null, result: event.result });
              break;
            case "no-match":
              this.emit({ status: "no-match", waiting: null });
              break;
          }
        },
      });
      return result;
    } catch (error) {
      this.emit({
        status: "error",
        waiting: null,
        errorMessage: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }
}
