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
import type { StatusResponse } from "../api/lookup-client";
import { StageGate, wrapPortsWithGate, type GateWaitInfo } from "./stage-gate";
import type { RecoveryMode } from "../archive/db";

export interface SlotSnapshot {
  precompute: PrecomputeProgress | null;
  lookupEvent: LookupEvent | null;
  /**
   * Last `status` seen for this slot's lookup, retained across the events that
   * carry none. `LookupEvent.status` rides on the `status` variant alone, so a
   * consumer reading it off `lookupEvent` loses the counts the moment the
   * terminal `downloaded` event lands — which is what used to empty the
   * progress bar to "0 / 0 endpoints" right as the lookup finished.
   */
  lookupStatus: StatusResponse | null;
  /**
   * Whether the result bytes are in hand. Tracked separately rather than
   * inferred from `lookupStatus`, because the last `ready` status is allowed
   * to under-report: the server only live-queries `processed_records` while a
   * job is running, and once it is ready serves the ~2s-stale checkpointed
   * column instead. Treating "counts agree" as the completion signal would
   * park a finished lookup just short of 100%.
   */
  lookupComplete: boolean;
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
  return { precompute: null, lookupEvent: null, lookupStatus: null, lookupComplete: false, verify: null };
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
            case "lookup": {
              // Only carry `lookupStatus`/`lookupComplete` forward on the
              // events that actually establish them; `slotPatch` spread-merges,
              // so an omitted key keeps its previous value rather than
              // clobbering it with the `undefined` the other variants hold.
              const patch: Partial<SlotSnapshot> = { lookupEvent: event.event };
              if (event.event.status) patch.lookupStatus = event.event.status;
              if (event.event.type === "downloaded") patch.lookupComplete = true;
              this.emit({ slots: this.slotPatch(event.which, patch) });
              break;
            }
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
