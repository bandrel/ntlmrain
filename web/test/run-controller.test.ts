import { describe, expect, it } from "vitest";
import { RunController, type RunSnapshot } from "../src/ui/run-controller";
import type { OrchestratorPorts } from "../src/pipeline/orchestrator";
import type { TuningSelection } from "../src/webgpu/tuning";

// Single DES ciphertext: skips des2/des3/NT-hash assembly (this task's
// "Single DES ciphertext" recovery mode), so only one precompute/lookup/
// verify cycle runs — keeps these state-machine tests short.
const SINGLE_DES = "AABBCCDD11223344";

function baseTuning(): TuningSelection {
  return {
    shader: "compact",
    workgroupSize: 64,
    minimumWorkgroups: 128,
    estimatedStepsPerSecond: 1,
    source: "auto-tune",
    selectionReason: "clear-winner",
    tuningElapsedMs: 0,
    deadlineReached: false,
    measurements: [],
  };
}

function fakePorts(): OrchestratorPorts {
  return {
    getTuning: async () => baseTuning(),
    precompute: async (_target, _tuning, onProgress) => {
      onProgress?.({ stepsDone: 1, stepsTotal: 1, batchSteps: 1, batchWorkgroups: 1, elapsedMs: 1, targetSteps: 1 });
      return new BigUint64Array([1n]);
    },
    encodeEndpointFile: () => new Uint8Array(0),
    lookup: async (_endpointFile, _expectedCount, onEvent) => {
      onEvent?.({ type: "submitted", submissionToken: "token", pollWithinSeconds: 1 });
      return new Uint8Array(0);
    },
    decodeCandidateFile: () => [{ position: 0n, start: 1n }],
    verifyCandidates: (_candidates, _target, _stopAtFirst, onProgress) => {
      onProgress?.({ candidatesDone: 1n, candidatesTotal: 1n, stepsDone: 1n, stepsTotal: 1n, verifiedKeys: 1n });
      return { keys: [42n] };
    },
    recoverPt3: () => null,
    assembleNtHash: () => new Uint8Array(0),
  };
}

/** Flush enough microtask turns for a chain of awaited promises to settle. */
async function flush(turns = 10): Promise<void> {
  for (let i = 0; i < turns; i += 1) {
    await Promise.resolve();
  }
}

function noMatchPorts(): OrchestratorPorts {
  return { ...fakePorts(), verifyCandidates: () => ({ keys: [] }) };
}

describe("RunController", () => {
  it("goes idle -> running -> done, recording des1's progress along the way", async () => {
    const controller = new RunController("single-des");
    const snapshots: RunSnapshot[] = [];
    controller.subscribe((snapshot) => snapshots.push(snapshot));

    expect(snapshots[0].status).toBe("idle");

    const result = await controller.start(SINGLE_DES, fakePorts(), true);

    expect(result?.des1Keys).toEqual([42n]);
    const statuses = snapshots.map((snapshot) => snapshot.status);
    expect(statuses[0]).toBe("idle");
    expect(statuses).toContain("running");
    expect(statuses.at(-1)).toBe("done");

    const finalSnapshot = controller.current;
    expect(finalSnapshot.slots.des1.precompute?.stepsDone).toBe(1);
    expect(finalSnapshot.slots.des1.lookupEvent?.type).toBe("submitted");
    expect(finalSnapshot.slots.des1.verify?.verifiedKeys).toBe(1n);
    expect(finalSnapshot.result?.des1Keys).toEqual([42n]);
  });

  it("reaches the no-match status when verification finds nothing", async () => {
    const controller = new RunController("single-des");
    const result = await controller.start(SINGLE_DES, noMatchPorts(), true);
    expect(result?.des1Keys).toEqual([]);
    expect(controller.current.status).toBe("no-match");
  });

  it("pauses before each stage when continueAutomatically is false, and resume() advances it", async () => {
    const controller = new RunController("single-des");
    const statuses: string[] = [];
    controller.subscribe((snapshot) => statuses.push(snapshot.status));

    const runPromise = controller.start(SINGLE_DES, fakePorts(), false);

    // First pause should be before precompute.
    await flush();
    expect(controller.current.status).toBe("paused");
    expect(controller.current.waiting).toEqual({ stage: "precompute", which: "des1" });
    controller.resume();

    await flush();
    expect(controller.current.waiting).toEqual({ stage: "lookup", which: "des1" });
    controller.resume();

    await flush();
    expect(controller.current.waiting).toEqual({ stage: "verify", which: "des1" });
    controller.resume();

    const result = await runPromise;
    expect(result?.des1Keys).toEqual([42n]);
    expect(controller.current.status).toBe("done");
    expect(statuses).toContain("paused");
  });

  it("switching to continue-automatically mid-run releases the current pause and skips future ones", async () => {
    const controller = new RunController("single-des");
    const runPromise = controller.start(SINGLE_DES, fakePorts(), false);

    await flush();
    expect(controller.current.status).toBe("paused");

    controller.setContinueAutomatically(true);
    const result = await runPromise;
    expect(result?.des1Keys).toEqual([42n]);
    expect(controller.current.status).toBe("done");
  });

  it("surfaces a thrown error as status 'error' with its message, and rethrows", async () => {
    const controller = new RunController("single-des");
    const failingPorts: OrchestratorPorts = {
      ...fakePorts(),
      getTuning: async () => {
        throw new Error("device lost");
      },
    };
    await expect(controller.start(SINGLE_DES, failingPorts, true)).rejects.toThrow("device lost");
    expect(controller.current.status).toBe("error");
    expect(controller.current.errorMessage).toBe("device lost");
  });
});
