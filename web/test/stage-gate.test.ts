import { describe, expect, it } from "vitest";
import { StageGate, wrapPortsWithGate } from "../src/ui/stage-gate";
import type { OrchestratorPorts } from "../src/pipeline/orchestrator";
import type { TuningSelection } from "../src/webgpu/tuning";

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

function fakePorts(): { ports: OrchestratorPorts; calls: string[] } {
  const calls: string[] = [];
  const ports: OrchestratorPorts = {
    getTuning: async () => baseTuning(),
    precompute: async () => {
      calls.push("precompute");
      return new BigUint64Array([1n]);
    },
    encodeEndpointFile: () => new Uint8Array(0),
    lookup: async () => {
      calls.push("lookup");
      return new Uint8Array(0);
    },
    decodeCandidateFile: () => [],
    verifyCandidates: () => {
      calls.push("verifyCandidates");
      return { keys: [] };
    },
    recoverPt3: () => null,
    assembleNtHash: () => new Uint8Array(0),
  };
  return { ports, calls };
}

describe("StageGate", () => {
  it("resolves immediately when continuing automatically", async () => {
    const gate = new StageGate(true);
    await expect(gate.wait("precompute", "des1")).resolves.toBeUndefined();
    expect(gate.waiting).toBeNull();
  });

  it("blocks until resume() is called when not continuing automatically", async () => {
    const gate = new StageGate(false);
    let resolved = false;
    const promise = gate.wait("lookup", "des2").then(() => {
      resolved = true;
    });
    // Give any microtasks a chance to run; the promise must still be pending.
    await Promise.resolve();
    expect(resolved).toBe(false);
    expect(gate.waiting).toEqual({ stage: "lookup", which: "des2" });

    gate.resume();
    await promise;
    expect(resolved).toBe(true);
    expect(gate.waiting).toBeNull();
  });

  it("resume() is a no-op when nothing is waiting", () => {
    const gate = new StageGate(false);
    expect(() => gate.resume()).not.toThrow();
    expect(gate.waiting).toBeNull();
  });

  it("switching to continue-automatically releases a pending wait", async () => {
    const gate = new StageGate(false);
    let resolved = false;
    const promise = gate.wait("verify", "des1").then(() => {
      resolved = true;
    });
    await Promise.resolve();
    expect(resolved).toBe(false);

    gate.setContinueAutomatically(true);
    await promise;
    expect(resolved).toBe(true);

    // Subsequent waits resolve immediately too.
    await expect(gate.wait("precompute", "des2")).resolves.toBeUndefined();
  });
});

describe("wrapPortsWithGate", () => {
  it("pauses at precompute, then lookup, then verify, labeling des1 before des2", async () => {
    const gate = new StageGate(false);
    const { ports, calls } = fakePorts();
    const wrapped = wrapPortsWithGate(ports, gate);

    // --- des1's precompute ---
    const precomputePromise = wrapped.precompute(new Uint8Array(8), baseTuning());
    await Promise.resolve();
    expect(calls).toEqual([]);
    expect(gate.waiting).toEqual({ stage: "precompute", which: "des1" });
    gate.resume();
    await precomputePromise;
    expect(calls).toEqual(["precompute"]);

    // --- des1's lookup, then its post-lookup verify gate ---
    const lookupPromise = wrapped.lookup(new Uint8Array(0), 1);
    await Promise.resolve();
    expect(gate.waiting).toEqual({ stage: "lookup", which: "des1" });
    gate.resume();
    await Promise.resolve();
    await Promise.resolve();
    expect(calls).toEqual(["precompute", "lookup"]);
    expect(gate.waiting).toEqual({ stage: "verify", which: "des1" });
    gate.resume();
    await lookupPromise;

    // --- des2's precompute is labeled des2, not des1 ---
    const secondPrecompute = wrapped.precompute(new Uint8Array(8), baseTuning());
    await Promise.resolve();
    expect(gate.waiting).toEqual({ stage: "precompute", which: "des2" });
    gate.resume();
    await secondPrecompute;
  });

  it("never pauses when continuing automatically, and calls through in order", async () => {
    const gate = new StageGate(true);
    const { ports, calls } = fakePorts();
    const wrapped = wrapPortsWithGate(ports, gate);

    await wrapped.precompute(new Uint8Array(8), baseTuning());
    await wrapped.lookup(new Uint8Array(0), 1);
    expect(calls).toEqual(["precompute", "lookup"]);
  });
});
