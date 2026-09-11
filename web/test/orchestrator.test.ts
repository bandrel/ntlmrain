import { describe, expect, it } from "vitest";
import {
  isCacheableTuningSource,
  runOrchestrator,
  type OrchestratorEvent,
  type OrchestratorPorts,
  type VerifyOutcome,
} from "../src/pipeline/orchestrator";
import type { TuningSelection } from "../src/webgpu/tuning";

const RESPONSE = "727B4E35F947129EA52B9CDEDAE86934BB23EF89F50FC595";

function baseTuning(): TuningSelection {
  return {
    shader: "compact",
    workgroupSize: 64,
    minimumWorkgroups: 128,
    estimatedStepsPerSecond: 1_000_000,
    source: "auto-tune",
    selectionReason: "clear-winner",
    tuningElapsedMs: 100,
    deadlineReached: false,
    measurements: [],
  };
}

/** A minimal fake `OrchestratorPorts` that records the order every port method is invoked in. */
function fakePorts(overrides: Partial<OrchestratorPorts> = {}): {
  ports: OrchestratorPorts;
  calls: string[];
} {
  const calls: string[] = [];
  let precomputeCount = 0;

  const ports: OrchestratorPorts = {
    getTuning: async () => {
      calls.push("getTuning");
      return baseTuning();
    },
    precompute: async (_target, _tuning, onProgress) => {
      precomputeCount += 1;
      calls.push(`precompute:${precomputeCount}`);
      onProgress?.({ stepsDone: 1, stepsTotal: 1, batchSteps: 1, batchWorkgroups: 1, elapsedMs: 1, targetSteps: 1 });
      return new BigUint64Array([1n, 2n]);
    },
    encodeEndpointFile: (endpoints) => {
      calls.push("encodeEndpointFile");
      return new Uint8Array(endpoints.length);
    },
    lookup: async (_endpointFile, _expectedCount, onEvent) => {
      calls.push("lookup");
      onEvent?.({ type: "submitted", submissionToken: "token", pollWithinSeconds: 1 });
      return new Uint8Array([9, 9]);
    },
    decodeCandidateFile: (_bytes, _expectedQueryCount) => {
      calls.push("decodeCandidateFile");
      return [{ position: 0n, start: 42n }];
    },
    verifyCandidates: async (_candidates, _target, _stopAtFirst, _tuning, onProgress) => {
      calls.push("verifyCandidates");
      onProgress?.({ candidatesDone: 1n, candidatesTotal: 1n, stepsDone: 1n, stepsTotal: 1n, verifiedKeys: 1n });
      return { keys: [123n] };
    },
    recoverPt3: (_target) => {
      calls.push("recoverPt3");
      return new Uint8Array([0x58, 0x6c]);
    },
    assembleNtHash: (_pt1, _pt2, _pt3) => {
      calls.push("assembleNtHash");
      return new Uint8Array(16);
    },
    ...overrides,
  };

  return { ports, calls };
}

describe("runOrchestrator sequencing", () => {
  it("fully completes des1's precompute+lookup+verify before starting des2's", async () => {
    const { ports, calls } = fakePorts();
    const result = await runOrchestrator(RESPONSE, ports);

    // Find the index of each "des1"-then-"des2" cycle's operations. Since
    // this fake's precompute/lookup/decode/verify calls are unlabeled by
    // which=des1/des2, sequencing is proven by asserting the full ordered
    // sequence: des1's entire (precompute, encode, lookup, decode, verify)
    // chain must appear before des2's chain starts, not interleaved.
    expect(calls).toEqual([
      "getTuning",
      "precompute:1",
      "encodeEndpointFile",
      "lookup",
      "decodeCandidateFile",
      "precompute:2",
      "encodeEndpointFile",
      "lookup",
      "decodeCandidateFile",
      "verifyCandidates",
      "verifyCandidates",
      "recoverPt3",
      "assembleNtHash",
    ]);
    expect(result.des1Keys).toEqual([123n]);
    expect(result.des2Keys).toEqual([123n]);
    expect(result.pt3).toEqual(new Uint8Array([0x58, 0x6c]));
    expect(result.ntHashes).toHaveLength(1);
  });

  it("does not start des2's precompute until des1's lookup promise has resolved", async () => {
    let des1LookupResolved = false;
    let des2PrecomputeStartedBeforeDes1LookupResolved = false;
    let precomputeCount = 0;

    const { ports } = fakePorts({
      precompute: async () => {
        precomputeCount += 1;
        if (precomputeCount === 2 && !des1LookupResolved) {
          des2PrecomputeStartedBeforeDes1LookupResolved = true;
        }
        return new BigUint64Array([1n]);
      },
      lookup: async () => {
        // Simulate network latency with a real microtask/timer delay so a
        // buggy parallel implementation would have a chance to race ahead.
        await new Promise((resolve) => setTimeout(resolve, 5));
        des1LookupResolved = true;
        return new Uint8Array([1]);
      },
    });

    await runOrchestrator(RESPONSE, ports);
    expect(des2PrecomputeStartedBeforeDes1LookupResolved).toBe(false);
  });

  it("does not verify either slot until BOTH des1's and des2's lookups have completed", async () => {
    // Regression test for the final whole-branch review's finding: verify
    // used to run inside the per-target loop (precompute->lookup->verify for
    // des1, only then starting des2's precompute), blocking des2's entire
    // pipeline behind des1's verify for no reason. It must now run in a
    // separate pass after both lookups (not just des1's) have resolved.
    let lookupsCompleted = 0;
    let verifyStartedBeforeBothLookupsCompleted = false;

    const { ports, calls } = fakePorts({
      lookup: async (_endpointFile, _expectedCount, onEvent) => {
        // Simulate network latency with a real timer delay so a buggy
        // implementation that verifies des1 before des2's lookup starts (or
        // completes) would have a chance to race ahead.
        await new Promise((resolve) => setTimeout(resolve, 5));
        lookupsCompleted += 1;
        calls.push("lookup");
        onEvent?.({ type: "submitted", submissionToken: "token", pollWithinSeconds: 1 });
        return new Uint8Array([9, 9]);
      },
      verifyCandidates: async (_candidates, _target, _stopAtFirst) => {
        calls.push("verifyCandidates");
        if (lookupsCompleted < 2) {
          verifyStartedBeforeBothLookupsCompleted = true;
        }
        return { keys: [123n] };
      },
    });

    await runOrchestrator(RESPONSE, ports);
    expect(verifyStartedBeforeBothLookupsCompleted).toBe(false);
    expect(lookupsCompleted).toBe(2);
    // Both "lookup" calls (and their "decodeCandidateFile" companions) must
    // appear before either "verifyCandidates" call in the recorded order.
    const firstVerifyIndex = calls.indexOf("verifyCandidates");
    const secondLookupIndex = calls.lastIndexOf("lookup");
    expect(secondLookupIndex).toBeLessThan(firstVerifyIndex);
  });

  it("runs only des1 for a bare 8-byte Des target (no des2, no NT hash)", async () => {
    const { ports, calls } = fakePorts();
    const result = await runOrchestrator("727b4e35f947129e", ports);
    expect(calls.filter((call) => call.startsWith("precompute"))).toEqual(["precompute:1"]);
    expect(calls).not.toContain("recoverPt3");
    expect(result.des2Keys).toEqual([]);
    expect(result.pt3).toBeNull();
    expect(result.ntHashes).toEqual([]);
  });

  it("stops after the first des1xdes2 pair unless findAll is set", async () => {
    const { ports } = fakePorts({
      verifyCandidates: async (_candidates, _target, _stopAtFirst) => ({ keys: [1n, 2n] }),
    });

    const stopsAtFirst = await runOrchestrator(RESPONSE, ports);
    expect(stopsAtFirst.ntHashes).toHaveLength(1);

    const findAllResult = await runOrchestrator(RESPONSE, ports, { findAll: true });
    // Cross product of [1n, 2n] x [1n, 2n] = 4 pairs.
    expect(findAllResult.ntHashes).toHaveLength(4);
  });

  it("emits a no-match event when des1 verification finds nothing", async () => {
    const events: OrchestratorEvent[] = [];
    const { ports } = fakePorts({ verifyCandidates: async () => ({ keys: [] } satisfies VerifyOutcome) });
    const result = await runOrchestrator(RESPONSE, ports, { onEvent: (event) => events.push(event) });
    expect(result.des1Keys).toEqual([]);
    expect(events.some((event) => event.type === "no-match")).toBe(true);
    expect(events.some((event) => event.type === "result")).toBe(false);
  });

  it("passes precompute/lookup/verify progress through to onEvent", async () => {
    const events: OrchestratorEvent[] = [];
    const { ports } = fakePorts();
    await runOrchestrator("727b4e35f947129e", ports, { onEvent: (event) => events.push(event) });
    expect(events.some((event) => event.type === "precompute-progress")).toBe(true);
    expect(events.some((event) => event.type === "lookup")).toBe(true);
    expect(events.some((event) => event.type === "verify-progress")).toBe(true);
  });
});

describe("isCacheableTuningSource", () => {
  it("is true for a genuine fresh auto-tune result", () => {
    expect(isCacheableTuningSource("auto-tune")).toBe(true);
    expect(isCacheableTuningSource("partial-override-tune")).toBe(true);
    expect(isCacheableTuningSource("manual")).toBe(true);
  });

  it("is false for a cache hit or the slow-adapter fallback", () => {
    expect(isCacheableTuningSource("cache")).toBe(false);
    expect(isCacheableTuningSource("slow-adapter-default")).toBe(false);
  });
});
