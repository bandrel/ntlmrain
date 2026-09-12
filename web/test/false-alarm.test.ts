import { describe, expect, it } from "vitest";
import {
  STATE_BYTES,
  FALSE_PARAMS_BYTES,
  COMPLETION_FOUND_MAGIC,
  gcd,
  lcm,
  falseAlarmBatchCapacity,
  sortCandidatesDescendingByPosition,
  initCandidateStates,
  encodeCandidateStates,
  decodeCandidateStates,
  buildFalseParamsBuffer,
  validateFalseAlarmMarkers,
  initialStepBudget,
  retargetStepBudget,
  trimRegularTail,
  advanceCandidateMirrorWithoutHit,
  compactCandidateStates,
  processCandidateHits,
  FALSE_ALARM_WORKGROUP_STORAGE_BYTES,
  type CandidateStateFields,
  type FalseAlarmCandidateInput,
  type FalseAlarmDeviceLimits,
} from "../src/webgpu/false-alarm";
import { COMPLETION_MAGIC } from "../src/webgpu/precompute";

function state(overrides: Partial<CandidateStateFields> = {}): CandidateStateFields {
  return {
    indexLo: 0,
    indexHi: 0,
    resultLo: 0,
    resultHi: 0,
    targetPosition: 0,
    nextPosition: 0,
    found: 0,
    ...overrides,
  };
}

describe("byte layouts", () => {
  it("CandidateState is exactly 32 bytes/record, fields at documented offsets", () => {
    expect(STATE_BYTES).toBe(32);
    const buffer = encodeCandidateStates([
      state({ indexLo: 0x11223344, indexHi: 0x55667788, resultLo: 0x99aabbcc, resultHi: 0xddeeff00 }),
    ]);
    expect(buffer.byteLength).toBe(32);
    const view = new DataView(buffer);
    expect(view.getUint32(0, true)).toBe(0x11223344); // index.lo
    expect(view.getUint32(4, true)).toBe(0x55667788); // index.hi
    expect(view.getUint32(8, true)).toBe(0x99aabbcc); // result.lo
    expect(view.getUint32(12, true)).toBe(0xddeeff00); // result.hi
    expect(view.getUint32(16, true)).toBe(0); // target_position
    expect(view.getUint32(20, true)).toBe(0); // next_position
    expect(view.getUint32(24, true)).toBe(0); // found
    expect(view.getUint32(28, true)).toBe(0); // padding
  });

  it("round-trips through encode/decode", () => {
    const states = [
      state({ indexLo: 1, indexHi: 2, resultLo: 3, resultHi: 4, targetPosition: 5, nextPosition: 6, found: 1 }),
      state({ indexLo: 7, indexHi: 8, targetPosition: 9 }),
    ];
    const bytes = new Uint8Array(encodeCandidateStates(states));
    expect(decodeCandidateStates(bytes, 2)).toEqual(states);
  });

  it("FalseParams is exactly 32 bytes, fields at documented offsets", () => {
    expect(FALSE_PARAMS_BYTES).toBe(32);
    const buffer = buildFalseParamsBuffer({
      targetLo: 0x11223344,
      targetHi: 0x55667788,
      reductionOffset: 65_536,
      candidateCount: 4_096,
      stepBudget: 64,
    });
    expect(buffer.byteLength).toBe(32);
    const view = new DataView(buffer);
    expect(view.getUint32(0, true)).toBe(0x11223344); // target_lo
    expect(view.getUint32(4, true)).toBe(0x55667788); // target_hi
    expect(view.getUint32(8, true)).toBe(65_536); // reduction_offset
    expect(view.getUint32(12, true)).toBe(4_096); // candidate_count
    expect(view.getUint32(16, true)).toBe(64); // step_budget
    expect(view.getUint32(20, true)).toBe(0); // padding0
    expect(view.getUint32(24, true)).toBe(0); // padding1
    expect(view.getUint32(28, true)).toBe(0); // padding2
  });
});

describe("completion magics", () => {
  it("match src/gpu.rs's COMPLETION_MAGIC/COMPLETION_FOUND_MAGIC exactly", () => {
    expect(COMPLETION_MAGIC).toBe(0x4259_3731);
    expect(COMPLETION_FOUND_MAGIC).toBe(0x4259_3732);
  });
});

describe("validateFalseAlarmMarkers", () => {
  it("returns false when every marker is the plain completion magic", () => {
    const words = new Uint32Array([COMPLETION_MAGIC, COMPLETION_MAGIC, COMPLETION_MAGIC]);
    expect(validateFalseAlarmMarkers(words, 3, true)).toBe(false);
  });

  it("returns true when any marker is the found magic and allowFound is true", () => {
    const words = new Uint32Array([COMPLETION_MAGIC, COMPLETION_FOUND_MAGIC, COMPLETION_MAGIC]);
    expect(validateFalseAlarmMarkers(words, 3, true)).toBe(true);
  });

  it("throws on the found magic when allowFound is false", () => {
    const words = new Uint32Array([COMPLETION_FOUND_MAGIC]);
    expect(() => validateFalseAlarmMarkers(words, 1, false)).toThrow();
  });

  it("throws on any unexpected value or a short readback", () => {
    expect(() => validateFalseAlarmMarkers(new Uint32Array([0xdead_beef]), 1, true)).toThrow();
    expect(() => validateFalseAlarmMarkers(new Uint32Array([]), 1, true)).toThrow();
  });
});

describe("gcd/lcm", () => {
  it("computes gcd/lcm as expected", () => {
    expect(gcd(12, 18)).toBe(6);
    expect(gcd(64, 256)).toBe(64);
    expect(lcm(64, 4)).toBe(64);
    expect(lcm(64, 256)).toBe(256);
  });
});

describe("falseAlarmBatchCapacity", () => {
  const limits: FalseAlarmDeviceLimits = {
    maxComputeWorkgroupsPerDimension: 65_535,
    minStorageBufferOffsetAlignment: 256,
    maxBufferSize: 1 << 30,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
  };

  it("matches a hand-computed capacity for a typical device", () => {
    // offsetCandidateAlignment = 256 / gcd(256, 32) = 256/32 = 8
    // alignment = lcm(64, 8) = 64
    // raw = min(65535*64, 128MiB/32) = min(4_194_240, 4_194_304) = 4_194_240
    // capacity = floor(4_194_240/64)*64 = 4_194_240
    expect(falseAlarmBatchCapacity(limits, 64)).toBe(4_194_240);
  });

  it("throws when the storage binding is too small for even one aligned batch", () => {
    const tinyLimits: FalseAlarmDeviceLimits = {
      ...limits,
      maxStorageBufferBindingSize: 32, // exactly 1 candidate, but alignment needs 64
    };
    expect(() => falseAlarmBatchCapacity(tinyLimits, 64)).toThrow();
  });
});

describe("sortCandidatesDescendingByPosition", () => {
  it("sorts descending by position without mutating the input", () => {
    const input: FalseAlarmCandidateInput[] = [
      { start: 1n, position: 5 },
      { start: 2n, position: 50 },
      { start: 3n, position: 1 },
    ];
    const sorted = sortCandidatesDescendingByPosition(input);
    expect(sorted.map((c) => c.position)).toEqual([50, 5, 1]);
    expect(input.map((c) => c.position)).toEqual([5, 50, 1]);
  });
});

describe("initCandidateStates", () => {
  it("splits a 64-bit start into lo/hi u32s and zeroes result/next_position/found", () => {
    const start = 0x1122_3344_5566_7788n;
    const [initialized] = initCandidateStates([{ start, position: 42 }]);
    expect(initialized).toEqual({
      indexLo: 0x5566_7788,
      indexHi: 0x1122_3344,
      resultLo: 0,
      resultHi: 0,
      targetPosition: 42,
      nextPosition: 0,
      found: 0,
    });
  });
});

describe("step-budget scheduling", () => {
  it("initialStepBudget clamps maximum_position+1 to [1,64]", () => {
    expect(initialStepBudget(0)).toBe(1);
    expect(initialStepBudget(30)).toBe(31);
    expect(initialStepBudget(1_000)).toBe(64);
  });

  it("retargetStepBudget averages toward the desired rate for a fast round", () => {
    // budget=64, elapsed=400ms -> desired = round(64*800/400) = 128
    // elapsed(400) <= 1200, so budget = round((64+128)/2) = 96
    expect(retargetStepBudget(64, 400)).toBe(96);
  });

  it("retargetStepBudget jumps straight to desired for a slow (>1200ms) round", () => {
    // budget=64, elapsed=1600ms -> desired = round(64*800/1600) = 32
    expect(retargetStepBudget(64, 1_600)).toBe(32);
  });

  it("retargetStepBudget clamps to [1, 65536]", () => {
    expect(retargetStepBudget(1, 100_000)).toBe(1);
    expect(retargetStepBudget(65_536, 1)).toBe(65_536);
  });
});

describe("regular-mode ceiling trim", () => {
  it("trims exhausted candidates off the tail of a descending-sorted active array", () => {
    // Sorted descending by targetPosition: 90, 50, 10.
    const states = [state({ targetPosition: 90 }), state({ targetPosition: 50 }), state({ targetPosition: 10 })];
    // completedCeiling=11: only targetPosition=10 (needs 11 steps) is done.
    const result = trimRegularTail(states, 3, 11);
    expect(result.activeCount).toBe(2);
    expect(result.finishedStepsAdded).toBe(11n); // 10 + 1

    // completedCeiling=51: targetPosition=50 is now also done (needs 51 steps).
    const result2 = trimRegularTail(states, result.activeCount, 51);
    expect(result2.activeCount).toBe(1);
    expect(result2.finishedStepsAdded).toBe(51n); // 50 + 1
  });

  it("trims nothing when the ceiling hasn't reached the shortest remaining candidate", () => {
    const states = [state({ targetPosition: 90 }), state({ targetPosition: 50 })];
    const result = trimRegularTail(states, 2, 10);
    expect(result.activeCount).toBe(2);
    expect(result.finishedStepsAdded).toBe(0n);
  });
});

describe("irregular-mode mirror + compaction", () => {
  it("advances next_position mirrors, clamped to target_position+1", () => {
    const states = [state({ targetPosition: 100, nextPosition: 90 }), state({ targetPosition: 20, nextPosition: 15 })];
    advanceCandidateMirrorWithoutHit(states, 2, 10);
    expect(states[0].nextPosition).toBe(100); // 90+10=100, clamp to 101 -> 100
    expect(states[1].nextPosition).toBe(21); // 15+10=25, clamp to 21
  });

  it("compacts (stable swap-remove) candidates whose next_position exceeds target_position", () => {
    // Candidate 0 survives (nextPosition <= targetPosition), 1 is done, 2 survives.
    const states = [
      state({ targetPosition: 100, nextPosition: 50, indexLo: 111 }),
      state({ targetPosition: 20, nextPosition: 21, indexLo: 222 }),
      state({ targetPosition: 30, nextPosition: 30, indexLo: 333 }),
    ];
    const result = compactCandidateStates(states, 3);
    expect(result.activeCount).toBe(2);
    expect(result.moved).toBe(true);
    expect(result.finishedStepsAdded).toBe(21n); // 20 + 1
    // Order preserved: survivors 0 then 2, moved into the leading prefix.
    expect(states.slice(0, 2).map((s) => s.indexLo)).toEqual([111, 333]);
  });

  it("reports moved=false and no finished steps when every candidate survives in place", () => {
    const states = [state({ targetPosition: 100, nextPosition: 50 }), state({ targetPosition: 20, nextPosition: 10 })];
    const result = compactCandidateStates(states, 2);
    expect(result.activeCount).toBe(2);
    expect(result.moved).toBe(false);
    expect(result.finishedStepsAdded).toBe(0n);
  });
});

describe("processCandidateHits", () => {
  const target = new Uint8Array(8);

  it("verifies and clears a found candidate, appending it to recovered", () => {
    const states = [state({ found: 1, resultLo: 0x1234, resultHi: 0 })];
    const recovered: bigint[] = [];
    const result = processCandidateHits(states, 1, target, true, () => true, recovered);
    expect(result).toEqual({ rejected: 0, stop: false, changed: true });
    expect(recovered).toEqual([0x1234n]);
    expect(states[0].found).toBe(0);
    expect(states[0].resultLo).toBe(0);
    expect(states[0].resultHi).toBe(0);
  });

  it("decodes result as a 64-bit little-endian-halves index", () => {
    const states = [state({ found: 1, resultLo: 0xffff_ffff, resultHi: 1 })];
    const recovered: bigint[] = [];
    processCandidateHits(states, 1, target, true, () => true, recovered);
    expect(recovered).toEqual([(1n << 32n) | 0xffff_ffffn]);
  });

  it("rejects a false alarm (verifyExact returns false), incrementing rejected and clearing state", () => {
    const states = [state({ found: 1, resultLo: 99 })];
    const recovered: bigint[] = [];
    const result = processCandidateHits(states, 1, target, true, () => false, recovered);
    expect(result).toEqual({ rejected: 1, stop: false, changed: true });
    expect(recovered).toEqual([]);
    expect(states[0].found).toBe(0);
  });

  it("deduplicates a verified index that's already in recovered", () => {
    const states = [state({ found: 1, resultLo: 42 })];
    const recovered: bigint[] = [42n];
    processCandidateHits(states, 1, target, true, () => true, recovered);
    expect(recovered).toEqual([42n]);
  });

  it("stops immediately (early return) on the first verified hit when findAll is false, leaving it uncleared", () => {
    const states = [state({ found: 1, resultLo: 1 }), state({ found: 1, resultLo: 2 })];
    const recovered: bigint[] = [];
    const result = processCandidateHits(states, 2, target, false, () => true, recovered);
    expect(result.stop).toBe(true);
    expect(recovered).toEqual([1n]);
    // The stopping candidate's found/result are deliberately left untouched
    // (native never clears it either, since the GPU state is never
    // touched again after a stop-at-first return).
    expect(states[0].found).toBe(1);
    expect(states[0].resultLo).toBe(1);
    // The second candidate was never reached.
    expect(states[1].found).toBe(1);
  });

  it("keeps going past a rejected hit even when findAll is false", () => {
    const states = [state({ found: 1, resultLo: 1 }), state({ found: 1, resultLo: 2 })];
    const recovered: bigint[] = [];
    const verifyExact = (index: bigint) => index === 2n;
    const result = processCandidateHits(states, 2, target, false, verifyExact, recovered);
    expect(result.stop).toBe(true);
    expect(result.rejected).toBe(1);
    expect(recovered).toEqual([2n]);
    expect(states[0].found).toBe(0); // rejected candidate 1 was cleared
  });

  it("ignores candidates whose found flag is 0", () => {
    const states = [state({ found: 0 }), state({ found: 1, resultLo: 5 })];
    const recovered: bigint[] = [];
    const result = processCandidateHits(states, 2, target, true, () => true, recovered);
    expect(result.changed).toBe(true);
    expect(recovered).toEqual([5n]);
  });
});

describe("workgroup storage constants", () => {
  it("verify shaders need 4 bytes more than precompute's equivalent (extra found_in_group atomic)", () => {
    expect(FALSE_ALARM_WORKGROUP_STORAGE_BYTES.compact).toBe(2_184);
    expect(FALSE_ALARM_WORKGROUP_STORAGE_BYTES.expanded).toBe(31_752);
  });
});
