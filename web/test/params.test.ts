import { describe, expect, it } from "vitest";
import {
  buildParamsBuffer,
  PARAMS_BYTE_LENGTH,
  targetToHashWords,
  COMPLETION_MAGIC,
  validateMarkers,
} from "../src/webgpu/precompute";

describe("buildParamsBuffer", () => {
  it("is exactly 48 bytes", () => {
    const buffer = buildParamsBuffer({
      hashLo: 0,
      hashHi: 0,
      reductionOffset: 0,
      chainLen: 881_689,
      endpointStart: 0,
      sliceStart: 0,
      sliceSteps: 65_536,
      outputLen: 881_688,
    });
    expect(buffer.byteLength).toBe(48);
    expect(PARAMS_BYTE_LENGTH).toBe(48);
  });

  it("places every field at its documented offset, little-endian", () => {
    const buffer = buildParamsBuffer({
      hashLo: 0x11223344,
      hashHi: 0x55667788,
      reductionOffset: 0x99aabbcc,
      chainLen: 881_689,
      endpointStart: 65_536,
      sliceStart: 65_536,
      sliceSteps: 65_536,
      outputLen: 12_345,
      benchmarkSteps: 42,
      mode: 1,
    });
    const view = new DataView(buffer);
    expect(view.getUint32(0, true)).toBe(0x11223344); // hash_lo
    expect(view.getUint32(4, true)).toBe(0x55667788); // hash_hi
    expect(view.getUint32(8, true)).toBe(0x99aabbcc); // reduction_offset
    expect(view.getUint32(12, true)).toBe(881_689); // chain_len
    expect(view.getUint32(16, true)).toBe(65_536); // endpoint_start
    expect(view.getUint32(20, true)).toBe(65_536); // slice_start
    expect(view.getUint32(24, true)).toBe(65_536); // slice_steps
    expect(view.getUint32(28, true)).toBe(12_345); // output_len
    expect(view.getUint32(32, true)).toBe(42); // benchmark_steps
    expect(view.getUint32(36, true)).toBe(1); // mode
    expect(view.getUint32(40, true)).toBe(0); // padding0
    expect(view.getUint32(44, true)).toBe(0); // padding1
  });

  it("defaults benchmark_steps and mode to 0", () => {
    const buffer = buildParamsBuffer({
      hashLo: 0,
      hashHi: 0,
      reductionOffset: 0,
      chainLen: 881_689,
      endpointStart: 0,
      sliceStart: 0,
      sliceSteps: 65_536,
      outputLen: 881_688,
    });
    const view = new DataView(buffer);
    expect(view.getUint32(32, true)).toBe(0);
    expect(view.getUint32(36, true)).toBe(0);
  });
});

describe("targetToHashWords", () => {
  it("reads two little-endian u32s from an 8-byte target", () => {
    const target = new Uint8Array([0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);
    expect(targetToHashWords(target)).toEqual({ hashLo: 0x11223344, hashHi: 0x55667788 });
  });

  it("rejects a target that isn't exactly 8 bytes", () => {
    expect(() => targetToHashWords(new Uint8Array(7))).toThrow();
    expect(() => targetToHashWords(new Uint8Array(9))).toThrow();
  });
});

describe("validateMarkers", () => {
  it("passes when every marker in range equals COMPLETION_MAGIC", () => {
    const markers = new Uint32Array([COMPLETION_MAGIC, COMPLETION_MAGIC, COMPLETION_MAGIC]);
    expect(() => validateMarkers(markers, 3)).not.toThrow();
  });

  it("ignores markers past the checked group count", () => {
    const markers = new Uint32Array([COMPLETION_MAGIC, COMPLETION_MAGIC, 0xdeadbeef]);
    expect(() => validateMarkers(markers, 2)).not.toThrow();
  });

  it("throws on any non-magic marker within range", () => {
    const markers = new Uint32Array([COMPLETION_MAGIC, 0xdeadbeef, COMPLETION_MAGIC]);
    expect(() => validateMarkers(markers, 3)).toThrow(/workgroup 1/);
  });

  it("throws when a marker is missing entirely", () => {
    const markers = new Uint32Array([COMPLETION_MAGIC]);
    expect(() => validateMarkers(markers, 2)).toThrow(/workgroup 1/);
  });
});
