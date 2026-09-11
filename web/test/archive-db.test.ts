// `fake-indexeddb/auto` installs a real (in-memory) IndexedDB implementation
// on the global scope so `archive/db.ts` can be exercised under Vitest's
// "node" environment without a browser.
import "fake-indexeddb/auto";
import { beforeEach, describe, expect, it } from "vitest";
import { deleteRun, generateRunId, getRun, listRuns, putRun, type RunRecord } from "../src/archive/db";

function sampleRecord(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    schema_version: 1,
    run_id: generateRunId(),
    created_at: new Date().toISOString(),
    command: "web-ui recover",
    input: { raw: "AABBCCDD11223344", mode: "single-des" },
    compute: { kind: "webgpu", implementation: "webgpu", backend: { vendor: "test-vendor" } },
    selected_device: { vendor: "test-vendor", limits: { maxBufferSize: 1024 } },
    tuning: null,
    outputs: { "des1.endpoints": "881688 endpoints (not persisted)" },
    result: { matched: false, des1Keys: [], des2Keys: [], pt3Hex: null, ntHashesHex: [] },
    ...overrides,
  };
}

describe("archive/db.ts", () => {
  beforeEach(async () => {
    // Each test gets a clean slate: fake-indexeddb keeps its data in-memory
    // per-process, so explicitly delete any prior test's records rather than
    // relying on database isolation vitest doesn't provide between tests.
    for (const summary of await listRuns()) {
      await deleteRun(summary.run_id);
    }
  });

  it("generates 12-hex-character run IDs", () => {
    const id = generateRunId();
    expect(id).toHaveLength(12);
    expect(id).toMatch(/^[0-9a-f]{12}$/);
  });

  it("writes a record and reads it back unchanged", async () => {
    const record = sampleRecord();
    await putRun(record);
    const fetched = await getRun(record.run_id);
    expect(fetched).toEqual(record);
  });

  it("returns null for a run ID that was never written", async () => {
    expect(await getRun("does-not-exist")).toBeNull();
  });

  it("lists summaries for every written run, most recent first", async () => {
    const older = sampleRecord({ created_at: "2024-01-01T00:00:00.000Z" });
    const newer = sampleRecord({
      created_at: "2024-06-01T00:00:00.000Z",
      result: { matched: true, des1Keys: ["1"], des2Keys: ["2"], pt3Hex: "aa", ntHashesHex: ["deadbeef"] },
    });
    await putRun(older);
    await putRun(newer);

    const summaries = await listRuns();
    expect(summaries.map((summary) => summary.run_id)).toEqual([newer.run_id, older.run_id]);
    expect(summaries[0]).toEqual({
      run_id: newer.run_id,
      created_at: newer.created_at,
      input_mode: "single-des",
      matched: true,
      nt_hash_hex: "deadbeef",
    });
    expect(summaries[1].matched).toBe(false);
    expect(summaries[1].nt_hash_hex).toBeNull();
  });

  it("overwrites an existing record when put again with the same run ID", async () => {
    const record = sampleRecord();
    await putRun(record);
    const updated: RunRecord = { ...record, result: { ...record.result, matched: true } };
    await putRun(updated);

    const fetched = await getRun(record.run_id);
    expect(fetched?.result.matched).toBe(true);
    expect(await listRuns()).toHaveLength(1);
  });

  it("deletes a record", async () => {
    const record = sampleRecord();
    await putRun(record);
    await deleteRun(record.run_id);
    expect(await getRun(record.run_id)).toBeNull();
    expect(await listRuns()).toHaveLength(0);
  });
});
