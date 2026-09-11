// IndexedDB-backed run-history archive.
//
// One record per completed (or failed) run, keyed by a generated run ID.
// The stored shape mirrors `src/artifacts.rs`'s `RunManifest` field-for-field
// (`schema_version`, `run_id`, `created_at`, `command`, `input`, `compute`,
// `selected_device`, `tuning`, `outputs`, `result`) so a browser run and a
// native CLI run describe themselves the same way, even though nothing here
// reads/writes the native JSON file — this is a completely separate
// IndexedDB store, not a port of `src/artifacts.rs`'s file I/O.
//
// `outputs` deliberately never holds raw endpoint/candidate bytes: a single
// DES1 endpoint set is already ~7MB (881,688 endpoints * 8 bytes), and a run
// with a des2 slot doubles that. Storing that per archived run would make
// the archive balloon after a handful of runs, so `outputs` here only ever
// holds short human-readable descriptions (e.g. endpoint/candidate counts),
// never the underlying bytes. Callers who want the raw bytes must keep them
// in memory themselves for the lifetime of the run.

import type { TuningSelection } from "../webgpu/tuning";

export const ARCHIVE_SCHEMA_VERSION = 1;

const DB_NAME = "ntlmrain-run-archive";
const DB_VERSION = 1;
const STORE_NAME = "runs";

export type RecoveryMode = "full" | "single-des";

export interface ArchivedInput {
  raw: string;
  mode: RecoveryMode;
}

export interface ArchivedBackend {
  vendor?: string;
  architecture?: string;
  description?: string;
}

export interface ArchivedCompute {
  kind: "webgpu";
  implementation: "webgpu";
  backend: ArchivedBackend;
}

export interface ArchivedDevice {
  vendor?: string;
  architecture?: string;
  description?: string;
  limits: Record<string, number>;
}

export interface ArchivedResult {
  matched: boolean;
  des1Keys: string[];
  des2Keys: string[];
  pt3Hex: string | null;
  ntHashesHex: string[];
  errorMessage?: string;
}

/** Mirrors `src/artifacts.rs::RunManifest`, field-for-field. */
export interface RunRecord {
  schema_version: number;
  run_id: string;
  created_at: string;
  command: string;
  input: ArchivedInput;
  compute: ArchivedCompute;
  selected_device: ArchivedDevice;
  tuning: TuningSelection | null;
  outputs: Record<string, string>;
  result: ArchivedResult;
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  input_mode: RecoveryMode;
  matched: boolean;
  nt_hash_hex: string | null;
}

/** A 12-hex-character run ID, matching `src/artifacts.rs::new_run_id`'s length. */
export function generateRunId(): string {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "run_id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("failed to open the ntlmrain run-archive database"));
  });
}

/** Store (or overwrite) one run's archive record. */
export async function putRun(record: RunRecord): Promise<void> {
  const db = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(record);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("failed to write the run-archive entry"));
    });
  } finally {
    db.close();
  }
}

/** Fetch one run's full archive record, or `null` if it does not exist. */
export async function getRun(runId: string): Promise<RunRecord | null> {
  const db = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).get(runId);
      request.onsuccess = () => resolve((request.result as RunRecord | undefined) ?? null);
      request.onerror = () => reject(request.error ?? new Error("failed to read the run-archive entry"));
    });
  } finally {
    db.close();
  }
}

/** List every archived run's summary, most recent first. */
export async function listRuns(): Promise<RunSummary[]> {
  const db = await openDatabase();
  try {
    const records = await new Promise<RunRecord[]>((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve((request.result as RunRecord[] | undefined) ?? []);
      request.onerror = () => reject(request.error ?? new Error("failed to list the run-archive entries"));
    });
    return records
      .map((record) => ({
        run_id: record.run_id,
        created_at: record.created_at,
        input_mode: record.input.mode,
        matched: record.result.matched,
        nt_hash_hex: record.result.ntHashesHex[0] ?? null,
      }))
      .sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0));
  } finally {
    db.close();
  }
}

/** Delete one archived run. Not surfaced in the UI yet, but useful for tests/cleanup. */
export async function deleteRun(runId: string): Promise<void> {
  const db = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).delete(runId);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("failed to delete the run-archive entry"));
    });
  } finally {
    db.close();
  }
}
