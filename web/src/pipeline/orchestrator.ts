// End-to-end recovery pipeline: parse -> precompute -> lookup -> verify ->
// assemble, wired to Task 2's `crypto-wasm`, Task 3's `webgpu/precompute.ts`,
// Task 4's `webgpu/tuning.ts` + `tuning-cache.ts`, and this task's own
// `api/lookup-client.ts` / `pipeline/input-parser.ts`.
//
// Every dependency on those other tasks' modules is reached through the
// `OrchestratorPorts` interface below rather than called directly, so
// `runOrchestrator`'s sequencing (des1 fully before des2 — a deliberate
// parity choice with the native CLI's ordering, not an oversight) can be
// unit-tested with fake ports that just record call order, without a real
// GPU or network. `createDefaultPorts` wires the real implementations for
// production use.

import { parseTarget, desTargets, k3Ciphertext, type Target } from "./input-parser";
import * as lookupClient from "../api/lookup-client";
import type { LookupClientConfig, StatusResponse } from "../api/lookup-client";
import {
  encode_endpoint_file,
  decode_candidate_file,
  verify_candidates,
  is_exact_des_key_match,
  recover_pt3,
  assemble_nt_hash,
  type VerifyProgressJs,
} from "../crypto";
import {
  createPrecomputePipeline,
  runPrecompute,
  type PrecomputeDeviceLimits,
  type PrecomputeProgress,
  type ShaderVariant,
} from "../webgpu/precompute";
import {
  checkCandidatesWithProgress,
  createFalseAlarmPipeline,
  supportsFalseAlarmShader,
  FALSE_ALARM_WORKGROUP_STORAGE_BYTES,
  type FalseAlarmCandidateInput,
  type FalseAlarmProgress,
} from "../webgpu/false-alarm";
import {
  autoTuneDevice,
  type AutoTuneDeviceLimits,
  type TuningKey,
  type TuningSelection,
} from "../webgpu/tuning";
import {
  computeTuningCacheKey,
  getCachedTuning,
  putCachedTuning,
  type TuningCacheKeyInputs,
} from "../webgpu/tuning-cache";

// ---------------------------------------------------------------------------
// Candidates / verification (crypto-wasm's wire types, kept as plain
// bigint-bearing objects here rather than js_sys handles so the ports
// interface has no wasm-bindgen dependency at the type level).
// ---------------------------------------------------------------------------

export interface CandidateRecord {
  /** The rainbow chain's position (reduction-step count), NOT a query index. */
  position: bigint;
  start: bigint;
}

export interface VerifyProgress {
  candidatesDone: bigint;
  candidatesTotal: bigint;
  stepsDone: bigint;
  stepsTotal: bigint;
  verifiedKeys: bigint;
}

export interface VerifyOutcome {
  keys: bigint[];
}

export type DesSlot = "des1" | "des2";

export interface LookupEvent {
  type: "submitted" | "status" | "downloaded";
  submissionToken?: string;
  pollWithinSeconds?: number | null;
  status?: StatusResponse;
  bytesDownloaded?: number;
}

export type OrchestratorEvent =
  | { type: "parsed"; target: Target }
  | { type: "tuning"; selection: TuningSelection }
  | { type: "precompute-progress"; which: DesSlot; progress: PrecomputeProgress }
  | { type: "lookup"; which: DesSlot; event: LookupEvent }
  | { type: "verify-progress"; which: DesSlot; progress: VerifyProgress }
  | { type: "result"; result: OrchestratorResult }
  | { type: "no-match" };

export interface RecoveredNtHash {
  pt1Index: bigint;
  pt2Index: bigint;
  ntHash: Uint8Array;
}

export interface OrchestratorResult {
  des1Keys: bigint[];
  des2Keys: bigint[];
  pt3: Uint8Array | null;
  ntHashes: RecoveredNtHash[];
}

export interface OrchestratorOptions {
  /** Stop after the first des1xdes2 pair unless `true`. Default `false`. */
  findAll?: boolean;
  onEvent?: (event: OrchestratorEvent) => void;
}

/** Table index for a single-target lookup; matches native's `TABLE_INDEX` usage (always 0 here). */
const TABLE_INDEX = 0;

/**
 * Seams this orchestrator calls through, injectable for testing. Production
 * callers should build these via {@link createDefaultPorts}.
 */
export interface OrchestratorPorts {
  getTuning(): Promise<TuningSelection>;
  precompute(
    target: Uint8Array,
    tuning: TuningSelection,
    onProgress?: (progress: PrecomputeProgress) => void,
  ): Promise<BigUint64Array>;
  encodeEndpointFile(endpoints: BigUint64Array): Uint8Array;
  lookup(endpointFile: Uint8Array, expectedCount: number, onEvent?: (event: LookupEvent) => void): Promise<Uint8Array>;
  decodeCandidateFile(bytes: Uint8Array, expectedQueryCount: number): CandidateRecord[];
  /**
   * Async as of Task 8's GPU verify port: the WASM serial verify this
   * replaced as the default wiring was a single blocking call, but the GPU
   * driver dispatches real `await`-ed rounds, so this had to become a
   * `Promise`. `tuning` is threaded through (mirroring `precompute`'s own
   * second parameter) because native's `GpuContext` compiles its
   * false-alarm/verify pipeline from the SAME tuned shader/workgroup as
   * precompute (`compile_pipelines` compiles both from one `TuningSelection`)
   * — the default GPU port below relies on this to pick/cache the right
   * verify pipeline instead of re-tuning or guessing.
   */
  verifyCandidates(
    candidates: CandidateRecord[],
    target: Uint8Array,
    stopAtFirst: boolean,
    tuning: TuningSelection,
    onProgress?: (progress: VerifyProgress) => void,
  ): Promise<VerifyOutcome>;
  /** Empty result means "not found" (native's `Option::None`). */
  recoverPt3(target: Uint8Array): Uint8Array | null;
  assembleNtHash(pt1Index: bigint, pt2Index: bigint, pt3: Uint8Array): Uint8Array;
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

/**
 * Run the full recovery pipeline for `input` (raw hex or a Responder/
 * hashcat-style capture line) against `ports`.
 *
 * Steps (matching the plan's task brief and the native CLI's ordering):
 *   1. Parse `input`.
 *   2-3. Precompute + submit/poll/download + decode des1.
 *   4. Only after step 3 fully resolves, repeat 2-3 for des2 if present
 *      (strictly sequential, not parallel — a deliberate parity choice).
 *   5. Only after BOTH des1's and des2's lookups have fully completed,
 *      verify each slot's candidate set against its own 8-byte target (in a
 *      separate pass, not interleaved into the loop above) — this ensures
 *      des2's precompute+lookup is never blocked behind des1's verify, which
 *      is otherwise independent work with its own network round-trip.
 *   6. If a des3 ciphertext is present, recover it locally (once, since it
 *      does not depend on which des1/des2 keys were found).
 *   7. Assemble NT hash(es) over the des1 x des2 cross product, stopping
 *      after the first pair unless `options.findAll` is set.
 *   8. Emit a final "result" event, or "no-match" if nothing verified.
 */
export async function runOrchestrator(
  input: string,
  ports: OrchestratorPorts,
  options: OrchestratorOptions = {},
): Promise<OrchestratorResult> {
  const parsed = parseTarget(input);
  options.onEvent?.({ type: "parsed", target: parsed.target });

  const targets = desTargets(parsed.target);
  const pt3Ciphertext = k3Ciphertext(parsed.target);

  const tuning = await ports.getTuning();
  options.onEvent?.({ type: "tuning", selection: tuning });

  // Steps 2-4: precompute + submit/poll/download + decode, strictly
  // sequential target-by-target (des1 fully before des2 starts) — but verify
  // is deliberately NOT run inside this loop (see step 5 below).
  const candidatesBySlot: CandidateRecord[][] = [];
  for (let index = 0; index < targets.length; index += 1) {
    const which: DesSlot = index === 0 ? "des1" : "des2";
    const target = targets[index];

    // Step 2: precompute this DES target's endpoints, strictly after the
    // previous slot's lookup (step 3) has fully resolved — the loop body
    // below only advances to the next iteration once `await`s here settle.
    const endpoints = await ports.precompute(target, tuning, (progress) =>
      options.onEvent?.({ type: "precompute-progress", which, progress }),
    );
    const endpointFile = ports.encodeEndpointFile(endpoints);

    // Step 3: submit/poll/download + decode.
    const candidateFile = await ports.lookup(endpointFile, endpoints.length, (event) =>
      options.onEvent?.({ type: "lookup", which, event }),
    );
    candidatesBySlot.push(ports.decodeCandidateFile(candidateFile, endpoints.length));
  }

  // Step 5: verify each slot's candidates, only after every slot's lookup
  // (the loop above) has fully completed.
  const desKeys: bigint[][] = [];
  for (let index = 0; index < targets.length; index += 1) {
    const which: DesSlot = index === 0 ? "des1" : "des2";
    const outcome = await ports.verifyCandidates(
      candidatesBySlot[index],
      targets[index],
      !options.findAll,
      tuning,
      (progress) => options.onEvent?.({ type: "verify-progress", which, progress }),
    );
    desKeys.push(outcome.keys);
  }

  const des1Keys = desKeys[0] ?? [];
  const des2Keys = desKeys[1] ?? [];

  // Step 6: recover DES3's plaintext once — it depends only on the fixed
  // des3 ciphertext, not on which des1/des2 keys verification found.
  let pt3: Uint8Array | null = null;
  if (pt3Ciphertext) {
    const recovered = ports.recoverPt3(pt3Ciphertext);
    pt3 = recovered && recovered.length === 2 ? recovered : null;
  }

  // Step 7: assemble NT hash(es) over the des1 x des2 cross product. Only
  // meaningful once all three DES parts are in hand (a des2 target and a
  // recovered pt3), so this is naturally empty for bare Des/TwoDes targets
  // or an unrecovered pt3.
  const ntHashes: RecoveredNtHash[] = [];
  if (pt3 !== null) {
    outer: for (const pt1Index of des1Keys) {
      for (const pt2Index of des2Keys) {
        ntHashes.push({ pt1Index, pt2Index, ntHash: ports.assembleNtHash(pt1Index, pt2Index, pt3) });
        if (!options.findAll) break outer;
      }
    }
  }

  const result: OrchestratorResult = { des1Keys, des2Keys, pt3, ntHashes };

  // "No match" means the pipeline required a DES slot that verification
  // never hit: des1 is always required, des2 only when a second DES target
  // was present at all.
  const matched = des1Keys.length > 0 && (targets.length < 2 || des2Keys.length > 0);
  if (!matched) {
    options.onEvent?.({ type: "no-match" });
  } else {
    options.onEvent?.({ type: "result", result });
  }
  return result;
}

// ---------------------------------------------------------------------------
// Default (production) ports, wiring the real Task 2/3/4 modules.
// ---------------------------------------------------------------------------

export interface DefaultPortsOptions {
  device: GPUDevice;
  lutBuffer: GPUBuffer;
  precomputeDeviceLimits: PrecomputeDeviceLimits;
  tuningDeviceLimits: AutoTuneDeviceLimits;
  supportedShaders: ShaderVariant[];
  tuningCacheKeyInputs: TuningCacheKeyInputs;
  lookupConfig: LookupClientConfig;
  incumbentTuning?: TuningKey;
  /**
   * When `true`, skip the cache-hit fast path (which otherwise returns a
   * cached selection directly with no fresh tune at all) and run
   * `autoTuneDevice` again, supplying the cached selection as `incumbent` so
   * `stableTuningWinner`'s hysteresis logic (keep the cached incumbent when
   * it's within noise of a new leader) actually gets exercised. Without this,
   * no caller ever reaches `autoTuneDevice` with a real incumbent, since the
   * only path that calls it today is the cache-miss path (where, by
   * construction, no incumbent can exist) — see the final whole-branch
   * review's finding on this. Default `false`: the default fast path (cache
   * hit -> use cached value, no tune) is unchanged.
   */
  forceRetune?: boolean;
}

/**
 * `TuningSelection.source` strings that represent a genuine fresh
 * auto-tune result worth persisting — i.e. everything `tuning.ts`'s
 * `autoTuneDevice` can return EXCEPT `"cache"` (a value it never actually
 * produces itself; reserved here for a selection this file re-labels after
 * a cache hit, mirroring `src/gpu.rs`'s `cached.source = "cache".into()`)
 * and `"slow-adapter-default"` (a deliberately-unmeasured fallback that
 * `src/gpu.rs`'s own cache-write gate (`selection.source != "cache" &&
 * selection.source != "slow-adapter-default"`) also refuses to persist).
 * `tuning-cache.ts`'s `putCachedTuning` does not enforce this itself (see
 * that file's review notes) — this orchestrator is the first real caller
 * wiring tuning to the cache, so it is responsible for the check.
 */
export function isCacheableTuningSource(source: string): boolean {
  return source !== "cache" && source !== "slow-adapter-default";
}

async function defaultGetTuning(options: DefaultPortsOptions): Promise<TuningSelection> {
  const cacheKey = await computeTuningCacheKey(options.tuningCacheKeyInputs);
  const cached = await getCachedTuning(cacheKey);
  if (cached && !options.forceRetune) {
    // Matches `src/gpu.rs`'s `cached.source = "cache".into()` /
    // `selection_reason = "cached-winner"` relabeling on a cache hit.
    return { ...cached, source: "cache", selectionReason: "cached-winner" };
  }

  // On a forced re-tune after a cache hit, feed the cached selection back in
  // as `incumbent` so the hysteresis tie-break in `stableTuningWinner` has a
  // real incumbent to prefer (rather than always tuning from a blank slate,
  // which is the only thing that happened here before this fix).
  const incumbent: TuningKey | undefined =
    options.incumbentTuning ?? (cached ? { shader: cached.shader, workgroupSize: cached.workgroupSize } : undefined);

  const selection = await autoTuneDevice({
    device: options.device,
    lutBuffer: options.lutBuffer,
    limits: options.tuningDeviceLimits,
    supportedShaders: options.supportedShaders,
    incumbent,
  });
  if (isCacheableTuningSource(selection.source)) {
    await putCachedTuning(cacheKey, selection);
  }
  return selection;
}

function defaultPrecompute(
  options: DefaultPortsOptions,
  pipelineCache: Map<string, GPUComputePipeline>,
): OrchestratorPorts["precompute"] {
  return async (target, tuning, onProgress) => {
    const cacheKey = `${tuning.shader}:${tuning.workgroupSize}`;
    let pipeline = pipelineCache.get(cacheKey);
    if (!pipeline) {
      pipeline = createPrecomputePipeline(options.device, tuning.shader, tuning.workgroupSize);
      pipelineCache.set(cacheKey, pipeline);
    }
    return runPrecompute(options.device, pipeline, options.lutBuffer, options.precomputeDeviceLimits, {
      target,
      workgroupSize: tuning.workgroupSize,
      onProgress,
    });
  };
}

async function defaultLookup(
  lookupConfig: LookupClientConfig,
  endpointFile: Uint8Array,
  expectedCount: number,
  onEvent?: (event: LookupEvent) => void,
): Promise<Uint8Array> {
  const receipt = await lookupClient.submit(lookupConfig, endpointFile);
  onEvent?.({
    type: "submitted",
    submissionToken: receipt.submissionToken,
    pollWithinSeconds: receipt.pollWithinSeconds,
  });
  try {
    await lookupClient.pollUntilReady(lookupConfig, receipt.submissionToken, expectedCount, receipt.pollWithinSeconds, {
      onStatus: (current) => onEvent?.({ type: "status", status: current }),
    });
    const bytes = await lookupClient.result(lookupConfig, receipt.submissionToken);
    onEvent?.({ type: "downloaded", bytesDownloaded: bytes.length });
    return bytes;
  } catch (error) {
    // Fire-and-forget cancel on any failure, matching native's
    // `if result.is_err() { self.cancel_best_effort(...) }`.
    await lookupClient.cancel(lookupConfig, receipt.submissionToken);
    throw error;
  }
}

function defaultDecodeCandidateFile(bytes: Uint8Array, expectedQueryCount: number): CandidateRecord[] {
  const records = decode_candidate_file(bytes, BigInt(expectedQueryCount)) as unknown as ArrayLike<{
    ordinal: bigint;
    start: bigint;
  }>;
  return Array.from(records, (record) => ({ position: record.ordinal, start: record.start }));
}

/**
 * Serial WASM verify (Task 2's `crypto-wasm::verify_candidates_serial`),
 * kept available as a small-batch/no-WebGPU fallback path even though it is
 * no longer the default wiring below — see `isCacheableTuningSource`-style
 * doc comment on `createDefaultPorts` for why. `tuning` is accepted (and
 * ignored) purely so this has the identical `OrchestratorPorts.verifyCandidates`
 * shape as `defaultVerifyCandidates` and the two are interchangeable.
 */
export async function wasmVerifyCandidates(
  candidates: CandidateRecord[],
  target: Uint8Array,
  stopAtFirst: boolean,
  _tuning: TuningSelection,
  onProgress?: (progress: VerifyProgress) => void,
): Promise<VerifyOutcome> {
  const starts = new BigUint64Array(candidates.map((candidate) => candidate.start));
  const positions = new Uint32Array(candidates.map((candidate) => Number(candidate.position)));
  const outcome = verify_candidates(starts, positions, target, TABLE_INDEX, stopAtFirst, (progress: VerifyProgressJs) => {
    onProgress?.({
      candidatesDone: BigInt(progress.candidates_done),
      candidatesTotal: BigInt(progress.candidates_total),
      stepsDone: BigInt(progress.steps_done),
      stepsTotal: BigInt(progress.steps_total),
      verifiedKeys: BigInt(progress.verified_keys),
    });
  });
  return { keys: Array.from(outcome.keys, BigInt) };
}

/**
 * GPU-based verify (Task 8's port of `src/gpu.rs`'s
 * `check_candidates_with_progress`) — the default wiring below. A real
 * lookup returns millions of candidates (Task 7 measured 2,967,049 in a
 * real run), which `wasmVerifyCandidates` above cannot finish checking in
 * any practical browser-tab time; this is the actual fix.
 *
 * Reuses the SAME shader/workgroup `tuning` picked for precompute (native's
 * `GpuContext::create` compiles both pipelines from one `TuningSelection`;
 * see the `OrchestratorPorts.verifyCandidates` doc comment) rather than
 * re-tuning independently for verify.
 */
function defaultVerifyCandidatesGpu(
  options: DefaultPortsOptions,
  falseAlarmPipelineCache: Map<string, GPUComputePipeline>,
): OrchestratorPorts["verifyCandidates"] {
  return async (candidates, target, stopAtFirst, tuning, onProgress) => {
    if (candidates.length === 0) return { keys: [] };

    if (!supportsFalseAlarmShader(options.device, tuning.shader)) {
      // Fail loudly rather than silently falling back to a slower path or
      // producing a confusing WebGPU validation error deep inside a
      // dispatch — the false-alarm/verify shader's workgroup-storage need
      // is 4 bytes larger than the precompute shader of the same variant
      // (see `FALSE_ALARM_WORKGROUP_STORAGE_BYTES`'s doc comment), so a
      // device that tuned to this shader for precompute can still be just
      // short of what verify needs.
      throw new Error(
        `device's maxComputeWorkgroupStorageSize (${options.device.limits.maxComputeWorkgroupStorageSize}) is ` +
          `insufficient for the "${tuning.shader}" false-alarm verify shader (needs ` +
          `${FALSE_ALARM_WORKGROUP_STORAGE_BYTES[tuning.shader]} bytes)`,
      );
    }

    const cacheKey = `${tuning.shader}:${tuning.workgroupSize}`;
    let pipeline = falseAlarmPipelineCache.get(cacheKey);
    if (!pipeline) {
      pipeline = createFalseAlarmPipeline(options.device, tuning.shader, tuning.workgroupSize);
      falseAlarmPipelineCache.set(cacheKey, pipeline);
    }

    const inputs: FalseAlarmCandidateInput[] = candidates.map((candidate) => ({
      start: candidate.start,
      position: Number(candidate.position),
    }));

    const recovered = await checkCandidatesWithProgress(
      options.device,
      pipeline,
      options.lutBuffer,
      options.precomputeDeviceLimits,
      tuning.workgroupSize,
      inputs,
      {
        target,
        tableIndex: TABLE_INDEX,
        findAll: !stopAtFirst,
        verifyExact: is_exact_des_key_match,
        onProgress: (progress: FalseAlarmProgress) => {
          onProgress?.({
            candidatesDone: BigInt(candidates.length - progress.activeCandidates),
            candidatesTotal: BigInt(candidates.length),
            stepsDone: progress.completedSteps,
            stepsTotal: progress.totalSteps,
            verifiedKeys: BigInt(progress.acceptedHits),
          });
        },
      },
    );

    return { keys: recovered };
  };
}

function defaultRecoverPt3(target: Uint8Array): Uint8Array | null {
  const recovered = recover_pt3(target);
  return recovered.length === 2 ? recovered : null;
}

function defaultAssembleNtHash(pt1Index: bigint, pt2Index: bigint, pt3: Uint8Array): Uint8Array {
  return assemble_nt_hash(pt1Index, pt2Index, pt3);
}

/**
 * Build production `OrchestratorPorts`, wiring the real Task 2/3/4 modules.
 *
 * Callers MUST `await initCrypto()` (from `../crypto`) before invoking
 * `runOrchestrator` with these ports — matching `crypto/index.ts`'s own
 * contract that nothing there initializes the WASM module lazily, so this
 * file does not call it on the caller's behalf either.
 */
export function createDefaultPorts(options: DefaultPortsOptions): OrchestratorPorts {
  const pipelineCache = new Map<string, GPUComputePipeline>();
  const falseAlarmPipelineCache = new Map<string, GPUComputePipeline>();
  return {
    getTuning: () => defaultGetTuning(options),
    precompute: defaultPrecompute(options, pipelineCache),
    encodeEndpointFile: (endpoints) => encode_endpoint_file(endpoints),
    lookup: (endpointFile, expectedCount, onEvent) =>
      defaultLookup(options.lookupConfig, endpointFile, expectedCount, onEvent),
    decodeCandidateFile: defaultDecodeCandidateFile,
    // The GPU driver is the default: every caller of `createDefaultPorts`
    // already supplies a `GPUDevice` (required for precompute too), so
    // there is no scenario where a real production run has a device for
    // precompute but not for verify. `wasmVerifyCandidates` remains
    // exported above for a caller that explicitly wants the small-batch/
    // no-WebGPU serial path instead (e.g. tests, or a future manual
    // override) — swap it in via `ports.verifyCandidates = wasmVerifyCandidates`.
    verifyCandidates: defaultVerifyCandidatesGpu(options, falseAlarmPipelineCache),
    recoverPt3: defaultRecoverPt3,
    assembleNtHash: defaultAssembleNtHash,
  };
}
