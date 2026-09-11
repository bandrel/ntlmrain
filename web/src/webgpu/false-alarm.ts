// GPU false-alarm (candidate-verification) pipeline: drives the same
// `verify_compact.wgsl` / `verify_expanded.wgsl` shaders as the native CLI
// (`src/gpu.rs::GpuContext::check_candidates_with_progress`) to reject GPU
// rainbow-table false alarms and confirm real DES-key hits, in the browser.
//
// This exists because a real lookup against the production table returns
// on the order of 1-3 million candidates (Task 7 measured 2,967,049 in a
// real run), and crypto-wasm's single-threaded WASM verify (Task 2) cannot
// finish checking that many in a browser tab in any practical time. This
// file is a sibling driver to `webgpu/precompute.ts`, not a rewrite of it:
// it reuses that file's bind-group/offset-alignment reasoning and its
// `COMPLETION_MAGIC`/`ensureStorageSize`/`alignTo`/`divCeilU32` helpers
// rather than re-deriving them.
//
// Every constant, struct layout, and control-flow decision below is a
// direct port of `src/gpu.rs`'s `check_candidates_with_progress` and its
// helpers (`process_candidate_hits`, `advance_candidate_mirror_without_hit`,
// `compact_candidate_states`, `false_params`, `false_alarm_batch_capacity`,
// `validate_markers`, `gcd`, `lcm`) — checked against that file, not
// guessed.
//
// ---------------------------------------------------------------------------
// Two active-set bookkeeping modes (this is the part that actually matters
// for performance — read this before touching the dispatch loop):
//
//   - "Regular" mode runs before any hit has ever been found. Candidates are
//     sorted descending by `position`, so a `completedCeiling` that grows by
//     `budget` each round can trim exhausted candidates off the *tail* of
//     the active array with pure host-side arithmetic — no GPU readback at
//     all is needed to know which candidates are done.
//   - "Irregular" mode starts the moment any hit has been found (and stays
//     on for the rest of the run, mirroring native's one-way switch). Each
//     candidate's `nextPosition` is mirrored host-side, then compacted
//     (stable swap-remove) out of the active prefix once it exceeds its
//     `targetPosition`; the compacted slice is re-uploaded with a partial
//     buffer write.
//
// The full candidate-state buffer is read back from the GPU *only* when a
// round's completion markers show `COMPLETION_FOUND_MAGIC` somewhere (a
// cheap, small marker-buffer readback happens every round regardless — that
// one is unavoidable, since it is how the host learns whether a hit
// happened at all). A version that reads back full candidate state on every
// round (not just found rounds) reproduces the exact performance problem
// this task exists to fix; see `checkCandidatesWithProgress` below and
// confirm the `if (foundInRound)` guard is the only path that touches
// `stateReadback`.
// ---------------------------------------------------------------------------

import type { ShaderVariant } from "./precompute";
import { COMPLETION_MAGIC, alignTo, divCeilU32, ensureStorageSize, targetToHashWords, type PrecomputeDeviceLimits } from "./precompute";

export type FalseAlarmDeviceLimits = PrecomputeDeviceLimits;

/** `COMPLETION_FOUND_MAGIC` in `src/gpu.rs` (0x4259_3732) and both WGSL
 * shaders — "workgroup done AND at least one invocation in it found a hit
 * this round". `COMPLETION_MAGIC` (plain "workgroup done") is imported from
 * `./precompute`, which already carries the identical constant. */
export const COMPLETION_FOUND_MAGIC = 0x4259_3732;

/** `STATE_BYTES` in `src/gpu.rs`: bytes per `CandidateState` record. */
export const STATE_BYTES = 32;
/** `size_of::<FalseParams>()` in `src/gpu.rs`. */
export const FALSE_PARAMS_BYTES = 32;

/**
 * Workgroup-storage bytes the false-alarm/verify shaders need — precompute's
 * equivalent constants (`webgpu/device.ts`'s `EXPANDED_WORKGROUP_STORAGE_BYTES`
 * / `webgpu/tuning.ts`'s inline `2_180` for compact) PLUS 4 bytes each for the
 * verify shaders' extra `found_in_group: atomic<u32>` workgroup variable
 * (absent from the precompute shaders, which never report a "found"
 * condition). Deliberately a *separate* constant, not a reuse of precompute's
 * undercounted one: a device that just barely supports the precompute
 * variant's workgroup-storage requirement can still be 4 bytes short of what
 * the false-alarm variant of the same shader class needs.
 */
export const FALSE_ALARM_WORKGROUP_STORAGE_BYTES: Record<ShaderVariant, number> = {
  compact: 2_184,
  expanded: 31_752,
};

/** Whether `device` has enough workgroup storage to compile the false-alarm
 * shader for `variant` at all (independent of whether it was ever tuned). */
export function supportsFalseAlarmShader(device: GPUDevice, variant: ShaderVariant): boolean {
  return device.limits.maxComputeWorkgroupStorageSize >= FALSE_ALARM_WORKGROUP_STORAGE_BYTES[variant];
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

// ---------------------------------------------------------------------------
// `gcd`/`lcm`, ported verbatim from `src/gpu.rs` (used only by
// `falseAlarmBatchCapacity` below; native's precompute path never needed
// these, which is why they don't already live in `webgpu/precompute.ts`).
// ---------------------------------------------------------------------------

export function gcd(left: number, right: number): number {
  let a = left >>> 0;
  let b = right >>> 0;
  while (b !== 0) {
    [a, b] = [b, a % b];
  }
  return a;
}

export function lcm(left: number, right: number): number {
  return (left / gcd(left, right)) * right;
}

/**
 * Mirrors `false_alarm_batch_capacity` in `src/gpu.rs`: the number of
 * candidates one `dispatchWorkgroups` call can cover, given the *whole*
 * batch (which may be far larger) is windowed across multiple dispatches
 * into a single persistent state buffer.
 */
export function falseAlarmBatchCapacity(limits: FalseAlarmDeviceLimits, workgroup: number): number {
  const offsetAlignment = Math.max(limits.minStorageBufferOffsetAlignment, 1);
  const offsetCandidateAlignment = offsetAlignment / gcd(offsetAlignment, STATE_BYTES);
  const alignment = lcm(workgroup, offsetCandidateAlignment);
  const raw = Math.min(
    limits.maxComputeWorkgroupsPerDimension * workgroup,
    Math.floor(limits.maxStorageBufferBindingSize / STATE_BYTES),
  );
  const capacity = Math.floor(raw / alignment) * alignment;
  if (capacity === 0) {
    throw new Error("storage binding is too small for one false-alarm batch");
  }
  return capacity;
}

// ---------------------------------------------------------------------------
// `CandidateState` (32 bytes/record, byte-identical to the WGSL struct in
// both `shaders/verify_compact.wgsl` and `shaders/verify_expanded.wgsl`):
//
//   index_lo         u32 @  0   (`index.x` in WGSL)
//   index_hi         u32 @  4   (`index.y` in WGSL)
//   result_lo        u32 @  8   (`result.x` in WGSL)
//   result_hi        u32 @ 12   (`result.y` in WGSL)
//   target_position  u32 @ 16
//   next_position    u32 @ 20
//   found            u32 @ 24
//   padding          u32 @ 28   (0)
// ---------------------------------------------------------------------------

export interface CandidateStateFields {
  indexLo: number;
  indexHi: number;
  resultLo: number;
  resultHi: number;
  targetPosition: number;
  nextPosition: number;
  found: number;
}

/** One decoded lookup-candidate: a rainbow-chain start value plus the
 * reduction-step position (`FalseAlarmCandidate` in `src/gpu.rs`). */
export interface FalseAlarmCandidateInput {
  start: bigint;
  position: number;
}

/** Mirrors `ordered.sort_by_key(|c| Reverse(c.position))` — load-bearing for
 * the regular-mode tail-trim optimization, not cosmetic (see module docs). */
export function sortCandidatesDescendingByPosition(
  candidates: readonly FalseAlarmCandidateInput[],
): FalseAlarmCandidateInput[] {
  return [...candidates].sort((a, b) => b.position - a.position);
}

/** Build the initial per-candidate state array from sorted candidates,
 * mirroring `check_candidates_with_progress`'s `states` construction. */
export function initCandidateStates(ordered: readonly FalseAlarmCandidateInput[]): CandidateStateFields[] {
  return ordered.map((candidate) => ({
    indexLo: Number(candidate.start & 0xffff_ffffn),
    indexHi: Number((candidate.start >> 32n) & 0xffff_ffffn),
    resultLo: 0,
    resultHi: 0,
    targetPosition: candidate.position >>> 0,
    nextPosition: 0,
    found: 0,
  }));
}

/** Encode `states` (in order) as a `STATE_BYTES`-per-record byte buffer,
 * matching the WGSL `CandidateState` layout documented above. */
export function encodeCandidateStates(states: readonly CandidateStateFields[]): ArrayBuffer {
  const buffer = new ArrayBuffer(states.length * STATE_BYTES);
  const view = new DataView(buffer);
  states.forEach((state, index) => {
    const base = index * STATE_BYTES;
    view.setUint32(base + 0, state.indexLo >>> 0, true);
    view.setUint32(base + 4, state.indexHi >>> 0, true);
    view.setUint32(base + 8, state.resultLo >>> 0, true);
    view.setUint32(base + 12, state.resultHi >>> 0, true);
    view.setUint32(base + 16, state.targetPosition >>> 0, true);
    view.setUint32(base + 20, state.nextPosition >>> 0, true);
    view.setUint32(base + 24, state.found >>> 0, true);
    view.setUint32(base + 28, 0, true);
  });
  return buffer;
}

/** Decode `count` `CandidateState` records from `bytes` (a GPU readback). */
export function decodeCandidateStates(bytes: Uint8Array, count: number): CandidateStateFields[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const states: CandidateStateFields[] = new Array(count);
  for (let index = 0; index < count; index += 1) {
    const base = index * STATE_BYTES;
    states[index] = {
      indexLo: view.getUint32(base + 0, true),
      indexHi: view.getUint32(base + 4, true),
      resultLo: view.getUint32(base + 8, true),
      resultHi: view.getUint32(base + 12, true),
      targetPosition: view.getUint32(base + 16, true),
      nextPosition: view.getUint32(base + 20, true),
      found: view.getUint32(base + 24, true),
    };
  }
  return states;
}

// ---------------------------------------------------------------------------
// `FalseParams` uniform (32 bytes, byte-identical to the WGSL struct):
//
//   target_lo         u32 @  0
//   target_hi         u32 @  4
//   reduction_offset  u32 @  8
//   candidate_count   u32 @ 12
//   step_budget       u32 @ 16
//   padding0          u32 @ 20   (0)
//   padding1          u32 @ 24   (0)
//   padding2          u32 @ 28   (0)
// ---------------------------------------------------------------------------

export interface FalseParamsFields {
  targetLo: number;
  targetHi: number;
  reductionOffset: number;
  candidateCount: number;
  stepBudget: number;
}

export function buildFalseParamsBuffer(fields: FalseParamsFields): ArrayBuffer {
  const buffer = new ArrayBuffer(FALSE_PARAMS_BYTES);
  const view = new DataView(buffer);
  view.setUint32(0, fields.targetLo >>> 0, true);
  view.setUint32(4, fields.targetHi >>> 0, true);
  view.setUint32(8, fields.reductionOffset >>> 0, true);
  view.setUint32(12, fields.candidateCount >>> 0, true);
  view.setUint32(16, fields.stepBudget >>> 0, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);
  return buffer;
}

// ---------------------------------------------------------------------------
// Completion-marker validation. Unlike precompute's markers (which may only
// ever be `COMPLETION_MAGIC`), the false-alarm shader's markers can also be
// `COMPLETION_FOUND_MAGIC` when `allowFound` is `true` (native always passes
// `allow_found = true` for this shader) — any other value is still a
// dispatch failure.
// ---------------------------------------------------------------------------

/** Returns whether any workgroup reported a found candidate this round. */
export function validateFalseAlarmMarkers(markerWords: Uint32Array, groups: number, allowFound: boolean): boolean {
  let found = false;
  for (let index = 0; index < groups; index += 1) {
    const marker = markerWords[index];
    if (marker === COMPLETION_MAGIC) continue;
    if (allowFound && marker === COMPLETION_FOUND_MAGIC) {
      found = true;
      continue;
    }
    const hex = marker === undefined ? "undefined" : `0x${marker.toString(16).padStart(8, "0")}`;
    throw new Error(`GPU false-alarm dispatch did not publish a completion marker for workgroup ${index} (got ${hex})`);
  }
  return found;
}

// ---------------------------------------------------------------------------
// Step-budget scheduling (`check_candidates_with_progress`'s scheduler,
// separate from and simpler than precompute's EWMA rate model).
// ---------------------------------------------------------------------------

/** `maximum_steps.clamp(1, 64)` — the initial guess before any round runs. */
export function initialStepBudget(maximumPosition: number): number {
  return clamp(maximumPosition + 1, 1, 64);
}

/** Retarget for ~800ms/round after a normal (no-hit) round, mirroring:
 * `desired = round(budget * 800 / max(elapsedMs, 1))`; if `elapsedMs > 1200`
 * use `desired` outright, else average toward it; clamp to `[1, 65536]`. */
export function retargetStepBudget(budget: number, elapsedMs: number): number {
  const desired = Math.round((budget * 800) / Math.max(elapsedMs, 1));
  const next = elapsedMs > 1200 ? desired : Math.round((budget + desired) / 2);
  return clamp(next, 1, 65_536);
}

// ---------------------------------------------------------------------------
// Regular-mode tail trim (pure host bookkeeping, no GPU readback).
// ---------------------------------------------------------------------------

export interface RegularTrimResult {
  activeCount: number;
  finishedStepsAdded: bigint;
}

/** Mirrors the `else` branch of `check_candidates_with_progress`'s per-round
 * bookkeeping: trim exhausted candidates off the tail of `states[0,
 * activeCount)` (which must already be sorted descending by
 * `targetPosition`) whose `targetPosition + 1 <= completedCeiling`. */
export function trimRegularTail(
  states: readonly CandidateStateFields[],
  activeCount: number,
  completedCeiling: number,
): RegularTrimResult {
  let count = activeCount;
  let finishedStepsAdded = 0n;
  while (count > 0 && states[count - 1].targetPosition + 1 <= completedCeiling) {
    count -= 1;
    finishedStepsAdded += BigInt(states[count].targetPosition) + 1n;
  }
  return { activeCount: count, finishedStepsAdded };
}

// ---------------------------------------------------------------------------
// Irregular-mode per-candidate mirror + compaction.
// ---------------------------------------------------------------------------

/** Mirrors `advance_candidate_mirror_without_hit`: after a no-hit round in
 * irregular mode, advance each active candidate's `nextPosition` mirror by
 * `budget`, clamped to `targetPosition + 1`. Mutates `states` in place. */
export function advanceCandidateMirrorWithoutHit(
  states: CandidateStateFields[],
  activeCount: number,
  budget: number,
): void {
  for (let index = 0; index < activeCount; index += 1) {
    const state = states[index];
    state.nextPosition = Math.min(state.nextPosition + budget, state.targetPosition + 1);
  }
}

export interface CompactionResult {
  activeCount: number;
  moved: boolean;
  finishedStepsAdded: bigint;
}

/** Mirrors `compact_candidate_states`: a stable compaction (not a
 * swap-remove) that keeps every candidate whose `nextPosition <=
 * targetPosition` in the active prefix, in original relative order, and
 * drops the rest. Mutates `states` in place. */
export function compactCandidateStates(states: CandidateStateFields[], activeCount: number): CompactionResult {
  let write = 0;
  let moved = false;
  let finishedStepsAdded = 0n;
  for (let read = 0; read < activeCount; read += 1) {
    const candidate = states[read];
    if (candidate.nextPosition <= candidate.targetPosition) {
      if (write !== read) {
        states[write] = candidate;
        moved = true;
      }
      write += 1;
    } else {
      finishedStepsAdded += BigInt(candidate.targetPosition) + 1n;
      moved = true;
    }
  }
  return { activeCount: write, moved, finishedStepsAdded };
}

// ---------------------------------------------------------------------------
// Hit processing (CPU-side exact-match rejection — the one part of this
// pipeline that talks to crypto-wasm rather than the GPU).
// ---------------------------------------------------------------------------

export interface HitProcessingResult {
  rejected: number;
  stop: boolean;
  changed: boolean;
}

/**
 * Mirrors `process_candidate_hits`: for every `found` candidate in
 * `states[0, activeCount)`, decode `result` as a 64-bit plaintext-space
 * index and call `verifyExact` (crypto-wasm's `is_exact_des_key_match`, or
 * an equivalent test double — never reimplemented here). A verified hit is
 * pushed to `recovered` (deduplicated); either way the lane's `found`/
 * `result` fields are cleared so it doesn't re-trigger and gets swept by the
 * next compaction pass. If `!findAll` and a hit verifies, returns
 * immediately (an early return, not a flag checked later) with `stop: true`
 * and does not clear/process any remaining found candidates in this call.
 */
export function processCandidateHits(
  states: CandidateStateFields[],
  activeCount: number,
  target: Uint8Array,
  findAll: boolean,
  verifyExact: (index: bigint, target: Uint8Array) => boolean,
  recovered: bigint[],
): HitProcessingResult {
  let rejected = 0;
  let changed = false;
  for (let index = 0; index < activeCount; index += 1) {
    const candidate = states[index];
    if (candidate.found === 0) continue;
    const resultIndex = BigInt(candidate.resultLo >>> 0) | (BigInt(candidate.resultHi >>> 0) << 32n);
    if (verifyExact(resultIndex, target)) {
      if (!recovered.includes(resultIndex)) recovered.push(resultIndex);
      if (!findAll) {
        // Early return, matching native exactly: the lane's found/result
        // are deliberately NOT cleared here, since the caller stops
        // dispatching entirely and the GPU state is never touched again.
        return { rejected, stop: true, changed: true };
      }
    } else {
      rejected += 1;
    }
    candidate.found = 0;
    candidate.resultLo = 0;
    candidate.resultHi = 0;
    changed = true;
  }
  return { rejected, stop: false, changed };
}

// ---------------------------------------------------------------------------
// Shader loading / pipeline creation, mirroring `webgpu/precompute.ts`'s
// pattern exactly (bundle WGSL text via Vite `?raw`, compile with the
// `WORKGROUP_SIZE` pipeline-overridable constant).
// ---------------------------------------------------------------------------

import verifyCompactSource from "../../../shaders/verify_compact.wgsl?raw";
import verifyExpandedSource from "../../../shaders/verify_expanded.wgsl?raw";

export { verifyCompactSource, verifyExpandedSource };

function shaderSource(variant: ShaderVariant): string {
  return variant === "compact" ? verifyCompactSource : verifyExpandedSource;
}

export function createFalseAlarmPipeline(
  device: GPUDevice,
  variant: ShaderVariant,
  workgroupSize: number,
): GPUComputePipeline {
  const module = device.createShaderModule({
    label: `ntlmrain false-alarm ${variant} WG ${workgroupSize}`,
    code: shaderSource(variant),
  });
  return device.createComputePipeline({
    label: `ntlmrain false-alarm ${variant} WG ${workgroupSize}`,
    layout: "auto",
    compute: {
      module,
      entryPoint: "main",
      constants: { WORKGROUP_SIZE: workgroupSize },
    },
  });
}

// ---------------------------------------------------------------------------
// Main verify loop.
// ---------------------------------------------------------------------------

export interface FalseAlarmProgress {
  completedCeiling: number;
  maximumPosition: number;
  completedSteps: bigint;
  totalSteps: bigint;
  batchSteps: bigint;
  elapsedMs: number;
  stepBudget: number;
  activeCandidates: number;
  rejectedGpuHits: number;
  acceptedHits: number;
  batchWorkgroups: number;
  dispatches: number;
}

export interface FalseAlarmOptions {
  /** 8-byte NetNTLMv1 hash target. */
  target: Uint8Array;
  /** 0 for a single-target lookup; matches native's `table_index`. */
  tableIndex?: number;
  /** `all` in `src/gpu.rs`: keep going after the first verified hit. */
  findAll: boolean;
  /** `crypto-wasm`'s `is_exact_des_key_match`, or an equivalent test double
   * — never reimplemented here (see module docs / `processCandidateHits`). */
  verifyExact: (index: bigint, target: Uint8Array) => boolean;
  onProgress?: (progress: FalseAlarmProgress) => void;
}

/**
 * Browser port of `src/gpu.rs`'s `check_candidates_with_progress`. Returns
 * the recovered plaintext-space indices (native's `Vec<u64>`, as `bigint`s).
 *
 * One single persistent state buffer sized `candidates.length * STATE_BYTES`
 * covers the *whole* input batch — no multi-buffer paging across separate
 * allocations (explicitly out of scope; see module docs). If that exceeds
 * the device's storage limits, this throws instead of silently truncating.
 */
export async function checkCandidatesWithProgress(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  lutBuffer: GPUBuffer,
  deviceLimits: FalseAlarmDeviceLimits,
  workgroupSize: number,
  candidates: readonly FalseAlarmCandidateInput[],
  options: FalseAlarmOptions,
): Promise<bigint[]> {
  if (candidates.length === 0) return [];

  const tableIndex = options.tableIndex ?? 0;
  const { hashLo: targetLo, hashHi: targetHi } = targetToHashWords(options.target);
  const reductionOffset = (tableIndex * 65_536) >>> 0;

  const ordered = sortCandidatesDescendingByPosition(candidates);
  let states = initCandidateStates(ordered);

  const stateBytes = states.length * STATE_BYTES;
  ensureStorageSize(deviceLimits, stateBytes, "candidate state");

  const queue = device.queue;
  const stateBuffer = device.createBuffer({
    label: "ntlmrain candidate state",
    size: stateBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  queue.writeBuffer(stateBuffer, 0, encodeCandidateStates(states));

  const stateReadback = device.createBuffer({
    label: "ntlmrain candidate readback",
    size: stateBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const uniform = device.createBuffer({
    label: "ntlmrain false-alarm parameters",
    size: FALSE_PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const capacity = falseAlarmBatchCapacity(deviceLimits, workgroupSize);
  const maximumGroups = divCeilU32(Math.min(capacity, states.length), workgroupSize);
  const markerBytes = alignTo(Math.max(maximumGroups, 1) * 4, 8);
  const markers = device.createBuffer({
    label: "ntlmrain false-alarm markers",
    size: markerBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const markerReadback = device.createBuffer({
    label: "ntlmrain false-alarm marker readback",
    size: markerBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  try {
    return await runLoop();
  } finally {
    stateBuffer.destroy();
    stateReadback.destroy();
    uniform.destroy();
    markers.destroy();
    markerReadback.destroy();
  }

  async function runLoop(): Promise<bigint[]> {
    const maximumPosition = ordered.reduce((max, c) => Math.max(max, c.position), 0);
    const maximumSteps = maximumPosition + 1;
    const totalSteps = ordered.reduce((sum, c) => sum + BigInt(c.position) + 1n, 0n);

    let budget = initialStepBudget(maximumPosition);
    let activeCount = states.length;
    let completedCeiling = 0;
    let finishedSteps = 0n;
    let reportedSteps = 0n;
    let irregularSchedule = false;
    const recovered: bigint[] = [];
    let rejected = 0;

    const bindGroupLayout = pipeline.getBindGroupLayout(0);

    while (activeCount > 0) {
      const maximumRemaining = irregularSchedule
        ? states.slice(0, activeCount).reduce((max, s) => Math.max(max, s.targetPosition + 1 - s.nextPosition), 0)
        : Math.max(maximumSteps - completedCeiling, 0);
      if (maximumRemaining === 0) break;
      budget = Math.max(Math.min(budget, maximumRemaining), 1);

      let foundInRound = false;
      let roundGroups = 0;
      let dispatches = 0;
      let offset = 0;
      const started = performance.now();

      while (offset < activeCount) {
        const count = Math.min(activeCount - offset, capacity);
        const groups = divCeilU32(count, workgroupSize);
        const activeMarkerBytes = alignTo(Math.max(groups, 1) * 4, 8);

        const offsetBytes = offset * STATE_BYTES;
        if (offsetBytes % deviceLimits.minStorageBufferOffsetAlignment !== 0) {
          throw new Error(
            `false-alarm dispatch offset ${offsetBytes} is not a multiple of the device's ` +
              `minStorageBufferOffsetAlignment (${deviceLimits.minStorageBufferOffsetAlignment}); this should be ` +
              "unreachable given falseAlarmBatchCapacity's alignment computation",
          );
        }

        const paramsBuffer = buildFalseParamsBuffer({
          targetLo,
          targetHi,
          reductionOffset,
          candidateCount: count,
          stepBudget: budget,
        });
        queue.writeBuffer(uniform, 0, paramsBuffer);

        const bindGroup = device.createBindGroup({
          label: "ntlmrain false-alarm bindings",
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: { buffer: stateBuffer, offset: offsetBytes, size: count * STATE_BYTES } },
            { binding: 2, resource: { buffer: lutBuffer } },
            { binding: 3, resource: { buffer: markers } },
          ],
        });

        const encoder = device.createCommandEncoder({ label: "ntlmrain false-alarm dispatch" });
        encoder.clearBuffer(markers, 0, activeMarkerBytes);
        const pass = encoder.beginComputePass({ label: "ntlmrain false-alarm pass" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(groups);
        pass.end();
        encoder.copyBufferToBuffer(markers, 0, markerReadback, 0, activeMarkerBytes);
        queue.submit([encoder.finish()]);

        await markerReadback.mapAsync(GPUMapMode.READ, 0, activeMarkerBytes);
        const markerWords = new Uint32Array(markerReadback.getMappedRange(0, activeMarkerBytes).slice(0));
        markerReadback.unmap();

        // Conditional-readback short-circuit lives here: this marker
        // readback is cheap (a handful of u32s) and happens every dispatch
        // regardless, but the FULL candidate-state buffer below is only
        // ever touched when this call returns true.
        foundInRound = validateFalseAlarmMarkers(markerWords, groups, true);
        roundGroups += groups;
        dispatches += 1;
        offset += count;
        // Stop dispatching further windows the instant any window's
        // markers show a found-magic this round (mirrors native's early
        // break) — this is an early exit from the window loop, not merely
        // a flag checked afterward.
        if (foundInRound) break;
      }
      const elapsedMs = performance.now() - started;

      if (foundInRound) {
        const activeBytes = activeCount * STATE_BYTES;
        const encoder = device.createCommandEncoder({ label: "ntlmrain candidate state copy" });
        encoder.copyBufferToBuffer(stateBuffer, 0, stateReadback, 0, activeBytes);
        queue.submit([encoder.finish()]);

        await stateReadback.mapAsync(GPUMapMode.READ, 0, activeBytes);
        const bytes = new Uint8Array(stateReadback.getMappedRange(0, activeBytes).slice(0));
        stateReadback.unmap();

        const readStates = decodeCandidateStates(bytes, activeCount);
        states = [...readStates, ...states.slice(activeCount)];

        const processed = processCandidateHits(states, activeCount, options.target, options.findAll, options.verifyExact, recovered);
        rejected += processed.rejected;
        if (processed.stop) {
          return recovered;
        }

        // A rejected hit stopped its lane early, while later windows in
        // this round may not have run at all — switch to the general
        // per-lane schedule only for this rare continuation path, exactly
        // mirroring native's comment on the equivalent branch.
        irregularSchedule = true;
        const compacted = compactCandidateStates(states, activeCount);
        finishedSteps += compacted.finishedStepsAdded;
        activeCount = compacted.activeCount;
        if (activeCount > 0) {
          queue.writeBuffer(stateBuffer, 0, encodeCandidateStates(states.slice(0, activeCount)));
        }
      } else if (irregularSchedule) {
        advanceCandidateMirrorWithoutHit(states, activeCount, budget);
        const compacted = compactCandidateStates(states, activeCount);
        finishedSteps += compacted.finishedStepsAdded;
        activeCount = compacted.activeCount;
        if (compacted.moved && activeCount > 0) {
          queue.writeBuffer(stateBuffer, 0, encodeCandidateStates(states.slice(0, activeCount)));
        }
      } else {
        completedCeiling += budget;
        const trimmed = trimRegularTail(states, activeCount, completedCeiling);
        activeCount = trimmed.activeCount;
        finishedSteps += trimmed.finishedStepsAdded;
      }

      let completedSteps: bigint;
      if (irregularSchedule) {
        completedCeiling =
          activeCount > 0
            ? states.slice(0, activeCount).reduce((min, s) => Math.min(min, s.nextPosition), Number.POSITIVE_INFINITY)
            : maximumSteps;
        completedSteps =
          finishedSteps + states.slice(0, activeCount).reduce((sum, s) => sum + BigInt(s.nextPosition), 0n);
      } else {
        completedSteps = finishedSteps + BigInt(activeCount) * BigInt(completedCeiling);
      }
      if (completedSteps > totalSteps) completedSteps = totalSteps;
      const batchSteps = completedSteps - reportedSteps;
      reportedSteps = completedSteps;

      options.onProgress?.({
        completedCeiling,
        maximumPosition,
        completedSteps,
        totalSteps,
        batchSteps,
        elapsedMs,
        stepBudget: budget,
        activeCandidates: activeCount,
        rejectedGpuHits: rejected,
        acceptedHits: recovered.length,
        batchWorkgroups: roundGroups,
        dispatches,
      });

      budget = retargetStepBudget(budget, elapsedMs);
    }

    return recovered;
  }
}
