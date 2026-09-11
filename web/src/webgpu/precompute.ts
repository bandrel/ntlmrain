// Checkpointed adaptive precompute pipeline: drives the same
// `precompute_compact.wgsl` / `precompute_expanded.wgsl` shaders as the
// native CLI (`src/gpu.rs::GpuContext::precompute_with_progress`) to
// produce byte-for-byte compatible rainbow-chain endpoints in the browser.
//
// Every numeric constant and the outer/inner loop structure below is a
// direct port of `src/gpu.rs` (checked against that file, not guessed):
// `precompute_params`, `sliced_steps_for_range`, `sliced_dispatch_width`,
// `steps_for_range`, `align_up_u32`, `align_to`, `div_ceil_u32`,
// `max_dispatch_invocations`, and the adaptive-rate smoothing in
// `precompute_with_progress`'s dispatch loop.
//
// Binding-strategy note (read this before touching bind group code):
// the plan's brief assumed the shader indexes its output buffer by
// `params.endpoint_start + gid` so a *full*-buffer binding could be used
// to dodge WebGPU's 256-byte storage-offset-alignment rule. Reading the
// actual WGSL (`shaders/precompute_compact.wgsl` / `_expanded.wgsl`)
// shows `global_gid = params.endpoint_start + gid` is used ONLY to derive
// `total_steps` (the DES chain step count for that endpoint); the buffer
// reads/writes (`output_buf[gid]`) are indexed by the *local*, 0-based
// dispatch invocation id. Binding the full buffer would therefore make
// every dispatch write to slots `[0, length)` regardless of the batch's
// real position, silently clobbering earlier checkpoints. So this file
// binds a `{buffer, offset, size}` sub-range per dispatch instead, exactly
// like the native code's `wgpu::BufferBinding` — offsets are always a
// multiple of `workgroup * 8` bytes (checkpoint/slice widths are aligned
// up to the workgroup size), which is a multiple of the standard 256-byte
// `minStorageBufferOffsetAlignment` for every supported workgroup size
// (32, 64, 128, 256, 512, 1024), so the alignment rule is satisfied without
// needing the full-buffer workaround.

export const DEFAULT_CHAIN_LEN = 881_689;
export const DEFAULT_CHECKPOINT_STEPS = 65_536;
export const MAX_HOST_DISPATCH_INVOCATIONS = 65_536;
export const DES_LUT_BYTES = 102_016;

// Native uses 1_800ms; the browser defaults lower because a blocking
// `await buffer.mapAsync()` for that long risks the tab looking hung.
// Exposed as a parameter so a later manual-override UI can raise it.
export const DEFAULT_BUDGET_MS = 800;

// Same initial guess native seeds `target_steps` with before the first
// measured rate is available (`precompute_with_progress`'s `let mut
// target_steps = ... 256_000_000`).
export const INITIAL_TARGET_STEPS = 256_000_000;
export const MIN_TARGET_STEPS = 64_000_000;
export const MAX_TARGET_STEPS = 4_000_000_000;
export const ADAPTIVE_ALPHA = 0.25;

export const COMPLETION_MAGIC = 0x4259_3731;

export type ShaderVariant = "compact" | "expanded";

// ---------------------------------------------------------------------------
// Params uniform (48 bytes / 12 u32 fields, offsets fixed by the shared
// WGSL `struct Params` in both shaders — verified against
// shaders/precompute_compact.wgsl and shaders/precompute_expanded.wgsl,
// which declare identical field order):
//
//   hash_lo           u32 @  0
//   hash_hi           u32 @  4
//   reduction_offset  u32 @  8   (0 for a single-target lookup)
//   chain_len         u32 @ 12   (881_689 by default)
//   endpoint_start    u32 @ 16   (this dispatch's global start position)
//   slice_start       u32 @ 20   (this checkpoint round's base position)
//   slice_steps       u32 @ 24   (checkpoint_steps, aligned to workgroup)
//   output_len        u32 @ 28   (THIS DISPATCH's width, not the full
//                                 881_688-endpoint array total — the field
//                                 is reused per-dispatch by the native code
//                                 too; see `precompute_params`'s call site)
//   benchmark_steps   u32 @ 32   (0; benchmark mode unused here)
//   mode              u32 @ 36   (0; benchmark mode unused here)
//   padding0          u32 @ 40   (0)
//   padding1          u32 @ 44   (0)
// ---------------------------------------------------------------------------

export const PARAMS_BYTE_LENGTH = 48;

export interface PrecomputeParamsFields {
  hashLo: number;
  hashHi: number;
  reductionOffset: number;
  chainLen: number;
  endpointStart: number;
  sliceStart: number;
  sliceSteps: number;
  /** This dispatch's local width, NOT the overall endpoint array length. */
  outputLen: number;
  benchmarkSteps?: number;
  mode?: number;
}

/** Build the 48-byte `Params` uniform buffer contents at the offsets above. */
export function buildParamsBuffer(fields: PrecomputeParamsFields): ArrayBuffer {
  const buffer = new ArrayBuffer(PARAMS_BYTE_LENGTH);
  const view = new DataView(buffer);
  view.setUint32(0, fields.hashLo >>> 0, true);
  view.setUint32(4, fields.hashHi >>> 0, true);
  view.setUint32(8, fields.reductionOffset >>> 0, true);
  view.setUint32(12, fields.chainLen >>> 0, true);
  view.setUint32(16, fields.endpointStart >>> 0, true);
  view.setUint32(20, fields.sliceStart >>> 0, true);
  view.setUint32(24, fields.sliceSteps >>> 0, true);
  view.setUint32(28, fields.outputLen >>> 0, true);
  view.setUint32(32, (fields.benchmarkSteps ?? 0) >>> 0, true);
  view.setUint32(36, (fields.mode ?? 0) >>> 0, true);
  view.setUint32(40, 0, true);
  view.setUint32(44, 0, true);
  return buffer;
}

/**
 * `hash_lo`/`hash_hi` are the target's 8 bytes read as two little-endian
 * u32s (`u32::from_le_bytes(target[0..4])` / `target[4..8]` in
 * `precompute_params`), verbatim.
 */
export function targetToHashWords(target: Uint8Array): { hashLo: number; hashHi: number } {
  if (target.byteLength !== 8) {
    throw new Error(`target must be exactly 8 bytes; got ${target.byteLength}`);
  }
  const view = new DataView(target.buffer, target.byteOffset, target.byteLength);
  return {
    hashLo: view.getUint32(0, true),
    hashHi: view.getUint32(4, true),
  };
}

// ---------------------------------------------------------------------------
// Pure integer helpers, ported verbatim from src/gpu.rs so the adaptive
// per-checkpoint dispatch width matches the native CLI's schedule exactly.
// All magnitudes here (chain_len up to 2^31-1, steps up to ~length^2) stay
// far below 2^53, so plain JS numbers are exact — no BigInt needed.
// ---------------------------------------------------------------------------

export function divCeilU32(value: number, divisor: number): number {
  return Math.ceil(value / divisor);
}

export function alignUpU32(value: number, alignment: number): number {
  if (alignment <= 1) return value;
  return Math.ceil(value / alignment) * alignment;
}

export function alignTo(value: number, alignment: number): number {
  if (alignment <= 1) return value;
  return Math.ceil(value / alignment) * alignment;
}

export function maxDispatchInvocations(
  maxComputeWorkgroupsPerDimension: number,
  workgroup: number,
): number {
  return Math.max(
    Math.min(maxComputeWorkgroupsPerDimension * workgroup, MAX_HOST_DISPATCH_INVOCATIONS),
    workgroup,
  );
}

/** Total DES-reduction steps to compute every endpoint in `[start, start+length)`. */
export function stepsForRange(start: number, length: number): number {
  if (length === 0) return 0;
  return length * start + Math.floor((length * (length - 1)) / 2);
}

/**
 * Total DES-reduction steps a dispatch covering `[start, start+length)`
 * actually performs *within the current checkpoint window*
 * `[sliceStart, sliceStart+sliceSteps)` — i.e. clamped per-thread to at
 * most `sliceSteps` steps, since threads past the window's end just save
 * their intermediate state for the next checkpoint round.
 */
export function slicedStepsForRange(
  start: number,
  length: number,
  sliceStart: number,
  sliceSteps: number,
): number {
  if (length === 0 || sliceSteps === 0) return 0;
  const end = start + length - 1;
  const activeStart = Math.max(start, sliceStart + 1);
  if (activeStart > end) return 0;

  const rampEnd = Math.min(end, sliceStart + sliceSteps - 1);
  let steps = 0;
  if (activeStart <= rampEnd) {
    const count = rampEnd - activeStart + 1;
    const first = activeStart - sliceStart;
    const last = rampEnd - sliceStart;
    steps += Math.floor((count * (first + last)) / 2);
  }
  const fullStart = Math.max(activeStart, sliceStart + sliceSteps);
  if (fullStart <= end) {
    steps += (end - fullStart + 1) * sliceSteps;
  }
  return steps;
}

/**
 * Binary search (mirroring `sliced_dispatch_width`) for the widest dispatch
 * starting at `start`, within a checkpoint window `[sliceStart,
 * sliceStart+sliceSteps)`, whose total step cost stays within
 * `targetSteps`, then rounds the result down to a multiple of `alignment`
 * (the workgroup size) the same way the native code does.
 */
export function slicedDispatchWidth(
  start: number,
  remaining: number,
  sliceStart: number,
  sliceSteps: number,
  targetSteps: number,
  maximumWidth: number,
  alignment: number,
): number {
  const maximum = Math.max(Math.min(remaining, maximumWidth), 1);
  let low = 1;
  let high = maximum;
  let best = 1;
  while (low <= high) {
    const middle = low + Math.floor((high - low) / 2);
    if (slicedStepsForRange(start, middle, sliceStart, sliceSteps) <= targetSteps) {
      best = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  if (maximum >= alignment) {
    best = Math.max(Math.floor(best / alignment), 1) * alignment;
  }
  return Math.min(Math.max(best, 1), maximum);
}

// ---------------------------------------------------------------------------
// EWMA-adaptive step-budget scheduler, mirroring the smoothing in
// `precompute_with_progress`: `rate = old*0.75 + measured*0.25` (i.e. new
// measurements get weight `ADAPTIVE_ALPHA = 0.25`), and `target_steps =
// clamp(rate * budget_ms/1000, MIN_TARGET_STEPS, MAX_TARGET_STEPS)`.
// ---------------------------------------------------------------------------

export class AdaptiveStepScheduler {
  private smoothedRate: number | null = null;
  private targetSteps: number;

  constructor(
    private readonly budgetMs: number = DEFAULT_BUDGET_MS,
    initialTargetSteps: number = INITIAL_TARGET_STEPS,
  ) {
    this.targetSteps = clamp(initialTargetSteps, MIN_TARGET_STEPS, MAX_TARGET_STEPS);
  }

  get currentTargetSteps(): number {
    return this.targetSteps;
  }

  get currentSmoothedRate(): number | null {
    return this.smoothedRate;
  }

  /** Record one completed dispatch's measured cost and update the budget. */
  recordBatch(batchSteps: number, elapsedMs: number): number {
    const elapsedSeconds = Math.max(elapsedMs / 1000, 0.001);
    const measured = batchSteps / elapsedSeconds;
    const rate =
      this.smoothedRate === null
        ? measured
        : this.smoothedRate * (1 - ADAPTIVE_ALPHA) + measured * ADAPTIVE_ALPHA;
    this.smoothedRate = rate;
    const desired = Math.floor((rate * this.budgetMs) / 1000);
    this.targetSteps = clamp(desired, MIN_TARGET_STEPS, MAX_TARGET_STEPS);
    return this.targetSteps;
  }
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

// ---------------------------------------------------------------------------
// Completion-marker validation. The precompute shader never sets the
// "found" magic (that's only the false-alarm/verify shader's concern), so
// every marker must equal COMPLETION_MAGIC exactly — any other value is a
// dispatch failure, not something to skip past silently.
// ---------------------------------------------------------------------------

export function validateMarkers(markerWords: Uint32Array, groups: number): void {
  for (let index = 0; index < groups; index += 1) {
    const marker = markerWords[index];
    if (marker !== COMPLETION_MAGIC) {
      const hex = marker === undefined ? "undefined" : `0x${marker.toString(16).padStart(8, "0")}`;
      throw new Error(
        `GPU precompute dispatch did not publish a completion marker for workgroup ${index} ` +
          `(expected 0x${COMPLETION_MAGIC.toString(16).padStart(8, "0")}, got ${hex})`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Shader loading / pipeline creation.
// ---------------------------------------------------------------------------

// Bundle the WGSL text directly via Vite's `?raw` so there's no extra
// network round trip and no dependency on the static-file server being
// configured correctly at dev time. These are the single source of truth;
// web/public/shaders/ only carries a copy for the documented convention
// and so des_lut.bin (a binary asset, which can't be `?raw`-imported
// usefully) has somewhere to be fetched from at runtime.
import precomputeCompactSource from "../../../shaders/precompute_compact.wgsl?raw";
import precomputeExpandedSource from "../../../shaders/precompute_expanded.wgsl?raw";

function shaderSource(variant: ShaderVariant): string {
  return variant === "compact" ? precomputeCompactSource : precomputeExpandedSource;
}

/**
 * Compile the precompute compute pipeline for one shader variant, with the
 * `WORKGROUP_SIZE` pipeline-overridable constant bound to `workgroupSize`
 * (mirrors `compile_pipeline`'s `PipelineCompilationOptions.constants`).
 */
export function createPrecomputePipeline(
  device: GPUDevice,
  variant: ShaderVariant,
  workgroupSize: number,
): GPUComputePipeline {
  const module = device.createShaderModule({
    label: `ntlmrain precompute ${variant} WG ${workgroupSize}`,
    code: shaderSource(variant),
  });
  return device.createComputePipeline({
    label: `ntlmrain precompute ${variant} WG ${workgroupSize}`,
    layout: "auto",
    compute: {
      module,
      entryPoint: "main",
      constants: { WORKGROUP_SIZE: workgroupSize },
    },
  });
}

/** Fetch `des_lut.bin` once and upload it to a STORAGE `GPUBuffer`, unchanged. */
export async function loadDesLut(
  device: GPUDevice,
  url = "/shaders/des_lut.bin",
): Promise<GPUBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch DES LUT from ${url}: ${response.status} ${response.statusText}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== DES_LUT_BYTES) {
    throw new Error(`DES LUT at ${url} is ${bytes.byteLength} bytes; expected ${DES_LUT_BYTES}`);
  }
  const buffer = device.createBuffer({
    label: "ntlmrain DES LUT",
    size: alignTo(bytes.byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  return buffer;
}

// ---------------------------------------------------------------------------
// Main precompute loop.
// ---------------------------------------------------------------------------

export interface PrecomputeProgress {
  stepsDone: number;
  stepsTotal: number;
  batchSteps: number;
  batchWorkgroups: number;
  elapsedMs: number;
  targetSteps: number;
}

export interface PrecomputeOptions {
  /** 8-byte NetNTLMv1 hash (or byte7 target) to precompute chains toward. */
  target: Uint8Array;
  chainLen?: number;
  /** 0 for a single-target lookup; matches native's `table_index`. */
  tableIndex?: number;
  checkpointSteps?: number;
  /** Compiled with `createPrecomputePipeline`; caller picks the workgroup size. */
  workgroupSize: number;
  /** Target time budget per dispatch, in ms. Default 800 (see module docs). */
  budgetMs?: number;
  onProgress?: (progress: PrecomputeProgress) => void;
}

export interface PrecomputeDeviceLimits {
  maxComputeWorkgroupsPerDimension: number;
  minStorageBufferOffsetAlignment: number;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}

/** Mirrors `ensure_storage_size` in src/gpu.rs: fail fast with a clear
 * message instead of letting `device.createBuffer`/`createBindGroup` throw
 * an opaque validation error deep inside the dispatch loop. */
function ensureStorageSize(limits: PrecomputeDeviceLimits, bytes: number, label: string): void {
  if (bytes > limits.maxBufferSize) {
    throw new Error(`${label} needs ${bytes} bytes; device max buffer size is ${limits.maxBufferSize}`);
  }
  if (bytes > limits.maxStorageBufferBindingSize) {
    throw new Error(
      `${label} needs ${bytes} bytes; device max storage binding is ${limits.maxStorageBufferBindingSize}`,
    );
  }
}

/**
 * Run the checkpointed adaptive precompute loop and return exactly
 * `chainLen - 1` endpoints (881,688 by default) in ascending
 * target-position (file-record) order, ready for
 * `crypto-wasm`'s `encode_endpoint_file`.
 */
export async function runPrecompute(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  lutBuffer: GPUBuffer,
  deviceLimits: PrecomputeDeviceLimits,
  options: PrecomputeOptions,
): Promise<BigUint64Array> {
  const chainLen = options.chainLen ?? DEFAULT_CHAIN_LEN;
  const tableIndex = options.tableIndex ?? 0;
  const workgroup = options.workgroupSize;
  const budgetMs = options.budgetMs ?? DEFAULT_BUDGET_MS;
  const outputLen = chainLen - 1;
  const outputBytes = outputLen * 8;
  ensureStorageSize(deviceLimits, outputBytes, "endpoint output");

  const queue = device.queue;

  const output = device.createBuffer({
    label: "ntlmrain endpoint output",
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const staging = device.createBuffer({
    label: "ntlmrain endpoint readback",
    size: outputBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const uniform = device.createBuffer({
    label: "ntlmrain precompute parameters",
    size: PARAMS_BYTE_LENGTH,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const checkpointSteps = alignUpU32(Math.max(options.checkpointSteps ?? DEFAULT_CHECKPOINT_STEPS, 1), workgroup);
  const maximumWidth = Math.min(
    maxDispatchInvocations(deviceLimits.maxComputeWorkgroupsPerDimension, workgroup),
    outputLen,
  );
  const maximumGroups = divCeilU32(maximumWidth, workgroup);
  const markerBytes = alignTo(Math.max(maximumGroups, 1) * 4, 8);
  const markers = device.createBuffer({
    label: "ntlmrain completion markers",
    size: markerBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const markerReadback = device.createBuffer({
    label: "ntlmrain completion marker readback",
    size: markerBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const { hashLo, hashHi } = targetToHashWords(options.target);
  const reductionOffset = (tableIndex * 65_536) >>> 0;

  const scheduler = new AdaptiveStepScheduler(budgetMs);
  let stepsDone = 0;
  const stepsTotal = stepsForRange(0, outputLen);
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  for (let sliceStart = 0; sliceStart < outputLen; sliceStart += checkpointSteps) {
    let start = sliceStart;
    while (start < outputLen) {
      const remaining = outputLen - start;
      const targetSteps = scheduler.currentTargetSteps;
      const length = slicedDispatchWidth(
        start,
        remaining,
        sliceStart,
        checkpointSteps,
        targetSteps,
        maximumWidth,
        workgroup,
      );
      const batchSteps = slicedStepsForRange(start, length, sliceStart, checkpointSteps);
      const groups = divCeilU32(length, workgroup);
      const activeMarkerBytes = alignTo(Math.max(groups, 1) * 4, 8);

      const offsetBytes = start * 8;
      if (offsetBytes % deviceLimits.minStorageBufferOffsetAlignment !== 0) {
        throw new Error(
          `precompute dispatch offset ${offsetBytes} is not a multiple of the device's ` +
            `minStorageBufferOffsetAlignment (${deviceLimits.minStorageBufferOffsetAlignment}); ` +
            "this should be unreachable when checkpoint/slice widths are aligned to the workgroup size",
        );
      }

      const paramsBuffer = buildParamsBuffer({
        hashLo,
        hashHi,
        reductionOffset,
        chainLen,
        endpointStart: start,
        sliceStart,
        sliceSteps: checkpointSteps,
        outputLen: length,
      });
      queue.writeBuffer(uniform, 0, paramsBuffer);

      const bindGroup = device.createBindGroup({
        label: "ntlmrain precompute bindings",
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: output, offset: offsetBytes, size: length * 8 } },
          { binding: 2, resource: { buffer: lutBuffer } },
          { binding: 3, resource: { buffer: markers } },
        ],
      });

      const encoder = device.createCommandEncoder({ label: "ntlmrain precompute dispatch" });
      encoder.clearBuffer(markers, 0, activeMarkerBytes);
      const pass = encoder.beginComputePass({ label: "ntlmrain precompute pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(groups);
      pass.end();
      encoder.copyBufferToBuffer(markers, 0, markerReadback, 0, activeMarkerBytes);

      const started = performance.now();
      queue.submit([encoder.finish()]);

      await markerReadback.mapAsync(GPUMapMode.READ, 0, activeMarkerBytes);
      const markerData = new Uint32Array(markerReadback.getMappedRange(0, activeMarkerBytes).slice(0));
      markerReadback.unmap();
      const elapsedMs = performance.now() - started;

      validateMarkers(markerData, groups);

      scheduler.recordBatch(batchSteps, elapsedMs);

      start += length;
      stepsDone += batchSteps;

      options.onProgress?.({
        stepsDone: Math.min(stepsDone, stepsTotal),
        stepsTotal,
        batchSteps,
        batchWorkgroups: groups,
        elapsedMs,
        targetSteps: scheduler.currentTargetSteps,
      });
    }
  }

  const copyEncoder = device.createCommandEncoder({ label: "ntlmrain endpoint copy" });
  copyEncoder.copyBufferToBuffer(output, 0, staging, 0, outputBytes);
  queue.submit([copyEncoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ, 0, outputBytes);
  const words = new Uint32Array(staging.getMappedRange(0, outputBytes).slice(0));
  staging.unmap();

  // The shader writes each endpoint at a GPU slot indexed by its distance
  // from the chain's end (slot i finished after i reduction steps), which
  // is the reverse of the file format's monotonically increasing endpoint
  // ordinals. GPU slot i therefore belongs at file record
  // output_len - 1 - i: reverse while unpacking rather than skip this step.
  const endpoints = new BigUint64Array(outputLen);
  for (let source = 0; source < outputLen; source += 1) {
    const lo = BigInt(words[source * 2]);
    const hi = BigInt(words[source * 2 + 1]);
    endpoints[outputLen - 1 - source] = lo | (hi << 32n);
  }

  return endpoints;
}
