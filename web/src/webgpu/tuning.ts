// Browser port of the native CLI's staged GPU auto-tune benchmark
// (`src/gpu.rs::auto_tune_device`, roughly lines 1761-2332 as of this port).
// Every constant, formula, and control-flow decision below was checked
// against that function (and its helpers: `median_f64`,
// `coefficient_of_variation`, `aggregate_tuning_samples`,
// `tuning_uncertainty`, `confirmed_slow_tuning_rates`,
// `alternating_tuning_order`, `stable_tuning_winner`,
// `calibrated_synthetic_steps`, `production_tuning_shape`,
// `production_tuning_measurements`, `benchmark_compiled_candidate`) rather
// than approximated — a wrong tie-break or weighting here doesn't crash,
// it just quietly picks a slower shader/workgroup combination.
//
// The 7 stages, in order, mirror `auto_tune_device`'s comments exactly:
//   1. validation          - compile+benchmark the Compact/WG64 baseline,
//                             capture its output as `referenceWords`.
//   2. pilot                - a few progressively-calibrated Compact runs;
//                             short-circuits to a safe default if the
//                             device is confirmed slow.
//   3. family-validation    - benchmark each supported shader variant at
//                             the baseline workgroup; drop any whose
//                             output doesn't match `referenceWords`.
//   4. family               - 3 alternating-order rounds per surviving
//                             variant, aggregated by median + CV.
//   5. workgroup-validation - benchmark every (surviving variant, valid
//                             workgroup) pair; drop mismatches.
//   6. workgroup            - 3 alternating-order rounds per pair.
//   7. production           - measure finalists at 3 points along the real
//                             chain, combine via weighted harmonic mean,
//                             then run `stableTuningWinner`'s tie-break /
//                             hysteresis logic (with up to 2 extra rounds
//                             if the leaders remain tied).
//
// `benchmark_compiled_candidate` in `gpu.rs` always compiles the
// *precompute* shader (never the false-alarm shader) for every tuning
// stage, so this file only ever calls `createPrecomputePipeline` from
// `./precompute` — there is no separate pipeline-compile path here.

import {
  DEFAULT_CHAIN_LEN,
  DEFAULT_CHECKPOINT_STEPS,
  MAX_HOST_DISPATCH_INVOCATIONS,
  buildParamsBuffer,
  createPrecomputePipeline,
  divCeilU32,
  alignTo,
  slicedStepsForRange,
  targetToHashWords,
  validateMarkers,
  type ShaderVariant,
} from "./precompute";
import { EXPANDED_WORKGROUP_STORAGE_BYTES } from "./device";

// ---------------------------------------------------------------------------
// Constants, ported verbatim from `src/gpu.rs`.
// ---------------------------------------------------------------------------

/** `TUNING_BUDGET = Duration::from_secs(15)`. */
export const TUNING_BUDGET_MS = 15_000;
/** `TUNING_SLOW_RATE`. */
export const TUNING_SLOW_RATE = 10_000_000;
/** `TUNING_PILOT_STEPS`. */
export const TUNING_PILOT_STEPS = 1_000_000;
/** `TUNING_MIN_SAMPLE_MS`. */
export const TUNING_MIN_SAMPLE_MS = 20.0;
/** `TUNING_FAMILY_SAMPLE_MS`. */
export const TUNING_FAMILY_SAMPLE_MS = 50.0;
/** `TUNING_WORKGROUP_SAMPLE_MS`. */
export const TUNING_WORKGROUP_SAMPLE_MS = 35.0;
/** `TUNING_PRODUCTION_SAMPLE_MS`. */
export const TUNING_PRODUCTION_SAMPLE_MS = 150.0;
/** `TUNING_SAMPLE_ROUNDS`. */
export const TUNING_SAMPLE_ROUNDS = 3;
/** `TuningSelection.minimum_workgroups` is hard-coded to 128 in `gpu.rs`. */
const MINIMUM_WORKGROUPS = 128;
/** Fixed pilot dispatch width (`pilot_invocations` in `auto_tune_device`). */
const PILOT_INVOCATIONS = 4_096;
/** Fixed validation dispatch shape, reused across validation/family-validation/workgroup-validation. */
const VALIDATION_SHAPE: SyntheticBenchmarkShape = { kind: "synthetic", invocations: 4_096, steps: 64 };
/** Dummy 8-byte synthetic target used for every tuning benchmark dispatch. */
const SYNTHETIC_TARGET = new Uint8Array([0x25, 0x77, 0x89, 0x87, 0x04, 0x01, 0xc9, 0x65]);
/** Synthetic validation/pilot/family/workgroup benchmark `chain_len`. */
const SYNTHETIC_CHAIN_LEN = 1_025;

/**
 * `(fraction, weight)` pairs for the production stage's 3 measurement
 * points along the chain, and the weighted-harmonic-mean weights:
 * `POINTS: [(f64, f64); 3] = [(0.20, 0.16), (0.60, 0.40), (0.88, 0.44)]`.
 */
const PRODUCTION_POINTS: ReadonlyArray<readonly [number, number]> = [
  [0.2, 0.16],
  [0.6, 0.4],
  [0.88, 0.44],
];

/**
 * Rust's `f64::MIN_POSITIVE` (smallest *normal* `f64`), used verbatim as the
 * floor in `production_tuning_measurements`'s `1.0 / weighted_inverse.max(f64::MIN_POSITIVE)`.
 * Deliberately not `Number.MIN_VALUE` (JS's smallest *subnormal* double,
 * ~5e-324) — that's a different, much smaller floor than Rust's.
 */
const F64_MIN_POSITIVE = 2.2250738585072014e-308;

/** `ShaderVariant::workgroup_storage_bytes()`. */
const WORKGROUP_STORAGE_BYTES: Record<ShaderVariant, number> = {
  compact: 2_180,
  expanded: EXPANDED_WORKGROUP_STORAGE_BYTES,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TuningKey {
  shader: ShaderVariant;
  workgroupSize: number;
}

export interface TuningMeasurement {
  shader: ShaderVariant;
  workgroupSize: number;
  stage: string;
  elapsedMs: number;
  steps: number;
  stepsPerSecond: number;
  valid: boolean;
  note?: string;
  sampleCount: number;
  rateCv: number;
}

export interface TuningSelection {
  shader: ShaderVariant;
  workgroupSize: number;
  minimumWorkgroups: number;
  estimatedStepsPerSecond: number;
  source: string;
  selectionReason: string;
  tuningElapsedMs: number;
  deadlineReached: boolean;
  measurements: TuningMeasurement[];
}

export interface StableTuningOutcome {
  key: TuningKey;
  reason: string;
  tied: TuningKey[];
}

function keysEqual(a: TuningKey, b: TuningKey): boolean {
  return a.shader === b.shader && a.workgroupSize === b.workgroupSize;
}

function keyOf(key: TuningKey): string {
  return `${key.shader}:${key.workgroupSize}`;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

// ---------------------------------------------------------------------------
// Pure aggregation / statistics helpers (`median_f64`, `coefficient_of_variation`,
// `aggregate_tuning_samples`, `tuning_uncertainty`, `confirmed_slow_tuning_rates`,
// `alternating_tuning_order`). These have no GPU dependency and are covered by
// unit tests indirectly through `stableTuningWinner` and directly below.
// ---------------------------------------------------------------------------

export function medianF64(values: number[]): number {
  const sorted = values.filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
  return sorted[middle];
}

export function coefficientOfVariation(values: number[]): number {
  const valid = values.filter((value) => Number.isFinite(value) && value > 0);
  if (valid.length < 2) return 0;
  const mean = valid.reduce((sum, value) => sum + value, 0) / valid.length;
  if (mean <= 0) return 0;
  const variance =
    valid.reduce((sum, value) => {
      const difference = value - mean;
      return sum + difference * difference;
    }, 0) / valid.length;
  return Math.sqrt(variance) / mean;
}

export function aggregateTuningSamples(
  key: TuningKey,
  stage: string,
  samples: TuningMeasurement[],
): TuningMeasurement | null {
  const valid = samples.filter((sample) => sample.valid);
  if (valid.length === 0) return null;
  const rates = valid.map((sample) => sample.stepsPerSecond);
  return {
    shader: key.shader,
    workgroupSize: key.workgroupSize,
    stage,
    elapsedMs: valid.reduce((sum, sample) => sum + sample.elapsedMs, 0),
    steps: valid.reduce((sum, sample) => sum + sample.steps, 0),
    stepsPerSecond: medianF64(rates),
    valid: true,
    note: undefined,
    sampleCount: valid.length,
    rateCv: coefficientOfVariation(rates),
  };
}

/** `tuning_uncertainty`: `(2.0 * measurement.rate_cv).clamp(0.03, 0.10)`. */
export function tuningUncertainty(measurement: TuningMeasurement): number {
  return clamp(2 * measurement.rateCv, 0.03, 0.1);
}

/** `confirmed_slow_tuning_rates`. */
export function confirmedSlowTuningRates(rates: number[]): boolean {
  const valid = rates.filter((rate) => Number.isFinite(rate) && rate > 0);
  return valid.length >= 2 && medianF64(valid) < TUNING_SLOW_RATE;
}

/** `alternating_tuning_order`: reverses on odd rounds to cancel thermal drift. */
export function alternatingTuningOrder(keys: TuningKey[], round: number): TuningKey[] {
  const order = [...keys];
  if (round % 2 === 1) order.reverse();
  return order;
}

// ---------------------------------------------------------------------------
// Winner selection: `stable_tuning_winner`. Ranks by production score
// descending, builds a noise-tolerance "tie set" around the leader, then
// applies hysteresis (keep the cached incumbent) / Compact+WG64 preference /
// (workgroup_storage_bytes, |workgroup-64|) minimization in that order.
// ---------------------------------------------------------------------------

export function stableTuningWinner(
  measurements: TuningMeasurement[],
  incumbent: TuningKey | null,
  deadlineReached: boolean,
): StableTuningOutcome | null {
  const ranked = measurements
    .filter((measurement) => measurement.valid && measurement.stepsPerSecond > 0)
    .sort((a, b) => b.stepsPerSecond - a.stepsPerSecond);
  const leader = ranked[0];
  if (!leader) return null;

  const tied = ranked.filter((candidate) => {
    const gap = Math.max(leader.stepsPerSecond - candidate.stepsPerSecond, 0) / Math.max(leader.stepsPerSecond, 1);
    return gap <= Math.max(tuningUncertainty(leader), tuningUncertainty(candidate));
  });
  const tiedKeys = tied.map((measurement) => ({
    shader: measurement.shader,
    workgroupSize: measurement.workgroupSize,
  }));

  if (incumbent && tiedKeys.some((key) => keysEqual(key, incumbent))) {
    return { key: incumbent, reason: "cached-winner-retained", tied: tiedKeys };
  }

  const compactWg64: TuningKey = { shader: "compact", workgroupSize: 64 };
  if (tiedKeys.some((key) => keysEqual(key, compactWg64))) {
    return {
      key: compactWg64,
      reason: tieBreakReason(tiedKeys.length, deadlineReached),
      tied: tiedKeys,
    };
  }

  let selected: TuningMeasurement | null = null;
  for (const candidate of tied) {
    if (selected === null || compareTieBreak(candidate, selected) < 0) {
      selected = candidate;
    }
  }
  if (!selected) return null;
  return {
    key: { shader: selected.shader, workgroupSize: selected.workgroupSize },
    reason: tieBreakReason(tiedKeys.length, deadlineReached),
    tied: tiedKeys,
  };
}

function tieBreakReason(tiedCount: number, deadlineReached: boolean): string {
  if (tiedCount === 1) return "clear-winner";
  return deadlineReached ? "deadline-tie-break" : "deterministic-tie-break";
}

/** `(workgroup_storage_bytes, |workgroup_size - 64|, shader.id(), workgroup_size)` minimization. */
function compareTieBreak(a: TuningMeasurement, b: TuningMeasurement): number {
  const storageDelta = WORKGROUP_STORAGE_BYTES[a.shader] - WORKGROUP_STORAGE_BYTES[b.shader];
  if (storageDelta !== 0) return storageDelta;
  const wgDelta = Math.abs(a.workgroupSize - 64) - Math.abs(b.workgroupSize - 64);
  if (wgDelta !== 0) return wgDelta;
  if (a.shader !== b.shader) return a.shader < b.shader ? -1 : 1;
  return a.workgroupSize - b.workgroupSize;
}

// ---------------------------------------------------------------------------
// Synthetic-rate calibration and production-shape helpers
// (`calibrated_synthetic_steps`, `production_tuning_shape`,
// `production_tuning_measurements`).
// ---------------------------------------------------------------------------

export function calibratedSyntheticSteps(rate: number, targetMs: number, invocations: number): number {
  if (!Number.isFinite(rate) || rate <= 0 || invocations === 0) return 1;
  const value = Math.ceil((rate * targetMs) / 1000 / invocations);
  return clamp(value, 1, 0xffff_ffff);
}

interface ProductionBenchmarkShape {
  kind: "production";
  endpointStart: number;
  invocations: number;
  sliceSteps: number;
}

interface SyntheticBenchmarkShape {
  kind: "synthetic";
  invocations: number;
  steps: number;
}

type BenchmarkShape = SyntheticBenchmarkShape | ProductionBenchmarkShape;

/** `production_tuning_shape`. */
export function productionTuningShape(rate: number, fraction: number, targetMs: number): ProductionBenchmarkShape {
  const endpointStart = Math.trunc((DEFAULT_CHAIN_LEN - 1) * fraction);
  const invocations = Math.min(MAX_HOST_DISPATCH_INVOCATIONS, DEFAULT_CHAIN_LEN - 1 - endpointStart);
  const sliceSteps = clamp(calibratedSyntheticSteps(rate, targetMs, invocations), 1, DEFAULT_CHECKPOINT_STEPS);
  return { kind: "production", endpointStart, invocations, sliceSteps };
}

/**
 * `production_tuning_measurements`: combine each finalist's 3 measured
 * points into a single score via the weighted harmonic mean
 * `rate = 1 / Σ(weight_i / rate_i)`, dropping any finalist missing a point.
 */
export function productionTuningMeasurements(
  finalists: TuningKey[],
  samples: Map<string, TuningMeasurement[]>,
): TuningMeasurement[] {
  const results: TuningMeasurement[] = [];
  for (const key of finalists) {
    let weightedInverse = 0;
    let elapsedMs = 0;
    let steps = 0;
    let sampleCount = 0;
    let maximumCv = 0;
    let complete = true;
    for (let pointIndex = 0; pointIndex < PRODUCTION_POINTS.length; pointIndex += 1) {
      const weight = PRODUCTION_POINTS[pointIndex][1];
      const pointSamples = samples.get(productionSampleKey(key, pointIndex)) ?? [];
      const point = aggregateTuningSamples(key, "production-point", pointSamples);
      if (!point) {
        complete = false;
        break;
      }
      weightedInverse += weight / Math.max(point.stepsPerSecond, 1);
      elapsedMs += point.elapsedMs;
      steps += point.steps;
      sampleCount += point.sampleCount;
      maximumCv = Math.max(maximumCv, point.rateCv);
    }
    if (complete) {
      results.push({
        shader: key.shader,
        workgroupSize: key.workgroupSize,
        stage: "production",
        elapsedMs,
        steps,
        stepsPerSecond: 1 / Math.max(weightedInverse, F64_MIN_POSITIVE),
        valid: true,
        note: undefined,
        sampleCount,
        rateCv: maximumCv,
      });
    }
  }
  return results;
}

function productionSampleKey(key: TuningKey, pointIndex: number): string {
  return `${keyOf(key)}::${pointIndex}`;
}

/** `valid_workgroups`: the native candidate set, filtered by what the device supports. */
export function validWorkgroups(limits: {
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
}): number[] {
  return [32, 64, 128, 256, 512, 1_024].filter(
    (size) => size <= limits.maxComputeInvocationsPerWorkgroup && size <= limits.maxComputeWorkgroupSizeX,
  );
}

function failedMeasurement(shader: ShaderVariant, workgroupSize: number, stage: string, note: string): TuningMeasurement {
  return {
    shader,
    workgroupSize,
    stage,
    elapsedMs: 0,
    steps: 0,
    stepsPerSecond: 0,
    valid: false,
    note,
    sampleCount: 0,
    rateCv: 0,
  };
}

// ---------------------------------------------------------------------------
// Deadline: `TuningDeadline`.
// ---------------------------------------------------------------------------

class TuningDeadline {
  private readonly startedAt: number;

  constructor(
    private readonly limitMs: number,
    private readonly now: () => number = () => performance.now(),
  ) {
    this.startedAt = this.now();
  }

  elapsed(): number {
    return this.now() - this.startedAt;
  }

  remaining(): number {
    return Math.max(this.limitMs - this.elapsed(), 0);
  }

  /** `remaining_ms > expected_ms.max(0) + 10.0`. */
  canSchedule(expectedMs: number): boolean {
    return this.remaining() > Math.max(expectedMs, 0) + 10;
  }

  expired(): boolean {
    return this.elapsed() >= this.limitMs;
  }
}

// ---------------------------------------------------------------------------
// GPU benchmark dispatch: `benchmark_compiled_candidate`. Always compiles/
// runs the *precompute* shader (matching `tuning_pipeline`'s use of
// `key.0.precompute_source()`), never the false-alarm shader.
// ---------------------------------------------------------------------------

async function benchmarkCandidate(
  device: GPUDevice,
  lutBuffer: GPUBuffer,
  pipeline: GPUComputePipeline,
  variant: ShaderVariant,
  workgroupSize: number,
  stage: string,
  shape: BenchmarkShape,
): Promise<{ measurement: TuningMeasurement; words: Uint32Array }> {
  const { hashLo, hashHi } = targetToHashWords(SYNTHETIC_TARGET);
  let invocations: number;
  let paramsBuffer: ArrayBuffer;
  let steps: number;

  if (shape.kind === "synthetic") {
    invocations = shape.invocations;
    paramsBuffer = buildParamsBuffer({
      hashLo,
      hashHi,
      reductionOffset: 0,
      chainLen: SYNTHETIC_CHAIN_LEN,
      endpointStart: 0,
      sliceStart: 0,
      sliceSteps: 0xffff_ffff,
      outputLen: invocations,
      benchmarkSteps: shape.steps,
      mode: 1,
    });
    steps = invocations * shape.steps;
  } else {
    invocations = shape.invocations;
    paramsBuffer = buildParamsBuffer({
      hashLo,
      hashHi,
      reductionOffset: 0,
      chainLen: DEFAULT_CHAIN_LEN,
      endpointStart: shape.endpointStart,
      sliceStart: 0,
      sliceSteps: shape.sliceSteps,
      outputLen: invocations,
      benchmarkSteps: 0,
      mode: 0,
    });
    steps = slicedStepsForRange(shape.endpointStart, invocations, 0, shape.sliceSteps);
  }

  const groups = divCeilU32(invocations, workgroupSize);
  const outputBytes = invocations * 8;
  const markerBytes = alignTo(Math.max(groups, 1) * 4, 8);

  const queue = device.queue;
  const output = device.createBuffer({
    label: "ntlmrain tune output",
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: "ntlmrain tune readback",
    size: outputBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const uniform = device.createBuffer({
    label: "ntlmrain tune params",
    size: paramsBuffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  queue.writeBuffer(uniform, 0, paramsBuffer);
  const markers = device.createBuffer({
    label: "ntlmrain tune markers",
    size: markerBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const markerReadback = device.createBuffer({
    label: "ntlmrain tune marker readback",
    size: markerBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    label: "ntlmrain tune bindings",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: lutBuffer } },
      { binding: 3, resource: { buffer: markers } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "ntlmrain tune dispatch" });
  encoder.clearBuffer(markers, 0, markerBytes);
  const pass = encoder.beginComputePass({ label: "ntlmrain tune pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(groups);
  pass.end();
  encoder.copyBufferToBuffer(markers, 0, markerReadback, 0, markerBytes);
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);

  const started = performance.now();
  queue.submit([encoder.finish()]);

  await markerReadback.mapAsync(GPUMapMode.READ, 0, markerBytes);
  const markerData = new Uint32Array(markerReadback.getMappedRange(0, markerBytes).slice(0));
  markerReadback.unmap();
  const elapsedMs = performance.now() - started;

  validateMarkers(markerData, groups);

  await readback.mapAsync(GPUMapMode.READ, 0, outputBytes);
  const words = new Uint32Array(readback.getMappedRange(0, outputBytes).slice(0));
  readback.unmap();

  const elapsedSeconds = Math.max(elapsedMs / 1000, 0.000_001);
  return {
    measurement: {
      shader: variant,
      workgroupSize,
      stage,
      elapsedMs: elapsedSeconds * 1000,
      steps,
      stepsPerSecond: steps / elapsedSeconds,
      valid: true,
      note: undefined,
      sampleCount: 1,
      rateCv: 0,
    },
    words,
  };
}

function wordsEqual(a: Uint32Array, b: Uint32Array): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Full staged auto-tune orchestration: `auto_tune_device`.
// ---------------------------------------------------------------------------

export interface AutoTuneDeviceLimits {
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
}

export interface AutoTuneOptions {
  device: GPUDevice;
  lutBuffer: GPUBuffer;
  limits: AutoTuneDeviceLimits;
  /** Shader variants the device can actually run; Compact always, Expanded iff `supportsExpandedShader()`. */
  supportedShaders: ShaderVariant[];
  forcedVariant?: ShaderVariant;
  forcedWorkgroup?: number;
  incumbent?: TuningKey;
  /** Injectable clock for deterministic testing; defaults to `performance.now`. */
  now?: () => number;
}

export async function autoTuneDevice(options: AutoTuneOptions): Promise<TuningSelection> {
  const { device, lutBuffer, limits } = options;
  const deadline = new TuningDeadline(TUNING_BUDGET_MS, options.now);
  const variants = options.forcedVariant ? [options.forcedVariant] : [...options.supportedShaders];
  const supportedWorkgroups = validWorkgroups(limits);
  if (supportedWorkgroups.length === 0) {
    throw new Error("no supported workgroup size for this device");
  }
  // A manual override must be validated against what the negotiated device
  // actually supports (`limits.maxComputeInvocationsPerWorkgroup` /
  // `maxComputeWorkgroupSizeX`) — otherwise picking 512/1024 on a device
  // that never had those limits raised (see `device.ts`) creates a pipeline
  // with an unsupported workgroup size and WebGPU throws an opaque
  // validation error deep inside dispatch, instead of this clear message.
  if (options.forcedWorkgroup !== undefined && !supportedWorkgroups.includes(options.forcedWorkgroup)) {
    throw new Error(
      `workgroup size ${options.forcedWorkgroup} is not supported by this device ` +
        `(supported sizes: ${supportedWorkgroups.join(", ")})`,
    );
  }
  const candidateWorkgroups = options.forcedWorkgroup !== undefined ? [options.forcedWorkgroup] : supportedWorkgroups;
  const baselineWorkgroup = supportedWorkgroups.includes(64) ? 64 : supportedWorkgroups[0];
  const baselineKey: TuningKey = { shader: "compact", workgroupSize: baselineWorkgroup };
  const automaticRequest = options.forcedVariant === undefined && options.forcedWorkgroup === undefined;
  const incumbentKey = options.incumbent ?? null;

  const pipelines = new Map<string, GPUComputePipeline>();
  const getPipeline = (key: TuningKey): GPUComputePipeline => {
    const cacheKey = keyOf(key);
    let pipeline = pipelines.get(cacheKey);
    if (!pipeline) {
      pipeline = createPrecomputePipeline(device, key.shader, key.workgroupSize);
      pipelines.set(cacheKey, pipeline);
    }
    return pipeline;
  };

  const measurements: TuningMeasurement[] = [];

  // --- 1. validation ---------------------------------------------------
  const baselinePipeline = getPipeline(baselineKey);
  const validationResult = await benchmarkCandidate(
    device,
    lutBuffer,
    baselinePipeline,
    baselineKey.shader,
    baselineKey.workgroupSize,
    "validation",
    VALIDATION_SHAPE,
  );
  const baselineValidation = validationResult.measurement;
  const referenceWords = validationResult.words;
  measurements.push({ ...baselineValidation });

  // --- 2. pilot ----------------------------------------------------------
  let pilotSteps = Math.ceil(TUNING_PILOT_STEPS / PILOT_INVOCATIONS);
  const pilotSamples: TuningMeasurement[] = [];
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const last = pilotSamples[pilotSamples.length - 1];
    const expectedMs = last
      ? clamp(
          (PILOT_INVOCATIONS * pilotSteps * 1000) / Math.max(last.stepsPerSecond, 1),
          TUNING_MIN_SAMPLE_MS,
          2_000,
        )
      : TUNING_MIN_SAMPLE_MS;
    if (!deadline.canSchedule(expectedMs)) break;
    const { measurement: sample } = await benchmarkCandidate(
      device,
      lutBuffer,
      baselinePipeline,
      baselineKey.shader,
      baselineKey.workgroupSize,
      "pilot",
      { kind: "synthetic", invocations: PILOT_INVOCATIONS, steps: pilotSteps },
    );
    const longEnough = sample.elapsedMs >= TUNING_MIN_SAMPLE_MS;
    pilotSteps = Math.max(
      pilotSteps,
      calibratedSyntheticSteps(sample.stepsPerSecond, TUNING_MIN_SAMPLE_MS, PILOT_INVOCATIONS),
    );
    pilotSamples.push(sample);
    if (pilotSamples.length >= 2 && longEnough) break;
  }
  if (pilotSamples.length === 0) {
    pilotSamples.push({ ...baselineValidation });
  }
  const pilot = aggregateTuningSamples(baselineKey, "pilot", pilotSamples);
  if (!pilot) {
    throw new Error("GPU tuning failed: no valid pilot samples");
  }
  const pilotRate = pilot.stepsPerSecond;
  measurements.push({ ...pilot });

  if (
    automaticRequest &&
    confirmedSlowTuningRates(pilotSamples.map((sample) => sample.stepsPerSecond))
  ) {
    return {
      shader: "compact",
      workgroupSize: baselineWorkgroup,
      minimumWorkgroups: MINIMUM_WORKGROUPS,
      estimatedStepsPerSecond: pilotRate,
      source: "slow-adapter-default",
      selectionReason: "slow-adapter-default",
      tuningElapsedMs: deadline.elapsed(),
      deadlineReached: deadline.expired(),
      measurements,
    };
  }

  // --- 3. family-validation ----------------------------------------------
  const calibrationRates = new Map<string, number>();
  calibrationRates.set(keyOf(baselineKey), pilotRate);
  const familyKeys: TuningKey[] = [];
  for (const variant of variants) {
    const key: TuningKey = { shader: variant, workgroupSize: baselineWorkgroup };
    if (!deadline.canSchedule(TUNING_MIN_SAMPLE_MS)) break;
    let pipeline: GPUComputePipeline;
    try {
      pipeline = getPipeline(key);
    } catch (error) {
      measurements.push(failedMeasurement(key.shader, key.workgroupSize, "family-validation", String(error)));
      continue;
    }
    if (!keysEqual(key, baselineKey) && !deadline.canSchedule(TUNING_MIN_SAMPLE_MS)) break;

    let calibration: TuningMeasurement;
    if (keysEqual(key, baselineKey)) {
      calibration = { ...baselineValidation };
    } else {
      try {
        const { measurement, words } = await benchmarkCandidate(
          device,
          lutBuffer,
          pipeline,
          key.shader,
          key.workgroupSize,
          "family-validation",
          VALIDATION_SHAPE,
        );
        calibration = { ...measurement, valid: wordsEqual(words, referenceWords) };
        if (!calibration.valid) {
          calibration.note = "output differed from Compact WG64 reference";
        }
      } catch (error) {
        measurements.push(failedMeasurement(key.shader, key.workgroupSize, "family-validation", String(error)));
        continue;
      }
    }
    measurements.push({ ...calibration });
    if (calibration.valid) {
      calibrationRates.set(keyOf(key), calibration.stepsPerSecond);
      familyKeys.push(key);
    }
  }
  if (familyKeys.length === 0) {
    throw new Error("GPU tuning failed: no shader variant survived family-validation");
  }

  // --- 4. family -----------------------------------------------------------
  const familySamples = new Map<string, TuningMeasurement[]>();
  for (let round = 0; round < TUNING_SAMPLE_ROUNDS; round += 1) {
    const order = alternatingTuningOrder(familyKeys, round);
    for (const key of order) {
      if (!deadline.canSchedule(TUNING_FAMILY_SAMPLE_MS)) break;
      const pipeline = pipelines.get(keyOf(key));
      if (!pipeline) throw new Error("GPU tuning failed: missing pipeline for family key");
      const seed = calibrationRates.get(keyOf(key)) ?? pilotRate;
      try {
        const { measurement } = await benchmarkCandidate(
          device,
          lutBuffer,
          pipeline,
          key.shader,
          key.workgroupSize,
          "family-sample",
          {
            kind: "synthetic",
            invocations: MAX_HOST_DISPATCH_INVOCATIONS,
            steps: calibratedSyntheticSteps(seed, TUNING_FAMILY_SAMPLE_MS, MAX_HOST_DISPATCH_INVOCATIONS),
          },
        );
        calibrationRates.set(keyOf(key), measurement.stepsPerSecond);
        const list = familySamples.get(keyOf(key)) ?? [];
        list.push(measurement);
        familySamples.set(keyOf(key), list);
      } catch (error) {
        measurements.push(failedMeasurement(key.shader, key.workgroupSize, "family-sample", String(error)));
      }
    }
  }
  const familyMeasurements = familyKeys
    .map((key) => aggregateTuningSamples(key, "family", familySamples.get(keyOf(key)) ?? []))
    .filter((measurement): measurement is TuningMeasurement => measurement !== null);
  measurements.push(...familyMeasurements.map((measurement) => ({ ...measurement })));

  // --- 5. workgroup-validation ---------------------------------------------
  const survivingVariants = familyMeasurements.map((measurement) => measurement.shader);
  const workgroupKeys: TuningKey[] = [];
  for (const variant of survivingVariants) {
    for (const workgroup of candidateWorkgroups) {
      const key: TuningKey = { shader: variant, workgroupSize: workgroup };
      if (!deadline.canSchedule(TUNING_MIN_SAMPLE_MS)) break;
      let pipeline: GPUComputePipeline;
      try {
        pipeline = getPipeline(key);
      } catch (error) {
        measurements.push(failedMeasurement(key.shader, key.workgroupSize, "workgroup-validation", String(error)));
        continue;
      }
      const cached = calibrationRates.get(keyOf(key));
      if (cached === undefined && !deadline.canSchedule(TUNING_MIN_SAMPLE_MS)) break;

      let calibration: TuningMeasurement;
      if (cached !== undefined) {
        calibration = {
          shader: key.shader,
          workgroupSize: key.workgroupSize,
          stage: "workgroup-validation",
          elapsedMs: 0,
          steps: 0,
          stepsPerSecond: cached,
          valid: true,
          note: "reused family validation",
          sampleCount: 1,
          rateCv: 0,
        };
      } else {
        try {
          const { measurement, words } = await benchmarkCandidate(
            device,
            lutBuffer,
            pipeline,
            key.shader,
            key.workgroupSize,
            "workgroup-validation",
            VALIDATION_SHAPE,
          );
          calibration = { ...measurement, valid: wordsEqual(words, referenceWords) };
          if (!calibration.valid) {
            calibration.note = "output differed from Compact WG64 reference";
          }
        } catch (error) {
          measurements.push(
            failedMeasurement(key.shader, key.workgroupSize, "workgroup-validation", String(error)),
          );
          continue;
        }
      }
      measurements.push({ ...calibration });
      if (calibration.valid) {
        calibrationRates.set(keyOf(key), calibration.stepsPerSecond);
        workgroupKeys.push(key);
      }
    }
  }

  // --- 6. workgroup ----------------------------------------------------------
  const workgroupSamples = new Map<string, TuningMeasurement[]>();
  for (let round = 0; round < TUNING_SAMPLE_ROUNDS; round += 1) {
    const order = alternatingTuningOrder(workgroupKeys, round);
    for (const key of order) {
      if (!deadline.canSchedule(TUNING_WORKGROUP_SAMPLE_MS)) break;
      const pipeline = pipelines.get(keyOf(key));
      if (!pipeline) throw new Error("GPU tuning failed: missing pipeline for workgroup key");
      const seed = calibrationRates.get(keyOf(key)) ?? pilotRate;
      try {
        const { measurement } = await benchmarkCandidate(
          device,
          lutBuffer,
          pipeline,
          key.shader,
          key.workgroupSize,
          "workgroup-sample",
          {
            kind: "synthetic",
            invocations: MAX_HOST_DISPATCH_INVOCATIONS,
            steps: calibratedSyntheticSteps(seed, TUNING_WORKGROUP_SAMPLE_MS, MAX_HOST_DISPATCH_INVOCATIONS),
          },
        );
        calibrationRates.set(keyOf(key), measurement.stepsPerSecond);
        const list = workgroupSamples.get(keyOf(key)) ?? [];
        list.push(measurement);
        workgroupSamples.set(keyOf(key), list);
      } catch (error) {
        measurements.push(failedMeasurement(key.shader, key.workgroupSize, "workgroup-sample", String(error)));
      }
    }
  }
  const workgroupMeasurements = workgroupKeys
    .map((key) => aggregateTuningSamples(key, "workgroup", workgroupSamples.get(keyOf(key)) ?? []))
    .filter((measurement): measurement is TuningMeasurement => measurement !== null);
  measurements.push(...workgroupMeasurements.map((measurement) => ({ ...measurement })));

  // Keep the best workgroup from every family and Compact WG64. Preliminary
  // samples choose finalists only; the final score is production-shaped.
  let finalists: TuningKey[] = [];
  for (const variant of variants) {
    let best: TuningMeasurement | null = null;
    for (const measurement of workgroupMeasurements) {
      if (measurement.shader !== variant) continue;
      if (!best || measurement.stepsPerSecond > best.stepsPerSecond) best = measurement;
    }
    if (best) finalists.push({ shader: best.shader, workgroupSize: best.workgroupSize });
  }
  if (automaticRequest && pipelines.has(keyOf(baselineKey))) {
    finalists.push(baselineKey);
  }
  finalists.sort((a, b) => (a.shader === b.shader ? a.workgroupSize - b.workgroupSize : a.shader < b.shader ? -1 : 1));
  finalists = dedupeKeys(finalists);
  if (finalists.length === 0) {
    if (automaticRequest) {
      finalists.push(baselineKey);
    } else {
      throw new Error("GPU tuning failed: no finalist candidates");
    }
  }
  const rateOf = (key: TuningKey): number =>
    workgroupMeasurements.find((measurement) => keysEqual(measurement, key))?.stepsPerSecond ?? 0;
  finalists.sort((left, right) => {
    if (keysEqual(left, baselineKey)) return -1;
    if (keysEqual(right, baselineKey)) return 1;
    return rateOf(right) - rateOf(left);
  });

  // --- 7. production ---------------------------------------------------------
  const productionSamples = new Map<string, TuningMeasurement[]>();
  for (const key of finalists) {
    const pipeline = pipelines.get(keyOf(key));
    if (!pipeline) continue;
    const seedRate =
      workgroupMeasurements.find((measurement) => keysEqual(measurement, key))?.stepsPerSecond ?? pilotRate;
    for (let pointIndex = 0; pointIndex < PRODUCTION_POINTS.length; pointIndex += 1) {
      const fraction = PRODUCTION_POINTS[pointIndex][0];
      if (!deadline.canSchedule(TUNING_PRODUCTION_SAMPLE_MS)) break;
      const shape = productionTuningShape(seedRate, fraction, TUNING_PRODUCTION_SAMPLE_MS);
      try {
        await benchmarkCandidate(device, lutBuffer, pipeline, key.shader, key.workgroupSize, "production-warmup", shape);
      } catch {
        // Warm-up failures are silently ignored, matching `let _ = ...` in gpu.rs.
      }
      for (let round = 0; round < TUNING_SAMPLE_ROUNDS; round += 1) {
        if (!deadline.canSchedule(TUNING_PRODUCTION_SAMPLE_MS)) break;
        try {
          const { measurement } = await benchmarkCandidate(
            device,
            lutBuffer,
            pipeline,
            key.shader,
            key.workgroupSize,
            "production-sample",
            shape,
          );
          const list = productionSamples.get(productionSampleKey(key, pointIndex)) ?? [];
          list.push(measurement);
          productionSamples.set(productionSampleKey(key, pointIndex), list);
        } catch (error) {
          measurements.push(failedMeasurement(key.shader, key.workgroupSize, "production-sample", String(error)));
          break;
        }
      }
    }
  }

  let productionMeasurements = productionTuningMeasurements(finalists, productionSamples);
  let outcome = stableTuningWinner(productionMeasurements, incumbentKey, deadline.expired());

  // If the leading pair remains statistically tied, spend at most two more
  // rounds (per production point) on the top two tied candidates.
  if (outcome && outcome.tied.length > 1) {
    const extended = outcome.tied.slice(0, 2);
    for (let round = 0; round < 2; round += 1) {
      const order = round % 2 === 1 ? [...extended].reverse() : extended;
      for (const key of order) {
        const pipeline = pipelines.get(keyOf(key));
        if (!pipeline) continue;
        const seedRate =
          productionMeasurements.find((measurement) => keysEqual(measurement, key))?.stepsPerSecond ?? pilotRate;
        for (let pointIndex = 0; pointIndex < PRODUCTION_POINTS.length; pointIndex += 1) {
          const fraction = PRODUCTION_POINTS[pointIndex][0];
          if (!deadline.canSchedule(TUNING_PRODUCTION_SAMPLE_MS)) break;
          const shape = productionTuningShape(seedRate, fraction, TUNING_PRODUCTION_SAMPLE_MS);
          try {
            const { measurement } = await benchmarkCandidate(
              device,
              lutBuffer,
              pipeline,
              key.shader,
              key.workgroupSize,
              "production-extra",
              shape,
            );
            const list = productionSamples.get(productionSampleKey(key, pointIndex)) ?? [];
            list.push(measurement);
            productionSamples.set(productionSampleKey(key, pointIndex), list);
          } catch {
            // Extra-round failures are silently ignored, matching `if let Ok(...)` in gpu.rs.
          }
        }
      }
    }
    productionMeasurements = productionTuningMeasurements(finalists, productionSamples);
    outcome = stableTuningWinner(productionMeasurements, incumbentKey, deadline.expired());
  }
  measurements.push(...productionMeasurements.map((measurement) => ({ ...measurement })));

  let selectedKey: TuningKey;
  let estimatedStepsPerSecond: number;
  let selectionReason: string;
  if (outcome) {
    selectedKey = outcome.key;
    estimatedStepsPerSecond =
      productionMeasurements.find((measurement) => keysEqual(measurement, outcome!.key))?.stepsPerSecond ?? pilotRate;
    selectionReason = outcome.reason;
  } else if (automaticRequest) {
    selectedKey = baselineKey;
    estimatedStepsPerSecond = pilotRate;
    selectionReason = "deadline-tie-break";
  } else {
    const candidates = [...workgroupMeasurements, ...familyMeasurements].filter(
      (measurement) =>
        (options.forcedVariant === undefined || measurement.shader === options.forcedVariant) &&
        (options.forcedWorkgroup === undefined || measurement.workgroupSize === options.forcedWorkgroup),
    );
    let best: TuningMeasurement | null = null;
    for (const candidate of candidates) {
      if (!best || candidate.stepsPerSecond > best.stepsPerSecond) best = candidate;
    }
    if (!best) {
      throw new Error("GPU tuning failed: no candidate matched the forced variant/workgroup");
    }
    selectedKey = { shader: best.shader, workgroupSize: best.workgroupSize };
    estimatedStepsPerSecond = best.stepsPerSecond;
    selectionReason = "deadline-tie-break";
  }

  return {
    shader: selectedKey.shader,
    workgroupSize: selectedKey.workgroupSize,
    minimumWorkgroups: MINIMUM_WORKGROUPS,
    estimatedStepsPerSecond,
    source: options.forcedVariant !== undefined || options.forcedWorkgroup !== undefined ? "partial-override-tune" : "auto-tune",
    selectionReason,
    tuningElapsedMs: deadline.elapsed(),
    deadlineReached: deadline.expired(),
    measurements,
  };
}

function dedupeKeys(keys: TuningKey[]): TuningKey[] {
  const result: TuningKey[] = [];
  for (const key of keys) {
    const previous = result[result.length - 1];
    if (previous && keysEqual(previous, key)) continue;
    result.push(key);
  }
  return result;
}
