// Entry point: wires the recovery form, GPU settings, 3-stage progress
// display, results panel, and run archive (Tasks 2-5's modules, plus this
// task's own `archive/`, `ui/`) into the actual page defined by
// `index.html`.

import { requestGpuDevice, supportsExpandedShader } from "./webgpu/device";
import {
  createPrecomputePipeline,
  loadDesLut,
  precomputeCompactSource,
  precomputeExpandedSource,
  runPrecompute,
} from "./webgpu/precompute";
import { verifyCompactSource, verifyExpandedSource } from "./webgpu/false-alarm";
import { initCrypto } from "./crypto";
import { createDefaultPorts, runOrchestrator, type OrchestratorPorts } from "./pipeline/orchestrator";
import { autoTuneDevice, validWorkgroups, type TuningSelection } from "./webgpu/tuning";
import {
  adaptiveBudgetMs,
  applyManualTuningOverride,
  defaultGpuSettings,
  hasManualTuningOverride,
  type GpuSettingsState,
} from "./ui/gpu-settings";
import { validateInput } from "./ui/validation";
import { recoveredKeysFromIndices } from "./ui/key-format";
import { RunController, type RunSnapshot } from "./ui/run-controller";
import { ProgressView, renderArchiveDetail, renderArchiveList, renderInputFeedback, renderResults } from "./ui/view";
import {
  generateRunId,
  getRun,
  listRuns,
  putRun,
  type ArchivedResult,
  type RecoveryMode,
  type RunRecord,
} from "./archive/db";

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
}

const modeFullRadio = byId<HTMLInputElement>("nr-mode-full");
const modeSingleDesRadio = byId<HTMLInputElement>("nr-mode-single-des");
const inputEl = byId<HTMLTextAreaElement>("nr-input");
const inputFeedbackEl = byId<HTMLElement>("nr-input-feedback");
const shaderSelect = byId<HTMLSelectElement>("nr-shader");
const shaderExpandedOption = byId<HTMLOptionElement>("nr-shader-expanded");
const workgroupSelect = byId<HTMLSelectElement>("nr-workgroup-size");
const powerPreferenceSelect = byId<HTMLSelectElement>("nr-power-preference");
const dispatchAdaptiveRadio = byId<HTMLInputElement>("nr-dispatch-adaptive");
const dispatchFixedRadio = byId<HTMLInputElement>("nr-dispatch-fixed");
const adaptiveTargetInput = byId<HTMLInputElement>("nr-adaptive-target");
const fixedStepsSelect = byId<HTMLSelectElement>("nr-fixed-steps");
const forceRetuneCheckbox = byId<HTMLInputElement>("nr-force-retune");
const continueAutomaticallyCheckbox = byId<HTMLInputElement>("nr-continue-automatically");
const startButton = byId<HTMLButtonElement>("nr-start");
const resumeButton = byId<HTMLButtonElement>("nr-resume");
const progressPanel = byId<HTMLElement>("nr-progress-panel");
const progressSlotsEl = byId<HTMLElement>("nr-progress-slots");
const slotTemplate = byId<HTMLTemplateElement>("nr-slot-template");
const resultsPanel = byId<HTMLElement>("nr-results-panel");
const resultsBodyEl = byId<HTMLElement>("nr-results-body");
const archiveListEl = byId<HTMLElement>("nr-archive-list");
const archiveDetailEl = byId<HTMLElement>("nr-archive-detail");

const progressView = new ProgressView(progressPanel, progressSlotsEl, slotTemplate);

let inputTouched = false;
let selectedArchiveRunId: string | null = null;
let activeController: RunController | null = null;

let currentDevice: GPUDevice | null = null;
let currentAdapter: GPUAdapter | null = null;
let currentLutBuffer: GPUBuffer | null = null;
let currentLutBytes: Uint8Array | null = null;
/** The `powerPreference` the currently-held device/LUT buffer were negotiated for. */
let currentPowerPreference: GPUPowerPreference | null = null;

/**
 * `GPUSupportedLimits` is a branded/read-only host object, not a plain
 * record, so it can't be spread directly into `ArchivedDevice.limits`
 * (`Record<string, number>`) — copy just the numeric fields we can enumerate.
 */
function plainLimits(limits: GPUSupportedLimits): Record<string, number> {
  const result: Record<string, number> = {};
  for (const key in limits) {
    const value = (limits as unknown as Record<string, unknown>)[key];
    if (typeof value === "number") result[key] = value;
  }
  return result;
}

function currentMode(): RecoveryMode {
  return modeSingleDesRadio.checked ? "single-des" : "full";
}

function currentGpuSettings(): GpuSettingsState {
  const shader = shaderSelect.value as GpuSettingsState["shader"];
  const workgroupSize = workgroupSelect.value === "auto" ? "auto" : Number(workgroupSelect.value);
  return {
    shader,
    workgroupSize,
    dispatchMode: dispatchFixedRadio.checked ? "fixed" : "adaptive",
    adaptiveTargetSeconds: Number(adaptiveTargetInput.value) || defaultGpuSettings().adaptiveTargetSeconds,
    fixedSteps: Number(fixedStepsSelect.value) || defaultGpuSettings().fixedSteps,
    powerPreference: powerPreferenceSelect.value as GPUPowerPreference,
    forceRetune: forceRetuneCheckbox.checked,
  };
}

function refreshValidation(): void {
  const result = validateInput(inputEl.value, currentMode());
  renderInputFeedback(inputFeedbackEl, result, inputTouched);
  startButton.disabled = inputTouched ? !result.ok || (result.ok && !result.modeMatches) : false;
}

inputEl.addEventListener("input", () => {
  inputTouched = true;
  refreshValidation();
});
modeFullRadio.addEventListener("change", refreshValidation);
modeSingleDesRadio.addEventListener("change", refreshValidation);

dispatchAdaptiveRadio.addEventListener("change", () => {
  adaptiveTargetInput.disabled = false;
  fixedStepsSelect.disabled = true;
});
dispatchFixedRadio.addEventListener("change", () => {
  adaptiveTargetInput.disabled = true;
  fixedStepsSelect.disabled = false;
});

continueAutomaticallyCheckbox.addEventListener("change", () => {
  activeController?.setContinueAutomatically(continueAutomaticallyCheckbox.checked);
});
resumeButton.addEventListener("click", () => {
  activeController?.resume();
});
startButton.addEventListener("click", () => {
  void startRun();
});

function renderSnapshot(snapshot: RunSnapshot): void {
  progressView.render(snapshot);
  renderResults(resultsPanel, resultsBodyEl, snapshot);
  resumeButton.hidden = snapshot.status !== "paused";
  startButton.disabled = snapshot.status === "running" || snapshot.status === "paused";
}

function toHexBytes(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function buildArchivedResult(
  result: Awaited<ReturnType<typeof runOrchestrator>> | null,
  matched: boolean,
): ArchivedResult {
  // `result.des1Keys`/`des2Keys` are byte7 indices, not plaintext/key bytes
  // themselves (see `ui/key-format.ts`'s module docs) — resolve them to the
  // actual recovered plaintext + expanded DES key before archiving, so a
  // past run's detail view shows the same thing the results panel did.
  return {
    matched,
    des1: recoveredKeysFromIndices(result?.des1Keys ?? []).map((entry) => ({
      plaintextHex: toHexBytes(entry.plaintext),
      keyHex: toHexBytes(entry.key),
    })),
    des2: recoveredKeysFromIndices(result?.des2Keys ?? []).map((entry) => ({
      plaintextHex: toHexBytes(entry.plaintext),
      keyHex: toHexBytes(entry.key),
    })),
    pt3Hex: result?.pt3 ? toHexBytes(result.pt3) : null,
    ntHashesHex: (result?.ntHashes ?? []).map((entry) => toHexBytes(entry.ntHash)),
  };
}

async function refreshArchiveList(): Promise<void> {
  const summaries = await listRuns();
  renderArchiveList(archiveListEl, summaries, selectedArchiveRunId, async (runId) => {
    selectedArchiveRunId = runId;
    const record = await getRun(runId);
    renderArchiveDetail(archiveDetailEl, record);
    await refreshArchiveList();
  });
}

interface PortsBuildResult {
  ports: OrchestratorPorts;
  /** Filled in once `getTuning()` has actually run, for the archive record. */
  tuningRef: { current: TuningSelection | null };
}

/**
 * Build this run's `OrchestratorPorts`, layering the GPU settings panel's
 * choices on top of `createDefaultPorts`'s production wiring (Task 5):
 *   - manual shader/workgroup override replaces `getTuning` with a direct
 *     `autoTuneDevice(...)` call carrying `forcedVariant`/`forcedWorkgroup`
 *     (`createDefaultPorts`'s own `defaultGetTuning` never forwards those —
 *     see `pipeline/orchestrator.ts`'s `DefaultPortsOptions`, which has no
 *     such fields — so a manual override bypasses the cache path entirely,
 *     matching "manual override" as an explicit opt-out of whatever was
 *     auto-tuned/cached before);
 *   - the dispatch-mode panel (Adaptive-with-time-budget vs Fixed-step-
 *     count) replaces `precompute`, since `defaultPrecompute` always uses
 *     `DEFAULT_BUDGET_MS` and never exposes `fixedTargetSteps` (a field this
 *     task added to Task 3's `PrecomputeOptions` for exactly this control).
 */
function buildPorts(
  device: GPUDevice,
  adapter: GPUAdapter,
  lutBuffer: GPUBuffer,
  lutBytes: Uint8Array,
  settings: GpuSettingsState,
): PortsBuildResult {
  const precomputeDeviceLimits = {
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    minStorageBufferOffsetAlignment: device.limits.minStorageBufferOffsetAlignment,
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
  };
  const tuningDeviceLimits = {
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
  };
  const supportedShaders: Array<"compact" | "expanded"> = supportsExpandedShader(device)
    ? ["compact", "expanded"]
    : ["compact"];

  const ports = createDefaultPorts({
    device,
    lutBuffer,
    precomputeDeviceLimits,
    tuningDeviceLimits,
    supportedShaders,
    // The tuning cache key digests the actual shipped WGSL sources and LUT
    // bytes, so a changed shader or LUT invalidates a previously-cached
    // tuning selection (see `webgpu/tuning-cache.ts::computeTuningCacheKey`).
    // `precomputeCompactSource`/`precomputeExpandedSource` and `desLutBytes`
    // are the real assets `webgpu/precompute.ts` bundles/fetches — re-used
    // here rather than re-fetched or duplicated.
    tuningCacheKeyInputs: {
      adapterInfo: adapter.info,
      device,
      precomputeCompactSource,
      precomputeExpandedSource,
      // As of Task 8, verification also runs a real GPU pipeline
      // (`webgpu/false-alarm.ts`, compiled from the same tuned
      // shader/workgroup as precompute) rather than crypto-wasm's serial
      // WASM path, so its shipped WGSL sources belong in the cache key too
      // — a changed verify shader should invalidate a previously-cached
      // tuning selection exactly like a changed precompute shader does.
      falseAlarmCompactSource: verifyCompactSource,
      falseAlarmExpandedSource: verifyExpandedSource,
      desLutBytes: lutBytes,
    },
    lookupConfig: { baseUrl: window.location.origin },
    forceRetune: settings.forceRetune,
  });

  const tuningRef: { current: TuningSelection | null } = { current: null };
  const originalGetTuning = ports.getTuning;
  ports.getTuning = async () => {
    const selection = hasManualTuningOverride(settings)
      ? await autoTuneDevice(
          applyManualTuningOverride({ device, lutBuffer, limits: tuningDeviceLimits, supportedShaders }, settings),
        )
      : await originalGetTuning();
    tuningRef.current = selection;
    return selection;
  };

  ports.precompute = async (target, tuning, onProgress) => {
    const pipeline = createPrecomputePipeline(device, tuning.shader, tuning.workgroupSize);
    return runPrecompute(device, pipeline, lutBuffer, precomputeDeviceLimits, {
      target,
      workgroupSize: tuning.workgroupSize,
      budgetMs: settings.dispatchMode === "adaptive" ? adaptiveBudgetMs(settings) : undefined,
      fixedTargetSteps: settings.dispatchMode === "fixed" ? settings.fixedSteps : undefined,
      onProgress,
    });
  };

  return { ports, tuningRef };
}

/**
 * Negotiate (or reuse) the WebGPU adapter/device and the DES LUT buffer.
 *
 * Only re-requests the adapter/device and re-uploads the LUT when the
 * relevant setting (`powerPreference`) actually changed from what's
 * currently held, or when nothing has been negotiated yet. Otherwise this
 * is a no-op that reuses the existing device/buffer across multiple runs in
 * one browser session. When a replacement genuinely is needed, the
 * previous device/buffer are destroyed first rather than left to leak until
 * GC (each is a real live GPU resource — the LUT buffer alone is ~100KB,
 * but every additional un-destroyed `GPUDevice` also keeps its full
 * pipeline/buffer graph alive).
 */
async function ensureGpu(powerPreference: GPUPowerPreference): Promise<void> {
  if (
    currentDevice &&
    currentAdapter &&
    currentLutBuffer &&
    currentLutBytes &&
    currentPowerPreference === powerPreference
  ) {
    return;
  }

  const { adapter, device } = await requestGpuDevice({ powerPreference });
  const previousLutBuffer = currentLutBuffer;
  const previousDevice = currentDevice;

  currentAdapter = adapter;
  currentDevice = device;
  currentPowerPreference = powerPreference;
  const lut = await loadDesLut(device);
  currentLutBuffer = lut.buffer;
  currentLutBytes = lut.bytes;
  shaderExpandedOption.disabled = !supportsExpandedShader(device);
  await initCrypto();

  // Destroy the previous generation's resources only after the new ones are
  // fully in place, so a mid-negotiation failure above leaves the old,
  // still-working device/buffer intact instead of tearing them down early.
  previousLutBuffer?.destroy();
  if (previousDevice && previousDevice !== device) {
    previousDevice.destroy();
  }
}

function reportGpuSettingsError(message: string): void {
  startButton.disabled = false;
  inputFeedbackEl.textContent = message;
  inputFeedbackEl.setAttribute("data-state", "error");
}

async function startRun(): Promise<void> {
  const mode = currentMode();
  const validation = validateInput(inputEl.value, mode);
  if (!validation.ok || !validation.modeMatches) {
    inputTouched = true;
    refreshValidation();
    return;
  }

  startButton.disabled = true;

  // Negotiate the device BEFORE reading the GPU settings panel's values:
  // whether Expanded/512/1024 are actually selectable depends on this
  // device's negotiated limits (`supportsExpandedShader`, `validWorkgroups`),
  // which don't exist until `ensureGpu` has run. Reading settings first (the
  // previous order) meant a first-run manual override could name a shader/
  // workgroup the device turns out not to support, surfacing as an opaque
  // WebGPU validation error deep in pipeline creation instead of a clear
  // message here.
  try {
    await ensureGpu(powerPreferenceSelect.value as GPUPowerPreference);
  } catch (error) {
    reportGpuSettingsError(error instanceof Error ? error.message : String(error));
    return;
  }
  if (!currentDevice || !currentAdapter || !currentLutBuffer || !currentLutBytes) return;

  const settings = currentGpuSettings();
  if (settings.shader === "expanded" && !supportsExpandedShader(currentDevice)) {
    reportGpuSettingsError("Expanded shader is not supported by this device's negotiated limits.");
    return;
  }
  if (settings.workgroupSize !== "auto") {
    const supportedWorkgroups = validWorkgroups({
      maxComputeInvocationsPerWorkgroup: currentDevice.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: currentDevice.limits.maxComputeWorkgroupSizeX,
    });
    if (!supportedWorkgroups.includes(settings.workgroupSize)) {
      reportGpuSettingsError(
        `Workgroup size ${settings.workgroupSize} is not supported by this device ` +
          `(supported sizes: ${supportedWorkgroups.join(", ")}).`,
      );
      return;
    }
  }

  const { ports, tuningRef } = buildPorts(currentDevice, currentAdapter, currentLutBuffer, currentLutBytes, settings);
  const controller = new RunController(mode);
  activeController = controller;
  progressView.reset();
  const unsubscribe = controller.subscribe(renderSnapshot);

  const runId = generateRunId();
  const startedAt = new Date().toISOString();
  let result: Awaited<ReturnType<typeof runOrchestrator>> | null = null;
  let matched = false;
  try {
    result = await controller.start(inputEl.value, ports, continueAutomaticallyCheckbox.checked);
    matched = controller.current.status === "done";
  } catch {
    // The controller's snapshot already carries the error message for
    // display; the run is still archived below (as a failed attempt).
  } finally {
    unsubscribe();
  }

  const record: RunRecord = {
    schema_version: 1,
    run_id: runId,
    created_at: startedAt,
    command: "web-ui recover",
    input: { raw: inputEl.value, mode },
    compute: {
      kind: "webgpu",
      implementation: "webgpu",
      backend: {
        vendor: currentAdapter.info.vendor,
        architecture: currentAdapter.info.architecture,
        description: currentAdapter.info.description,
      },
    },
    selected_device: {
      vendor: currentAdapter.info.vendor,
      architecture: currentAdapter.info.architecture,
      description: currentAdapter.info.description,
      limits: plainLimits(currentDevice.limits),
    },
    tuning: tuningRef.current,
    // Raw endpoint/candidate bytes are never archived (see archive/db.ts's
    // module docs): a single DES slot's endpoint set alone is ~7MB, so only
    // a short human-readable note is kept per output.
    outputs: { "des1.endpoints": "generated in-memory, not persisted" },
    result: buildArchivedResult(result, matched),
  };
  await putRun(record);
  await refreshArchiveList();
}

refreshValidation();
void refreshArchiveList();
