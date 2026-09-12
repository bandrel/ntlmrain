// Pure types/helpers behind the GPU settings panel. No DOM, no GPU calls —
// just translating the form's state into the shapes `webgpu/tuning.ts` and
// `webgpu/precompute.ts` (Task 3/4) actually accept, so this is unit
// testable without a real adapter.

import type { ShaderVariant } from "../webgpu/precompute";
import type { AutoTuneOptions, TuningKey } from "../webgpu/tuning";

export type ShaderChoice = "auto" | ShaderVariant;
export type DispatchModeChoice = "adaptive" | "fixed";

/** Mirrors `src/cli.rs`'s `--dispatch-mode fixed --fixed-steps <64m|128m|256m|512m|1b>` option list, verbatim. */
export const FIXED_STEP_OPTIONS = [
  { label: "64M", value: 64_000_000 },
  { label: "128M", value: 128_000_000 },
  { label: "256M", value: 256_000_000 },
  { label: "512M", value: 512_000_000 },
  { label: "1B", value: 1_000_000_000 },
] as const;

export const WORKGROUP_SIZE_OPTIONS = [32, 64, 128, 256, 512, 1_024] as const;

/** `webgpu/precompute.ts::DEFAULT_BUDGET_MS` expressed in the seconds unit this panel exposes. */
export const DEFAULT_ADAPTIVE_TARGET_SECONDS = 0.8;

export interface GpuSettingsState {
  shader: ShaderChoice;
  workgroupSize: number | "auto";
  dispatchMode: DispatchModeChoice;
  adaptiveTargetSeconds: number;
  fixedSteps: number;
  powerPreference: GPUPowerPreference;
  /**
   * "Re-tune" override: force a fresh `autoTuneDevice` run (supplying the
   * currently cached selection as the hysteresis `incumbent`) instead of
   * taking the default cache-hit fast path. See
   * `pipeline/orchestrator.ts`'s `DefaultPortsOptions.forceRetune`.
   */
  forceRetune: boolean;
}

export function defaultGpuSettings(): GpuSettingsState {
  return {
    shader: "auto",
    workgroupSize: "auto",
    dispatchMode: "adaptive",
    adaptiveTargetSeconds: DEFAULT_ADAPTIVE_TARGET_SECONDS,
    fixedSteps: FIXED_STEP_OPTIONS[0].value,
    powerPreference: "high-performance",
    forceRetune: false,
  };
}

/**
 * Whether the Expanded shader option should be selectable at all, per Task
 * 3's `supportsExpandedShader` limit check. Kept as a plain boolean-in,
 * boolean-out helper (rather than importing `supportsExpandedShader`
 * itself, which needs a live `GPUDevice`) so this stays testable without a
 * device.
 */
export function isExpandedShaderAllowed(deviceSupportsExpanded: boolean): boolean {
  return deviceSupportsExpanded;
}

/**
 * Fields of `AutoTuneOptions` this panel's manual overrides feed into,
 * layered on top of whatever the caller already built for automatic tuning.
 * `forcedVariant`/`forcedWorkgroup` are left `undefined` (i.e. "auto") when
 * the corresponding form field is `"auto"`.
 */
export function applyManualTuningOverride(
  base: Pick<AutoTuneOptions, "device" | "lutBuffer" | "limits" | "supportedShaders" | "incumbent">,
  settings: GpuSettingsState,
): AutoTuneOptions {
  return {
    ...base,
    forcedVariant: settings.shader === "auto" ? undefined : settings.shader,
    forcedWorkgroup: settings.workgroupSize === "auto" ? undefined : settings.workgroupSize,
  };
}

/** Whether any manual override is set, i.e. auto-tuning alone is not enough and `applyManualTuningOverride` must be used. */
export function hasManualTuningOverride(settings: GpuSettingsState): boolean {
  return settings.shader !== "auto" || settings.workgroupSize !== "auto";
}

/** `adaptiveTargetSeconds` converted to the millisecond unit `PrecomputeOptions.budgetMs` expects. */
export function adaptiveBudgetMs(settings: GpuSettingsState): number {
  return Math.max(Math.round(settings.adaptiveTargetSeconds * 1000), 1);
}

export function keysEqual(a: TuningKey, b: TuningKey): boolean {
  return a.shader === b.shader && a.workgroupSize === b.workgroupSize;
}
