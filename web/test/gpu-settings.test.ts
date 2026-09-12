import { describe, expect, it } from "vitest";
import {
  DEFAULT_ADAPTIVE_TARGET_SECONDS,
  FIXED_STEP_OPTIONS,
  WORKGROUP_SIZE_OPTIONS,
  adaptiveBudgetMs,
  applyManualTuningOverride,
  defaultGpuSettings,
  hasManualTuningOverride,
  isExpandedShaderAllowed,
} from "../src/ui/gpu-settings";

describe("gpu-settings.ts", () => {
  it("defaults to auto shader/workgroup, adaptive dispatch at 0.8s, high-performance power preference", () => {
    const settings = defaultGpuSettings();
    expect(settings.shader).toBe("auto");
    expect(settings.workgroupSize).toBe("auto");
    expect(settings.dispatchMode).toBe("adaptive");
    expect(settings.adaptiveTargetSeconds).toBe(DEFAULT_ADAPTIVE_TARGET_SECONDS);
    expect(settings.powerPreference).toBe("high-performance");
    expect(hasManualTuningOverride(settings)).toBe(false);
  });

  it("mirrors the native fixed-step option list (64M/128M/256M/512M/1B)", () => {
    expect(FIXED_STEP_OPTIONS.map((option) => option.value)).toEqual([
      64_000_000, 128_000_000, 256_000_000, 512_000_000, 1_000_000_000,
    ]);
  });

  it("exposes the standard WebGPU workgroup size ladder", () => {
    expect(WORKGROUP_SIZE_OPTIONS).toEqual([32, 64, 128, 256, 512, 1024]);
  });

  it("disallows Expanded only when the device limit check says so", () => {
    expect(isExpandedShaderAllowed(true)).toBe(true);
    expect(isExpandedShaderAllowed(false)).toBe(false);
  });

  it("hasManualTuningOverride is true when either shader or workgroup is overridden", () => {
    const settings = defaultGpuSettings();
    expect(hasManualTuningOverride({ ...settings, shader: "compact" })).toBe(true);
    expect(hasManualTuningOverride({ ...settings, workgroupSize: 128 })).toBe(true);
    expect(hasManualTuningOverride(settings)).toBe(false);
  });

  it("applyManualTuningOverride leaves forcedVariant/forcedWorkgroup undefined on auto", () => {
    const base = {
      device: {} as GPUDevice,
      lutBuffer: {} as GPUBuffer,
      limits: { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 },
      supportedShaders: ["compact" as const],
    };
    const options = applyManualTuningOverride(base, defaultGpuSettings());
    expect(options.forcedVariant).toBeUndefined();
    expect(options.forcedWorkgroup).toBeUndefined();
  });

  it("applyManualTuningOverride forwards explicit shader/workgroup overrides", () => {
    const base = {
      device: {} as GPUDevice,
      lutBuffer: {} as GPUBuffer,
      limits: { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 },
      supportedShaders: ["compact" as const, "expanded" as const],
    };
    const options = applyManualTuningOverride(base, {
      ...defaultGpuSettings(),
      shader: "expanded",
      workgroupSize: 128,
    });
    expect(options.forcedVariant).toBe("expanded");
    expect(options.forcedWorkgroup).toBe(128);
  });

  it("adaptiveBudgetMs converts seconds to a rounded millisecond budget, never below 1", () => {
    expect(adaptiveBudgetMs({ ...defaultGpuSettings(), adaptiveTargetSeconds: 0.8 })).toBe(800);
    expect(adaptiveBudgetMs({ ...defaultGpuSettings(), adaptiveTargetSeconds: 1.8 })).toBe(1800);
    expect(adaptiveBudgetMs({ ...defaultGpuSettings(), adaptiveTargetSeconds: 0 })).toBe(1);
  });
});
