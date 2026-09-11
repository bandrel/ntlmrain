// Regression test for the final whole-branch review's finding: workgroup
// sizes 512/1024 were unreachable because `requestGpuDevice` never asked for
// `maxComputeInvocationsPerWorkgroup`/`maxComputeWorkgroupSizeX` in
// `requiredLimits`, leaving both at WebGPU's default of 256.

import { afterEach, describe, expect, it } from "vitest";
import { requestGpuDevice, supportsExpandedShader, EXPANDED_WORKGROUP_STORAGE_BYTES } from "../src/webgpu/device";

interface FakeAdapter {
  limits: Record<string, number>;
  requestDevice: (descriptor: { requiredLimits: Record<string, number> }) => Promise<unknown>;
}

function installFakeGpu(adapterLimits: Record<string, number>): {
  requestDeviceCalls: Array<{ requiredLimits: Record<string, number> }>;
} {
  const requestDeviceCalls: Array<{ requiredLimits: Record<string, number> }> = [];
  const adapter: FakeAdapter = {
    limits: adapterLimits,
    requestDevice: async (descriptor) => {
      requestDeviceCalls.push(descriptor);
      return { limits: descriptor.requiredLimits };
    },
  };
  // Node's global `navigator` (present since Node 21) is a getter-only
  // accessor property, so a plain assignment throws; redefine it instead.
  Object.defineProperty(globalThis, "navigator", {
    value: { gpu: { requestAdapter: async () => adapter } },
    configurable: true,
    writable: true,
  });
  return { requestDeviceCalls };
}

const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");

describe("requestGpuDevice", () => {
  afterEach(() => {
    if (originalNavigatorDescriptor) {
      Object.defineProperty(globalThis, "navigator", originalNavigatorDescriptor);
    }
  });

  it("requests maxComputeInvocationsPerWorkgroup and maxComputeWorkgroupSizeX, not just the three storage-related limits", async () => {
    const adapterLimits = {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
      maxComputeWorkgroupStorageSize: 32_768,
      maxComputeInvocationsPerWorkgroup: 1_024,
      maxComputeWorkgroupSizeX: 1_024,
    };
    const { requestDeviceCalls } = installFakeGpu(adapterLimits);

    await requestGpuDevice({ powerPreference: "high-performance" });

    expect(requestDeviceCalls).toHaveLength(1);
    const requested = requestDeviceCalls[0].requiredLimits;
    expect(requested.maxComputeInvocationsPerWorkgroup).toBe(1_024);
    expect(requested.maxComputeWorkgroupSizeX).toBe(1_024);
    // Never ask for more than the adapter offers.
    expect(requested.maxComputeInvocationsPerWorkgroup).toBeLessThanOrEqual(adapterLimits.maxComputeInvocationsPerWorkgroup);
    expect(requested.maxComputeWorkgroupSizeX).toBeLessThanOrEqual(adapterLimits.maxComputeWorkgroupSizeX);
  });

  it("never requests more than the adapter's own reported ceiling when the adapter is limited to 256", async () => {
    const adapterLimits = {
      maxStorageBufferBindingSize: 128 << 20,
      maxBufferSize: 128 << 20,
      maxComputeWorkgroupStorageSize: 16_384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    };
    const { requestDeviceCalls } = installFakeGpu(adapterLimits);

    await requestGpuDevice({});

    const requested = requestDeviceCalls[0].requiredLimits;
    expect(requested.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(requested.maxComputeWorkgroupSizeX).toBe(256);
  });
});

describe("supportsExpandedShader", () => {
  it("is true only when the negotiated device's workgroup storage limit covers the Expanded shader's requirement", () => {
    const bigEnough = { limits: { maxComputeWorkgroupStorageSize: EXPANDED_WORKGROUP_STORAGE_BYTES } } as unknown as GPUDevice;
    const tooSmall = { limits: { maxComputeWorkgroupStorageSize: EXPANDED_WORKGROUP_STORAGE_BYTES - 1 } } as unknown as GPUDevice;
    expect(supportsExpandedShader(bigEnough)).toBe(true);
    expect(supportsExpandedShader(tooSmall)).toBe(false);
  });
});
