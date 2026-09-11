// WebGPU adapter/device acquisition, mirroring the native CLI's device
// selection in `src/gpu.rs::request_device`: raise the storage/buffer/
// workgroup-storage limits to whatever the adapter actually reports rather
// than settling for the (much smaller) WebGPU "default" limits, but never
// ask for more than the adapter offers (the browser throws if you do).

/// Workgroup storage the Expanded-LUT shader class needs
/// (`ShaderVariant::Expanded::workgroup_storage_bytes()` in `src/gpu.rs`).
/// Kept in sync manually with that Rust constant; there is no shared
/// source of truth across the Rust/TS boundary for this single number.
export const EXPANDED_WORKGROUP_STORAGE_BYTES = 31_748;

export interface GpuDeviceHandle {
  adapter: GPUAdapter;
  device: GPUDevice;
}

export interface RequestGpuDeviceOptions {
  /** Forwarded to `navigator.gpu.requestAdapter`. */
  powerPreference?: GPUPowerPreference;
}

/**
 * Request a WebGPU adapter and a device whose storage-related limits are
 * raised to the adapter's own reported ceiling (`maxStorageBufferBindingSize`,
 * `maxBufferSize`, `maxComputeWorkgroupStorageSize`). Because the values
 * requested are read directly off `adapter.limits`, they can never exceed
 * what the adapter supports — satisfying "clamp to whatever the adapter
 * actually supports" without needing a separate min() step.
 */
export async function requestGpuDevice(
  options: RequestGpuDeviceOptions = {},
): Promise<GpuDeviceHandle> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    throw new Error("WebGPU is not available in this browser (navigator.gpu is undefined)");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference,
  });
  if (!adapter) {
    throw new Error("navigator.gpu.requestAdapter() returned no adapter");
  }

  const limits = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
    },
  });

  return { adapter, device };
}

/**
 * Whether the Expanded-shader-class workgroup storage requirement is
 * actually available given the negotiated device's limits. Later tuning
 * work (Task 4) uses this to decide whether Expanded is even a candidate
 * before spending time benchmarking it.
 */
export function supportsExpandedShader(device: GPUDevice): boolean {
  return device.limits.maxComputeWorkgroupStorageSize >= EXPANDED_WORKGROUP_STORAGE_BYTES;
}
