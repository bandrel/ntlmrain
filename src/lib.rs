pub mod artifacts;
pub mod bitslice;
// `pub` (not `pub(crate)`) so `crypto-wasm` can call the portable bitsliced
// DES engine directly on its wasm32 target (which never matches the
// x86_64-only `fast-des` SIMD path gated out below).
#[cfg(not(target_arch = "x86_64"))]
pub mod bs_des;
#[cfg(not(target_arch = "x86_64"))]
pub mod bs_sboxes;
// These four modules are native-only: `cli` drives the whole CLI (GPU +
// filesystem + network); `compute` schedules native GPU/CPU precompute
// dispatch; `gpu` calls `wgpu::Instance::enumerate_adapters`, which only
// exists on wgpu's native backends, not its wasm32 WebGPU backend; and
// `local_lookup`/`remote_lookup` use Unix/Windows-only positioned-file-read
// syscalls and `reqwest`'s `blocking` client, neither of which build for
// wasm32. `crypto-wasm` (added for the browser web UI) only needs the pure
// `cpu`/`bs_des`/`bs_sboxes`/`formats`/`params` modules below, so gate the
// rest out for wasm32 rather than trying to port them — same idiom as the
// `bs_des`/`bs_sboxes` arch-gating above, just gating on the new target
// instead of the old one.
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
#[cfg(not(target_arch = "wasm32"))]
pub mod compute;
pub mod config;
pub mod cpu;
mod cpu_schedule;
pub mod cpu_verify;
pub mod formats;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
pub mod input;
#[cfg(not(target_arch = "wasm32"))]
pub mod local_lookup;
pub mod params;
mod platform;
#[cfg(not(target_arch = "wasm32"))]
pub mod remote_lookup;

pub const CHAIN_LEN: u32 = 881_689;
pub const TABLE_INDEX: u32 = 0;
pub const FIXED_CHALLENGE_HEX: &str = "1122334455667788";

#[cfg(not(target_arch = "wasm32"))]
pub use cli::{error_exit_code, run};
