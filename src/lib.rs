pub mod artifacts;
pub mod bitslice;
#[cfg(not(target_arch = "x86_64"))]
mod bs_des;
#[cfg(not(target_arch = "x86_64"))]
mod bs_sboxes;
// These four modules are native-only: `cli` drives the whole CLI (GPU +
// filesystem + network); `compute` schedules native GPU/CPU precompute
// dispatch; `gpu` calls `wgpu::Instance::enumerate_adapters`, which only
// exists on wgpu's native backends, not its wasm32 WebGPU backend; and
// `local_lookup`/`remote_lookup` use Unix/Windows-only positioned-file-read
// syscalls and `reqwest`'s `blocking` client, neither of which build for
// wasm32. `crypto-wasm` (added for the browser web UI) only needs the pure
// `cpu`/`bitslice`/`formats`/`params` modules below (it reaches the
// bitsliced DES engine through the already-arch-agnostic `bitslice`
// wrappers, not `bs_des`/`bs_sboxes` directly, so those two stay private),
// so gate the rest out for wasm32 rather than trying to port them.
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
