//! WASM bindings around `ntlmrain`'s pure CPU crypto and wire-format code.
//!
//! This crate does not reimplement any cryptography: every DES/bitslice
//! primitive here is a thin `#[wasm_bindgen]` wrapper that calls straight
//! into `ntlmrain::{cpu, bs_des, bs_sboxes, formats}`. The one piece of new
//! logic is [`verify`], a serial (non-rayon) reimplementation of
//! `ntlmrain::cpu_verify::verify_candidates_with_progress`'s chain-walking
//! algorithm, batched 64-wide to match `bs_des::netntlmv1_64`'s native lane
//! width instead of the native code's x86_64-only 512-wide SIMD batching.

mod crypto;
mod formats;
mod verify;

pub use crypto::*;
pub use formats::*;
pub use verify::*;

use wasm_bindgen::prelude::*;

/// Install a panic hook that forwards Rust panic messages to
/// `console.error` in the browser instead of an opaque "unreachable
/// executed" WebAssembly trap. Safe to call more than once; call it once
/// during module initialization.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
