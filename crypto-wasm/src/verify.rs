//! Serial port of `ntlmrain::cpu_verify::verify_candidates_with_progress`'s
//! candidate-chain-walk algorithm.
//!
//! This does not reimplement any DES: every hash is computed by
//! `ntlmrain::bitslice::netntlmv1_bitslice_batch` — the same already-tested
//! bitslice entry point the native code uses, which transparently picks
//! `fast-des`'s x86_64 SIMD path on x86_64 or the portable `bs_des`
//! bitslice engine everywhere else (including wasm32). Calling through
//! this arch-agnostic wrapper (rather than `bs_des::netntlmv1_64`
//! directly) is what makes `crypto-wasm` itself buildable on x86_64 at
//! all: `bs_des`/`bs_sboxes` are private to the root crate and only even
//! *exist* there when `not(target_arch = "x86_64")`. What's new here is
//! purely the *scheduling*: the native code runs this same algorithm
//! across rayon worker threads, each batching up to 512 chains at a time
//! inside a large-stack scoped OS thread (`with_bitslice_stack`). None of
//! that is available or needed in a wasm32 module (no threads): this is a
//! single-loop, single-thread port that wave-batches 64 chains at a
//! time — matching the portable `bs_des` bitslice engine's native lane
//! width (the same width the x86_64 SIMD path also groups by
//! internally), without needing any arch-specific branching here.
//!
//! Chain-walk semantics, ported verbatim from `cpu_verify.rs`:
//! - A candidate `(start, position)` is checked at chain positions
//!   `0..=position`: position 0 means `start` itself is the candidate
//!   answer (no reduction), matching the native `ChainState { index: start,
//!   position: 0, target_position: position }` seed.
//! - Each step hashes the chain's current index, checks it against
//!   `target` two independent ways (bitsliced-hash equality *and* the
//!   scalar `cpu::is_exact_des_key_match` guard), and if that fails,
//!   reduces with the *pre-increment* position
//!   (`hash + reduction_offset + position`, then `position += 1`) before
//!   comparing `position > target_position` to decide whether the chain is
//!   exhausted.

use ntlmrain::{
    bitslice::{index_to_fast_des_key, netntlmv1_bitslice_batch},
    cpu::is_exact_des_key_match,
    params::BYTE7_MASK,
};
use wasm_bindgen::prelude::*;

/// Keys hashed per wave, matching the portable bitslice engine's native
/// lane width (`ntlmrain::bitslice`'s non-x86_64 `imp`).
const LANES: usize = 64;

/// Progress snapshot, field-for-field matching
/// `ntlmrain::cpu_verify::CpuVerifyProgress`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifyProgress {
    pub candidates_done: u64,
    pub candidates_total: u64,
    pub steps_done: u64,
    pub steps_total: u64,
    pub verified_keys: u64,
}

/// Final result, field-for-field matching
/// `ntlmrain::cpu_verify::CpuVerifyResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyOutcome {
    pub keys: Vec<u64>,
    pub candidates_completed: u64,
    pub candidates_total: u64,
    pub steps_completed: u64,
    pub steps_total: u64,
    pub stopped_early: bool,
}

#[derive(Clone, Copy, Debug)]
struct ChainState {
    index: u64,
    position: u32,
    target_position: u32,
}

/// Serial port of `verify_candidates_with_progress`. `candidates` is a
/// list of `(start, position)` pairs (the same fields as native
/// `CpuCandidate { start, position }`). `stop_at_first` is the native
/// `all` flag inverted (`all = !stop_at_first`): when `true`, verification
/// stops at the first exact hit instead of walking every candidate to
/// completion. `progress` is called after every processed wave (the
/// wasm-bindgen wrapper in this module throttles how often that reaches
/// JS); it is *always* called once up front (even for an empty candidate
/// list, matching the native function) and once more with the final
/// totals when verification finishes.
pub fn verify_candidates_serial<F>(
    candidates: &[(u64, u32)],
    target: [u8; 8],
    table_index: u32,
    stop_at_first: bool,
    mut progress: F,
) -> VerifyOutcome
where
    F: FnMut(VerifyProgress),
{
    let candidates_total = candidates.len() as u64;
    let steps_total = candidates.iter().fold(0u64, |total, &(_, position)| {
        total.saturating_add(u64::from(position) + 1)
    });

    if candidates.is_empty() {
        progress(VerifyProgress {
            candidates_total,
            steps_total,
            ..VerifyProgress::default()
        });
        return VerifyOutcome {
            keys: Vec::new(),
            candidates_completed: 0,
            candidates_total,
            steps_completed: 0,
            steps_total,
            stopped_early: false,
        };
    }

    // Matches `cpu_verify.rs`'s `reduction_offset = u64::from(table_index) << 16`
    // exactly (a u64 shift, not `params::reduction_offset_from_table_index`'s
    // u32-wrapping multiply), so this port's overflow behavior is identical
    // to the native code for any `table_index`, not just realistic ones.
    let reduction_offset: u64 = u64::from(table_index) << 16;
    let target_hash = u64::from_le_bytes(target);

    let mut active: Vec<ChainState> = candidates
        .iter()
        .map(|&(start, position)| ChainState {
            index: start,
            position: 0,
            target_position: position,
        })
        .collect();

    let mut keys_found: Vec<u64> = Vec::new();
    let mut candidates_done: u64 = 0;
    let mut steps_done: u64 = 0;
    let mut verified_keys: u64 = 0;
    let mut stopped = false;

    let mut keys_batch = [0u64; LANES];
    let mut hash_values = [0u64; LANES];

    'outer: while !active.is_empty() {
        let active_len = active.len();
        let mut read = 0usize;
        let mut write = 0usize;

        while read < active_len {
            let n = (active_len - read).min(LANES);
            for (slot, state) in keys_batch[..n]
                .iter_mut()
                .zip(active[read..read + n].iter())
            {
                *slot = index_to_fast_des_key(state.index);
            }

            // `netntlmv1_bitslice_batch` already returns each ciphertext
            // converted to `byte7_hash_value`'s little-endian hash-word
            // representation (`ciphertext_to_hash_le`, applied internally
            // by both its x86_64 and portable backends), so `hash_values`
            // here is directly comparable to `target_hash` / usable in the
            // reduction formula with no further conversion.
            netntlmv1_bitslice_batch(&keys_batch[..n], &mut hash_values[..n]);
            steps_done += n as u64;

            for (i, &hash_value) in hash_values.iter().enumerate().take(n) {
                let slot = read + i;
                let state = active[slot];

                let exact_hit =
                    hash_value == target_hash && is_exact_des_key_match(state.index, &target);

                if exact_hit {
                    keys_found.push(state.index);
                    candidates_done += 1;
                    verified_keys += 1;
                    if stop_at_first {
                        stopped = true;
                        break;
                    }
                    continue;
                }

                let next_position = state.position + 1;
                if next_position > state.target_position {
                    candidates_done += 1;
                } else {
                    let next_index = hash_value
                        .wrapping_add(reduction_offset)
                        .wrapping_add(u64::from(state.position))
                        & BYTE7_MASK;
                    active[write] = ChainState {
                        index: next_index,
                        position: next_position,
                        target_position: state.target_position,
                    };
                    write += 1;
                }
            }

            read += n;
            progress(VerifyProgress {
                candidates_done: candidates_done.min(candidates_total),
                candidates_total,
                steps_done: steps_done.min(steps_total),
                steps_total,
                verified_keys,
            });
            if stopped {
                break;
            }
        }

        active.truncate(write);
        if stopped {
            break 'outer;
        }
    }

    keys_found.sort_unstable();
    keys_found.dedup();
    if stop_at_first && keys_found.len() > 1 {
        keys_found.truncate(1);
    }

    let candidates_completed = candidates_done.min(candidates_total);
    let steps_completed = steps_done.min(steps_total);
    VerifyOutcome {
        keys: keys_found,
        candidates_completed,
        candidates_total,
        steps_completed,
        steps_total,
        stopped_early: stopped && candidates_completed < candidates_total,
    }
}

/// Result handed back to JS, field-for-field matching [`VerifyOutcome`].
#[wasm_bindgen]
pub struct VerifyResultJs {
    keys: Vec<u64>,
    candidates_completed: u64,
    candidates_total: u64,
    steps_completed: u64,
    steps_total: u64,
    stopped_early: bool,
}

#[wasm_bindgen]
impl VerifyResultJs {
    #[wasm_bindgen(getter)]
    pub fn keys(&self) -> Vec<u64> {
        self.keys.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn candidates_completed(&self) -> u64 {
        self.candidates_completed
    }

    #[wasm_bindgen(getter)]
    pub fn candidates_total(&self) -> u64 {
        self.candidates_total
    }

    #[wasm_bindgen(getter)]
    pub fn steps_completed(&self) -> u64 {
        self.steps_completed
    }

    #[wasm_bindgen(getter)]
    pub fn steps_total(&self) -> u64 {
        self.steps_total
    }

    #[wasm_bindgen(getter)]
    pub fn stopped_early(&self) -> bool {
        self.stopped_early
    }
}

impl From<VerifyOutcome> for VerifyResultJs {
    fn from(outcome: VerifyOutcome) -> Self {
        Self {
            keys: outcome.keys,
            candidates_completed: outcome.candidates_completed,
            candidates_total: outcome.candidates_total,
            steps_completed: outcome.steps_completed,
            steps_total: outcome.steps_total,
            stopped_early: outcome.stopped_early,
        }
    }
}

/// Progress snapshot handed to the JS callback, field-for-field matching
/// [`VerifyProgress`] / native `CpuVerifyProgress`.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct VerifyProgressJs {
    candidates_done: u64,
    candidates_total: u64,
    steps_done: u64,
    steps_total: u64,
    verified_keys: u64,
}

#[wasm_bindgen]
impl VerifyProgressJs {
    #[wasm_bindgen(getter)]
    pub fn candidates_done(&self) -> u64 {
        self.candidates_done
    }

    #[wasm_bindgen(getter)]
    pub fn candidates_total(&self) -> u64 {
        self.candidates_total
    }

    #[wasm_bindgen(getter)]
    pub fn steps_done(&self) -> u64 {
        self.steps_done
    }

    #[wasm_bindgen(getter)]
    pub fn steps_total(&self) -> u64 {
        self.steps_total
    }

    #[wasm_bindgen(getter)]
    pub fn verified_keys(&self) -> u64 {
        self.verified_keys
    }
}

impl From<VerifyProgress> for VerifyProgressJs {
    fn from(progress: VerifyProgress) -> Self {
        Self {
            candidates_done: progress.candidates_done,
            candidates_total: progress.candidates_total,
            steps_done: progress.steps_done,
            steps_total: progress.steps_total,
            verified_keys: progress.verified_keys,
        }
    }
}

const PROGRESS_INTERVAL_MS: f64 = 250.0;

/// `#[wasm_bindgen]` boundary for [`verify_candidates_serial`].
///
/// `starts` and `positions` are parallel arrays (one candidate per index);
/// `target` must be exactly 8 bytes. `progress_callback` is called at most
/// every ~250ms (via `js_sys::Date::now()`) while verification runs, plus
/// exactly once more at the end with the final totals, so a fast run still
/// reports completion even if it never crosses the 250ms threshold.
#[wasm_bindgen]
pub fn verify_candidates(
    starts: &[u64],
    positions: &[u32],
    target: &[u8],
    table_index: u32,
    stop_at_first: bool,
    progress_callback: &js_sys::Function,
) -> Result<VerifyResultJs, JsError> {
    if starts.len() != positions.len() {
        return Err(JsError::new(
            "starts and positions must be parallel arrays of equal length",
        ));
    }
    let target: [u8; 8] = target
        .try_into()
        .map_err(|_| JsError::new("target must be exactly 8 bytes"))?;

    let candidates: Vec<(u64, u32)> = starts
        .iter()
        .copied()
        .zip(positions.iter().copied())
        .collect();

    let mut last_forward = js_sys::Date::now();
    let outcome = verify_candidates_serial(
        &candidates,
        target,
        table_index,
        stop_at_first,
        |progress| {
            let now = js_sys::Date::now();
            if now - last_forward >= PROGRESS_INTERVAL_MS {
                last_forward = now;
                let progress_js = VerifyProgressJs::from(progress);
                let _ = progress_callback.call1(&JsValue::UNDEFINED, &JsValue::from(progress_js));
            }
        },
    );

    let final_progress = VerifyProgressJs::from(VerifyProgress {
        candidates_done: outcome.candidates_completed,
        candidates_total: outcome.candidates_total,
        steps_done: outcome.steps_completed,
        steps_total: outcome.steps_total,
        verified_keys: outcome.keys.len() as u64,
    });
    let _ = progress_callback.call1(&JsValue::UNDEFINED, &JsValue::from(final_progress));

    Ok(VerifyResultJs::from(outcome))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntlmrain::cpu::{byte7_hash_to_index, byte7_index_to_plaintext, netntlmv1_hash};
    use ntlmrain::cpu_verify::{CpuCandidate, verify_candidates_with_progress};

    const PT1_INDEX: u64 = 0x0088_46f7_eaee_8fb1;
    const PT1_CIPHERTEXT: &str = "727b4e35f947129e";

    fn hex8(hex_str: &str) -> [u8; 8] {
        hex::decode(hex_str).unwrap().try_into().unwrap()
    }

    /// Every test below that calls `verify_candidates_serial` with a
    /// non-empty candidate list needs a bigger-than-default stack: the
    /// unrolled bitsliced-DES Feistel rounds it calls through
    /// (`ntlmrain::bitslice::netntlmv1_bitslice_batch`, portable `bs_des`
    /// backend on this non-x86_64 dev machine) have very large per-call
    /// stack frames in unoptimized (debug) builds, which overflow the
    /// default 2MiB test-thread stack (confirmed empirically: every test
    /// here that invokes `verify_candidates_serial` aborted with a stack
    /// overflow before this wrapper was added; `empty_candidate_list_*`,
    /// which returns before ever calling the batch function, did not).
    /// This mirrors `ntlmrain::bitslice::with_bitslice_stack`'s own
    /// scoped-thread pattern (`src/bitslice.rs:20-33`) rather than
    /// widening every cargo-launched process's stack workspace-wide.
    fn with_big_stack<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(f)
            .expect("spawn big-stack test thread")
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    }

    /// Walk `position` reduction steps from `start` and return the final
    /// chain index alongside its ciphertext, using the scalar reference
    /// implementation (independent of both the native parallel verifier
    /// and this port).
    fn walk(start: u64, position: u32) -> (u64, [u8; 8]) {
        let mut index = start;
        for current in 0..=position {
            let hash = netntlmv1_hash(&byte7_index_to_plaintext(index));
            if current == position {
                return (index, hash);
            }
            index = byte7_hash_to_index(&hash, 0, current);
        }
        unreachable!()
    }

    #[test]
    fn golden_vector_hits_at_position_zero() {
        // Documented golden NT hash decomposition's pt1: index
        // 0x008846f7eaee8fb1 hashes directly to ciphertext1
        // 727B4E35F947129E, so a candidate at position 0 (chain start IS
        // the answer, no reduction) must hit immediately at that exact
        // index.
        let target = hex8(PT1_CIPHERTEXT);
        with_big_stack(move || {
            let outcome = verify_candidates_serial(&[(PT1_INDEX, 0)], target, 0, true, |_| {});

            assert_eq!(outcome.keys, vec![PT1_INDEX]);
            assert_eq!(outcome.candidates_completed, 1);
            assert_eq!(outcome.candidates_total, 1);
            assert_eq!(outcome.steps_completed, 1);
            assert_eq!(outcome.steps_total, 1);
            assert!(!outcome.stopped_early);
        });
    }

    #[test]
    fn misses_when_position_is_reached_without_a_hit() {
        let (_, target) = walk(0x4321, 4);
        let mut miss_target = target;
        miss_target[7] ^= 1;
        with_big_stack(move || {
            let outcome = verify_candidates_serial(&[(0x4321, 4)], miss_target, 0, true, |_| {});
            assert!(outcome.keys.is_empty());
            assert_eq!(outcome.candidates_completed, 1);
            assert_eq!(outcome.steps_completed, 5);
            assert!(!outcome.stopped_early);
        });
    }

    #[test]
    fn finds_hit_at_nonzero_chain_position() {
        let (key, target) = walk(0x1234, 19);
        with_big_stack(move || {
            let outcome =
                verify_candidates_serial(&[(0x1234, 18), (0x1234, 19)], target, 0, false, |_| {});
            assert_eq!(outcome.keys, vec![key]);
            assert_eq!(outcome.candidates_completed, 2);
            assert_eq!(outcome.steps_completed, 39);
        });
    }

    #[test]
    fn stop_at_first_halts_before_exhausting_every_candidate() {
        let start = PT1_INDEX;
        let target = netntlmv1_hash(&byte7_index_to_plaintext(start));
        let candidates = [(start, 0), (start, 0), (1u64, 100_000)];

        with_big_stack(move || {
            let all = verify_candidates_serial(&candidates[..2], target, 0, false, |_| {});
            assert_eq!(all.keys, vec![start]);

            let first = verify_candidates_serial(&candidates, target, 0, true, |_| {});
            assert_eq!(first.keys, vec![start]);
            assert!(first.steps_completed < first.steps_total);
        });
    }

    #[test]
    fn empty_candidate_list_reports_zero_totals_and_calls_progress_once() {
        let mut calls = 0u32;
        let outcome = verify_candidates_serial(&[], [0u8; 8], 0, false, |_| calls += 1);
        assert_eq!(calls, 1);
        assert!(outcome.keys.is_empty());
        assert_eq!(outcome.candidates_total, 0);
        assert_eq!(outcome.steps_total, 0);
        assert!(!outcome.stopped_early);
    }

    /// Cross-check this crate's serial port against the native parallel
    /// implementation it was ported from, across a batch large enough to
    /// span more than one 64-wide wave, with a real hit buried inside.
    #[test]
    fn matches_native_parallel_verifier_across_a_multi_wave_batch() {
        let known_start = PT1_INDEX;
        let target = netntlmv1_hash(&byte7_index_to_plaintext(known_start));

        let mut serial_candidates: Vec<(u64, u32)> = (0u64..200)
            .map(|index| (0x10_0000 + index, (index % 8) as u32))
            .collect();
        serial_candidates[123] = (known_start, 0);

        let native_candidates: Vec<CpuCandidate> = serial_candidates
            .iter()
            .map(|&(start, position)| CpuCandidate { start, position })
            .collect();

        // The native call already runs on its own big-stack scoped thread
        // internally (`with_bitslice_stack`); only the serial port's call
        // needs `with_big_stack` here.
        let native = verify_candidates_with_progress(&native_candidates, target, 0, true, |_| {});
        let serial = with_big_stack(move || {
            verify_candidates_serial(&serial_candidates, target, 0, false, |_| {})
        });

        assert_eq!(serial.keys, native.keys);
        assert_eq!(serial.candidates_completed, native.candidates_completed);
        assert_eq!(serial.candidates_total, native.candidates_total);
        assert_eq!(serial.steps_completed, native.steps_completed);
        assert_eq!(serial.steps_total, native.steps_total);
    }
}
