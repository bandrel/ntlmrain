//! Thin `#[wasm_bindgen]` wrappers around `ntlmrain::cpu`'s pure functions.
//!
//! Every function here calls straight into the native `ntlmrain::cpu`
//! implementation with no logic of its own beyond JS-boundary type
//! conversion (`Uint8Array` <-> `[u8; N]`, `BigInt` <-> `u64`) and length
//! validation (the native signatures use fixed-size arrays; the JS
//! boundary only has slices, so a wrong-length input becomes a `JsError`
//! instead of a panic).
//!
//! Length validation is a plain Rust `Result` ([`LengthError`], no
//! `wasm_bindgen`/`js_sys` involved) so it stays natively testable; only
//! the outermost `#[wasm_bindgen]` functions convert that into a
//! `JsError`. `JsError::new` itself calls a wasm-bindgen import that
//! panics with "cannot call wasm-bindgen imported functions on non-wasm
//! targets" if ever invoked from a native (non-wasm32) test, so the error
//! *path* of the `#[wasm_bindgen]` functions can only be exercised by the
//! `wasm-pack test` suite — native tests exercise [`LengthError`]-returning
//! validation directly instead.

use std::fmt;

use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LengthError {
    what: &'static str,
    expected: usize,
}

impl fmt::Display for LengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} must be exactly {} bytes", self.what, self.expected)
    }
}

fn array7(bytes: &[u8], what: &'static str) -> Result<[u8; 7], LengthError> {
    <[u8; 7]>::try_from(bytes).map_err(|_| LengthError { what, expected: 7 })
}

fn array8(bytes: &[u8], what: &'static str) -> Result<[u8; 8], LengthError> {
    <[u8; 8]>::try_from(bytes).map_err(|_| LengthError { what, expected: 8 })
}

fn array2(bytes: &[u8], what: &'static str) -> Result<[u8; 2], LengthError> {
    <[u8; 2]>::try_from(bytes).map_err(|_| LengthError { what, expected: 2 })
}

/// `ntlmrain::cpu::netntlmv1_hash(&[u8; 7]) -> [u8; 8]`, verbatim.
///
/// `plaintext7` must be exactly 7 bytes; returns the 8-byte DES ciphertext
/// (fixed Crackalack challenge, post-final-permutation, big-endian byte
/// layout matching the native function's `[u8; 8]`).
#[wasm_bindgen]
pub fn netntlmv1_hash(plaintext7: &[u8]) -> Result<Vec<u8>, JsError> {
    let plaintext7 = array7(plaintext7, "plaintext7").map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ntlmrain::cpu::netntlmv1_hash(&plaintext7).to_vec())
}

/// `ntlmrain::cpu::byte7_index_to_plaintext(u64) -> [u8; 7]`, verbatim.
#[wasm_bindgen]
pub fn byte7_index_to_plaintext(index: u64) -> Vec<u8> {
    ntlmrain::cpu::byte7_index_to_plaintext(index).to_vec()
}

/// `ntlmrain::cpu::byte7_hash_to_index(&[u8; 8], u32, u32) -> u64`, verbatim.
///
/// `hash` must be exactly 8 bytes.
#[wasm_bindgen]
pub fn byte7_hash_to_index(
    hash: &[u8],
    reduction_offset: u32,
    position: u32,
) -> Result<u64, JsError> {
    let hash = array8(hash, "hash").map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ntlmrain::cpu::byte7_hash_to_index(
        &hash,
        reduction_offset,
        position,
    ))
}

/// `ntlmrain::cpu::is_exact_des_key_match(u64, &[u8; 8]) -> bool`, verbatim.
///
/// `target` must be exactly 8 bytes.
#[wasm_bindgen]
pub fn is_exact_des_key_match(index: u64, target: &[u8]) -> Result<bool, JsError> {
    let target = array8(target, "target").map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ntlmrain::cpu::is_exact_des_key_match(index, &target))
}

/// `ntlmrain::cpu::expand_des_key(&[u8; 7]) -> [u8; 8]`, verbatim.
///
/// Display-only DES key expansion (parity bits set); NOT used for
/// encryption. `plaintext7` must be exactly 7 bytes.
#[wasm_bindgen]
pub fn expand_des_key(plaintext7: &[u8]) -> Result<Vec<u8>, JsError> {
    let plaintext7 = array7(plaintext7, "plaintext7").map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ntlmrain::cpu::expand_des_key(&plaintext7).to_vec())
}

/// `ntlmrain::cpu::recover_pt3(&[u8; 8]) -> Option<[u8; 2]>`, verbatim
/// 65,536-iteration scalar brute force of the third DES block.
///
/// `target` must be exactly 8 bytes. Returns an empty `Uint8Array` for
/// `None` (no 2-byte plaintext hashes to `target`) or a 2-byte
/// `Uint8Array` for `Some` — `wasm-bindgen` has no direct `Option<Vec<u8>>`
/// mapping, and this keeps the boundary to plain `Uint8Array`/`u32`/`u64`/
/// `bool` types per this crate's design brief. Callers should branch on
/// `.length`.
#[wasm_bindgen]
pub fn recover_pt3(target: &[u8]) -> Result<Vec<u8>, JsError> {
    let target = array8(target, "target").map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ntlmrain::cpu::recover_pt3(&target)
        .map(|pt3| pt3.to_vec())
        .unwrap_or_default())
}

/// `ntlmrain::cpu::assemble_nt_hash(u64, u64, [u8; 2]) -> [u8; 16]`, verbatim.
///
/// `pt3` must be exactly 2 bytes.
#[wasm_bindgen]
pub fn assemble_nt_hash(pt1_index: u64, pt2_index: u64, pt3: &[u8]) -> Result<Vec<u8>, JsError> {
    let pt3 = array2(pt3, "pt3").map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ntlmrain::cpu::assemble_nt_hash(pt1_index, pt2_index, pt3).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Documented golden NT hash 8846F7EAEE8FB117AD06BDD830B7586C, decomposed
    // per this plan's "Wire-level facts": pt1-index-hex 0x008846f7eaee8fb1
    // (ciphertext 727B4E35F947129E), pt2-index-hex 0x0017ad06bdd830b7
    // (ciphertext A52B9CDEDAE86934), pt3 [0x58, 0x6c] (ciphertext
    // BB23EF89F50FC595).
    const PT1_INDEX: u64 = 0x0088_46f7_eaee_8fb1;
    const PT1_CIPHERTEXT: &str = "727b4e35f947129e";
    const PT2_INDEX: u64 = 0x0017_ad06_bdd8_30b7;
    const PT2_CIPHERTEXT: &str = "a52b9cdedae86934";
    const PT3_CIPHERTEXT: &str = "bb23ef89f50fc595";
    const NT_HASH: &str = "8846f7eaee8fb117ad06bdd830b7586c";

    #[test]
    fn netntlmv1_hash_round_trips_pt1_and_pt2() {
        let pt1 = byte7_index_to_plaintext(PT1_INDEX);
        assert_eq!(hex::encode(netntlmv1_hash(&pt1).unwrap()), PT1_CIPHERTEXT);

        let pt2 = byte7_index_to_plaintext(PT2_INDEX);
        assert_eq!(hex::encode(netntlmv1_hash(&pt2).unwrap()), PT2_CIPHERTEXT);
    }

    // Error paths route through `JsError::new`, which panics if invoked
    // outside a real wasm32 host ("cannot call wasm-bindgen imported
    // functions on non-wasm targets") — so length validation is exercised
    // directly against the plain-Rust `LengthError` helpers here instead
    // of through the `#[wasm_bindgen]` functions. The `wasm-pack test`
    // suite exercises the full `JsError` conversion in-browser.
    #[test]
    fn length_helpers_reject_wrong_length_inputs() {
        assert!(array7(&[0u8; 6], "plaintext7").is_err());
        assert!(array7(&[0u8; 8], "plaintext7").is_err());
        assert!(array7(&[0u8; 7], "plaintext7").is_ok());
        assert!(array8(&[0u8; 7], "target").is_err());
        assert!(array8(&[0u8; 8], "target").is_ok());
        assert!(array2(&[0u8; 1], "pt3").is_err());
        assert!(array2(&[0u8; 2], "pt3").is_ok());
    }

    #[test]
    fn recover_pt3_finds_final_two_bytes() {
        let target = hex::decode(PT3_CIPHERTEXT).unwrap();
        assert_eq!(recover_pt3(&target).unwrap(), vec![0x58, 0x6c]);
    }

    #[test]
    fn recover_pt3_returns_empty_when_not_found() {
        // All-zero ciphertext does not correspond to any 2-byte plaintext
        // padded with five zero bytes (verified against the native
        // exhaustive search returning `None`).
        assert!(recover_pt3(&[0u8; 8]).unwrap().is_empty());
    }

    #[test]
    fn assemble_nt_hash_produces_documented_hash() {
        let assembled = assemble_nt_hash(PT1_INDEX, PT2_INDEX, &[0x58, 0x6c]).unwrap();
        assert_eq!(hex::encode(assembled), NT_HASH);
    }

    #[test]
    fn is_exact_des_key_match_confirms_and_rejects() {
        let target = hex::decode(PT1_CIPHERTEXT).unwrap();
        assert!(is_exact_des_key_match(PT1_INDEX, &target).unwrap());
        assert!(!is_exact_des_key_match(PT1_INDEX + 1, &target).unwrap());
    }

    #[test]
    fn byte7_hash_to_index_matches_masked_addition() {
        let hash = hex::decode(PT2_CIPHERTEXT).unwrap();
        let index = byte7_hash_to_index(&hash, 0, 0).unwrap();
        assert_eq!(
            index,
            ntlmrain::cpu::byte7_hash_to_index(
                &<[u8; 8]>::try_from(hash.as_slice()).unwrap(),
                0,
                0
            )
        );
    }

    #[test]
    fn expand_des_key_matches_native_vector() {
        let pt1 = byte7_index_to_plaintext(PT1_INDEX);
        assert_eq!(
            hex::encode(expand_des_key(&pt1).unwrap()),
            "8923bdfdaf753f63"
        );
    }
}
