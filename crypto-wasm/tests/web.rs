//! Browser-run mirror of the native golden-vector test in
//! `src/verify.rs`/`src/crypto.rs`, exercising the actual
//! `#[wasm_bindgen]`-exposed functions (including the `JsError` error path
//! and the `js_sys::Function` progress callback) that native `cargo test`
//! cannot touch. Run with `wasm-pack test --headless --chrome`.

use wasm_bindgen::{JsCast, JsValue, prelude::*};
use wasm_bindgen_test::*;

use crypto_wasm::{
    assemble_nt_hash, byte7_index_to_plaintext, decode_candidate_file, encode_endpoint_file,
    netntlmv1_hash, recover_pt3, verify_candidates,
};

wasm_bindgen_test_configure!(run_in_browser);

// Documented golden NT hash 8846F7EAEE8FB117AD06BDD830B7586C's decomposition
// (same vector as the native tests): pt1-index-hex 0x008846f7eaee8fb1
// (ciphertext 727B4E35F947129E), pt2-index-hex 0x0017ad06bdd830b7
// (ciphertext A52B9CDEDAE86934), pt3 [0x58, 0x6c].
const PT1_INDEX: u64 = 0x0088_46f7_eaee_8fb1;
const PT1_CIPHERTEXT: [u8; 8] = [0x72, 0x7b, 0x4e, 0x35, 0xf9, 0x47, 0x12, 0x9e];
const PT2_INDEX: u64 = 0x0017_ad06_bdd8_30b7;
const PT2_CIPHERTEXT: [u8; 8] = [0xa5, 0x2b, 0x9c, 0xde, 0xda, 0xe8, 0x69, 0x34];
const PT3_CIPHERTEXT: [u8; 8] = [0xbb, 0x23, 0xef, 0x89, 0xf5, 0x0f, 0xc5, 0x95];

#[wasm_bindgen_test]
fn golden_vector_full_chain_hash_verify_recover_assemble() {
    // hash -> ciphertext round trip for both DES1/DES2 blocks.
    let pt1 = byte7_index_to_plaintext(PT1_INDEX);
    assert_eq!(netntlmv1_hash(&pt1).unwrap(), PT1_CIPHERTEXT);
    let pt2 = byte7_index_to_plaintext(PT2_INDEX);
    assert_eq!(netntlmv1_hash(&pt2).unwrap(), PT2_CIPHERTEXT);

    // recover_pt3 exhaustively finds the third DES block's plaintext.
    let pt3 = recover_pt3(&PT3_CIPHERTEXT).unwrap();
    assert_eq!(pt3, vec![0x58, 0x6c]);

    // assemble_nt_hash stitches the three recovered parts into the NT hash.
    let nt_hash = assemble_nt_hash(PT1_INDEX, PT2_INDEX, &pt3).unwrap();
    assert_eq!(hex::encode(nt_hash), "8846f7eaee8fb117ad06bdd830b7586c");

    // verify_candidates (the serial chain-walk port) hits at position 0:
    // the chain start IS the answer, no reduction needed.
    let starts = [PT1_INDEX];
    let positions = [0u32];
    let progress_calls = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let progress_calls_inner = progress_calls.clone();
    let progress = Closure::<dyn FnMut(JsValue)>::new(move |_progress: JsValue| {
        progress_calls_inner.set(progress_calls_inner.get() + 1);
    });
    let result = verify_candidates(
        &starts,
        &positions,
        &PT1_CIPHERTEXT,
        0,
        true,
        progress.as_ref().unchecked_ref(),
    )
    .expect("verify_candidates should succeed");

    // `verify_candidates` always calls back once more with the final
    // totals, even for a run that never crosses the ~250ms throttle.
    assert!(progress_calls.get() >= 1);

    assert_eq!(result.keys(), vec![PT1_INDEX]);
    assert_eq!(result.candidates_completed(), 1);
    assert_eq!(result.candidates_total(), 1);
    assert_eq!(result.steps_completed(), 1);
    assert_eq!(result.steps_total(), 1);
    assert!(!result.stopped_early());
}

#[wasm_bindgen_test]
fn crypto_wrappers_reject_wrong_length_inputs_via_jserror() {
    // Exercises the actual `JsError`-returning boundary functions, which
    // panic outside a real wasm host and so are untested by native
    // `cargo test` (see src/crypto.rs's doc comment).
    assert!(netntlmv1_hash(&[0u8; 6]).is_err());
    assert!(netntlmv1_hash(&[0u8; 8]).is_err());
    assert!(assemble_nt_hash(0, 0, &[0u8; 3]).is_err());
}

#[wasm_bindgen_test]
fn endpoint_and_candidate_formats_round_trip_through_js_array() {
    let endpoints: Vec<u64> = vec![PT1_INDEX, PT2_INDEX];
    let encoded = encode_endpoint_file(&endpoints);
    assert_eq!(&encoded[..8], b"NTLMEND1");

    // decode_candidate_file returns a js_sys::Array of CandidateRecordJs;
    // build a matching NTLMCAN1 file via the native encoder (not exercised
    // through crypto-wasm, since only decoding is exposed to JS) and
    // confirm the JS-facing getters round-trip correctly.
    let candidate_file = ntlmrain::formats::CandidateFile::new(
        2,
        vec![
            ntlmrain::formats::CandidateRecord {
                ordinal: 0,
                start: PT1_INDEX,
            },
            ntlmrain::formats::CandidateRecord {
                ordinal: 1,
                start: PT2_INDEX,
            },
        ],
    )
    .unwrap();
    let candidate_bytes = candidate_file.encode();

    let records = decode_candidate_file(&candidate_bytes, Some(2)).expect("decode should succeed");
    assert_eq!(records.length(), 2);

    // `CandidateRecordJs` is an *exported* wasm-bindgen struct, not an
    // *imported* JS type, so it has no `JsCast`/`instanceof` support to
    // downcast a `JsValue` back into it; its `ordinal`/`start` getters are
    // real JS properties on the object instead, so fetch them through
    // `js_sys::Reflect` and unwrap the `u64` getter's `bigint` value.
    let get_field = |value: &JsValue, name: &str| -> u64 {
        let field = js_sys::Reflect::get(value, &JsValue::from_str(name)).unwrap();
        let big: js_sys::BigInt = field.unchecked_into();
        String::from(big.to_string(10).unwrap()).parse().unwrap()
    };

    let first = records.get(0);
    assert_eq!(get_field(&first, "ordinal"), 0);
    assert_eq!(get_field(&first, "start"), PT1_INDEX);

    let second = records.get(1);
    assert_eq!(get_field(&second, "ordinal"), 1);
    assert_eq!(get_field(&second, "start"), PT2_INDEX);

    // Wrong expected_query_count is rejected with a JsError.
    assert!(decode_candidate_file(&candidate_bytes, Some(99)).is_err());
}
