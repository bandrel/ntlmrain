//! Thin `#[wasm_bindgen]` wrappers around `ntlmrain::formats`'s canonical
//! NTLMEND1/NTLMCAN1 wire formats (`ntlmrain::formats`, NOT
//! `ntlmrain::local_lookup`'s duplicate encoder/decoder).
//!
//! Only the two directions the browser actually needs are exposed:
//! endpoints are only ever *encoded* client-side (candidates are produced
//! server-side by looking them up), and candidates are only ever *decoded*
//! client-side (the server produces the NTLMCAN1 bytes; the browser reads
//! them back to drive verification).
//!
//! Decoding is split into a pure Rust step ([`decode_candidate_records`],
//! natively testable — it never touches `js_sys`) and a thin
//! `#[wasm_bindgen]`-exposed marshalling step
//! ([`decode_candidate_file`]) that builds the `js_sys::Array` handed back
//! to JS. `js_sys` types call through externs that only have a real
//! implementation inside a wasm32 host, so the marshalling step is
//! exercised by the `wasm-pack test` suite, not native `cargo test`.

use wasm_bindgen::prelude::*;

use ntlmrain::formats::{CandidateFile, CandidateRecord, EndpointFile, FormatError};

/// Encode a list of chain-endpoint indexes as an NTLMEND1 file
/// (`ntlmrain::formats::EndpointFile::encode`, verbatim).
#[wasm_bindgen]
pub fn encode_endpoint_file(endpoints: &[u64]) -> Vec<u8> {
    EndpointFile::new(endpoints.to_vec()).encode()
}

/// One decoded NTLMCAN1 record: a query ordinal paired with the rainbow
/// chain's starting index (`ntlmrain::formats::CandidateRecord`, verbatim).
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateRecordJs {
    ordinal: u64,
    start: u64,
}

#[wasm_bindgen]
impl CandidateRecordJs {
    #[wasm_bindgen(getter)]
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }

    #[wasm_bindgen(getter)]
    pub fn start(&self) -> u64 {
        self.start
    }
}

impl From<CandidateRecord> for CandidateRecordJs {
    fn from(record: CandidateRecord) -> Self {
        Self {
            ordinal: record.ordinal,
            start: record.start,
        }
    }
}

/// Pure decode step: `ntlmrain::formats::CandidateFile::decode` plus the
/// optional `expected_query_count` guard, with no `js_sys` involved. Kept
/// separate so it can be exercised by native `cargo test`.
fn decode_candidate_records(
    bytes: &[u8],
    expected_query_count: Option<u64>,
) -> Result<Vec<CandidateRecordJs>, FormatError> {
    let file = CandidateFile::decode(bytes)?;
    if let Some(expected) = expected_query_count
        && file.query_count != expected
    {
        return Err(FormatError::LengthMismatch {
            kind: "candidate query_count",
            expected: expected as usize,
            actual: file.query_count as usize,
        });
    }
    Ok(file
        .records
        .into_iter()
        .map(CandidateRecordJs::from)
        .collect())
}

/// Decode an NTLMCAN1 file (`ntlmrain::formats::CandidateFile::decode`) into
/// its ordinal/start candidate records.
///
/// If `expected_query_count` is `Some`, the decoded file's `query_count`
/// header field must match it exactly or this returns a `JsError` (guards
/// against silently pairing candidates decoded against the wrong endpoint
/// batch).
#[wasm_bindgen]
pub fn decode_candidate_file(
    bytes: &[u8],
    expected_query_count: Option<u64>,
) -> Result<js_sys::Array, JsError> {
    let records = decode_candidate_records(bytes, expected_query_count)
        .map_err(|error| JsError::new(&error.to_string()))?;
    let array = js_sys::Array::new_with_length(records.len() as u32);
    for (index, record) in records.into_iter().enumerate() {
        array.set(index as u32, JsValue::from(record));
    }
    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_endpoint_file_matches_native_encode() {
        let endpoints = vec![0x0123_4567_89ab_cdefu64, 7, 0];
        let encoded = encode_endpoint_file(&endpoints);
        assert_eq!(encoded, EndpointFile::new(endpoints).encode());
    }

    #[test]
    fn decode_candidate_records_round_trips_native_records() {
        let records = vec![
            CandidateRecord {
                ordinal: 0,
                start: 0x1111,
            },
            CandidateRecord {
                ordinal: 2,
                start: 0x2222,
            },
        ];
        let file = CandidateFile::new(3, records.clone()).unwrap();
        let bytes = file.encode();

        let decoded = decode_candidate_records(&bytes, Some(3)).unwrap();
        let expected: Vec<CandidateRecordJs> =
            records.into_iter().map(CandidateRecordJs::from).collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn decode_candidate_records_rejects_query_count_mismatch() {
        let file = CandidateFile::new(3, vec![]).unwrap();
        let bytes = file.encode();
        assert!(decode_candidate_records(&bytes, Some(4)).is_err());
        assert!(decode_candidate_records(&bytes, Some(3)).is_ok());
        assert!(decode_candidate_records(&bytes, None).is_ok());
    }

    #[test]
    fn decode_candidate_records_rejects_corrupt_bytes() {
        assert!(decode_candidate_records(&[0u8; 4], None).is_err());
    }
}
