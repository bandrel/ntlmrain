// Thin re-export wrapper around Task 2's `crypto-wasm` crate, built via
// `wasm-pack build --target web` in ../../../crypto-wasm (relative to this
// file) and imported by local path — deliberately not published to npm.
//
// Callers MUST `await initCrypto()` once before calling any other export
// here (wasm-bindgen's `--target web` output requires an explicit async
// init that instantiates the .wasm module; nothing here does it lazily,
// to keep control over *when* the network/compile work happens in the
// caller's hands).
//
// NOTE: crypto-wasm/pkg/ is a build artifact (crypto-wasm/pkg/.gitignore
// ignores it) and is not committed. Build it once before running `web/`:
//   cd crypto-wasm && wasm-pack build --target web --out-dir pkg

import initCryptoWasm, {
  init_panic_hook,
  assemble_nt_hash,
  byte7_hash_to_index,
  byte7_index_to_plaintext,
  decode_candidate_file,
  encode_endpoint_file,
  expand_des_key,
  is_exact_des_key_match,
  netntlmv1_hash,
  recover_pt3,
  verify_candidates,
  CandidateRecordJs,
  VerifyProgressJs,
  VerifyResultJs,
} from "../../../crypto-wasm/pkg/crypto_wasm.js";

let initialized: Promise<void> | null = null;

/**
 * Instantiate the crypto-wasm module and install its panic hook. Safe to
 * call more than once (subsequent calls resolve the same promise).
 */
export function initCrypto(): Promise<void> {
  if (!initialized) {
    initialized = initCryptoWasm().then(() => {
      init_panic_hook();
    });
  }
  return initialized;
}

export {
  assemble_nt_hash,
  byte7_hash_to_index,
  byte7_index_to_plaintext,
  decode_candidate_file,
  encode_endpoint_file,
  expand_des_key,
  is_exact_des_key_match,
  netntlmv1_hash,
  recover_pt3,
  verify_candidates,
  CandidateRecordJs,
  VerifyProgressJs,
  VerifyResultJs,
};

export type { InitInput, InitOutput, SyncInitInput } from "../../../crypto-wasm/pkg/crypto_wasm.js";
