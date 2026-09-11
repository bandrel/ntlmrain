// `runOrchestrator`'s `des1Keys`/`des2Keys` are `byte7` indices (`u64`s
// identifying a position in the 7-byte plaintext space — see
// `crypto-wasm/src/crypto.rs`'s `byte7_index_to_plaintext`/`byte7_hash_to_index`
// and `crypto-wasm/src/verify.rs`'s `VerifyOutcome.keys: Vec<u64>`), NOT the
// recovered plaintext or DES key bytes themselves. The results panel and
// the run archive both want the actual plaintext (7 bytes) and its
// expanded 8-byte DES key (parity bits set), so this converts an index to
// both, using Task 2's `crypto-wasm` exports directly.

import { byte7_index_to_plaintext, expand_des_key } from "../crypto";

export interface RecoveredKey {
  index: bigint;
  plaintext: Uint8Array;
  key: Uint8Array;
}

export function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Requires `initCrypto()` to have already resolved (same contract as every other `crypto-wasm` export). */
export function recoveredKeysFromIndices(indices: bigint[]): RecoveredKey[] {
  return indices.map((index) => {
    const plaintext = byte7_index_to_plaintext(index);
    const key = expand_des_key(plaintext);
    return { index, plaintext, key };
  });
}
