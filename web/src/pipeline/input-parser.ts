// Port of `src/input.rs`'s target parsing, verbatim (checked branch-for-
// branch against that file, not approximated):
// - Raw mode strips all whitespace, requires exactly 16/32/48 hex chars.
// - Capture mode (contains "::") accepts `USER::DOMAIN:LM:NT:CHALLENGE`
//   (6 fields, field[1] empty) or `USER::DOMAIN:NT:CHALLENGE` (5 fields,
//   field[1] empty).
// - The NTLMv1-ESS rejection (`lm_response[8..24]` all zero) is checked
//   BEFORE the fixed-challenge check, matching `parse_capture`'s ordering
//   in `src/input.rs` exactly (see the comment there: "The NT response
//   cannot be used with this table").

const FIXED_CHALLENGE_HEX = "1122334455667788";
const FIXED_CHALLENGE = hexToBytesUnchecked(FIXED_CHALLENGE_HEX);

export type Target =
  | { kind: "des"; bytes: Uint8Array }
  | { kind: "twoDes"; bytes: Uint8Array }
  | { kind: "fullResponse"; bytes: Uint8Array };

export interface CaptureMetadata {
  username: string;
  domain: string;
  lmResponse: Uint8Array | null;
}

export interface ParsedTarget {
  target: Target;
  capture: CaptureMetadata | null;
}

export type InputErrorKind =
  | "invalid-raw-length"
  | "invalid-hex"
  | "invalid-capture"
  | "invalid-capture-field"
  | "ntlmv1-ess"
  | "unsupported-challenge";

/** Mirrors `src/input.rs`'s `InputError` enum, field-for-field. */
export class InputParseError extends Error {
  readonly kind: InputErrorKind;
  readonly field?: string;
  readonly actual?: string;

  constructor(kind: InputErrorKind, message: string, extra?: { field?: string; actual?: string }) {
    super(message);
    this.name = "InputParseError";
    this.kind = kind;
    this.field = extra?.field;
    this.actual = extra?.actual;
  }
}

/**
 * Parse raw 16/32/48-character hexadecimal input or a conventional capture
 * (`USER::DOMAIN:LM:NT:CHALLENGE` / `USER::DOMAIN:NT:CHALLENGE`), matching
 * `src/input.rs::parse_target` exactly.
 */
export function parseTarget(input: string): ParsedTarget {
  const trimmed = input.trim();
  if (trimmed.includes("::")) {
    return parseCapture(trimmed);
  }
  return parseRaw(trimmed);
}

function parseRaw(input: string): ParsedTarget {
  const normalized = Array.from(input)
    .filter((character) => !isWhitespace(character))
    .join("");
  if (normalized.length !== 16 && normalized.length !== 32 && normalized.length !== 48) {
    throw new InputParseError("invalid-raw-length", "target must be 16, 32, or 48 hexadecimal characters");
  }
  const decoded = decodeHex(normalized);
  if (!decoded) {
    throw new InputParseError("invalid-hex", "target contains non-hexadecimal characters");
  }
  let target: Target;
  switch (decoded.length) {
    case 8:
      target = { kind: "des", bytes: decoded };
      break;
    case 16:
      target = { kind: "twoDes", bytes: decoded };
      break;
    case 24:
      target = { kind: "fullResponse", bytes: decoded };
      break;
    default:
      // Unreachable: normalized.length was validated above to be 16/32/48
      // hex chars, i.e. exactly 8/16/24 decoded bytes.
      throw new Error("unreachable: validated raw target length");
  }
  return { target, capture: null };
}

function parseCapture(input: string): ParsedTarget {
  const fields = input.split(":");
  let username: string;
  let domain: string;
  let lmText: string | null;
  let ntText: string;
  let challengeText: string;

  if (fields.length === 6 && fields[1] === "") {
    [username, , domain, lmText, ntText, challengeText] = fields;
  } else if (fields.length === 5 && fields[1] === "") {
    [username, , domain, ntText, challengeText] = fields;
    lmText = null;
  } else {
    throw new InputParseError(
      "invalid-capture",
      "invalid NetNTLMv1 capture; expected USER::DOMAIN:LM_RESPONSE:NT_RESPONSE:CHALLENGE",
    );
  }

  if (username.length === 0) {
    throw new InputParseError("invalid-capture-field", "NetNTLMv1 capture contains an invalid username", {
      field: "username",
    });
  }

  const ntResponse = decodeExact(ntText, 24, "NT response");
  const challenge = decodeExact(challengeText, 8, "challenge");
  const lmResponse = lmText !== null && lmText.length > 0 ? decodeExact(lmText, 24, "LM response") : null;

  // NTLMv1-ESS encodes an 8-byte client challenge followed by sixteen zero
  // bytes in the LM response. The NT response cannot be used with this
  // table. This check MUST come before the challenge check below, matching
  // `src/input.rs::parse_capture`'s ordering exactly.
  if (lmResponse !== null && allZero(lmResponse.subarray(8))) {
    throw new InputParseError("ntlmv1-ess", "NTLMv1-ESS captures are not supported by this table");
  }
  if (!bytesEqual(challenge, FIXED_CHALLENGE)) {
    const actual = toHexUpper(challenge);
    throw new InputParseError(
      "unsupported-challenge",
      `capture challenge ${actual} is unsupported; this table requires ${FIXED_CHALLENGE_HEX}`,
      { actual },
    );
  }

  return {
    target: { kind: "fullResponse", bytes: ntResponse },
    capture: { username, domain, lmResponse },
  };
}

/** The ciphertexts covered by the rainbow table (DES1 and optionally DES2). */
export function desTargets(target: Target): Uint8Array[] {
  const bytes = target.bytes;
  const targets = [bytes.slice(0, 8)];
  if (bytes.length >= 16) {
    targets.push(bytes.slice(8, 16));
  }
  return targets;
}

/** DES3 ciphertext, only for a 24-byte FullResponse target. */
export function k3Ciphertext(target: Target): Uint8Array | null {
  if (target.kind !== "fullResponse") {
    return null;
  }
  return target.bytes.slice(16, 24);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function decodeExact(input: string, byteLength: number, field: string): Uint8Array {
  if (input.length !== byteLength * 2) {
    throw new InputParseError("invalid-capture-field", `NetNTLMv1 capture contains an invalid ${field}`, {
      field,
    });
  }
  const decoded = decodeHex(input);
  if (!decoded) {
    throw new InputParseError("invalid-capture-field", `NetNTLMv1 capture contains an invalid ${field}`, {
      field,
    });
  }
  return decoded;
}

function isWhitespace(character: string): boolean {
  return /\s/.test(character);
}

function isHexDigit(character: string): boolean {
  return /^[0-9a-fA-F]$/.test(character);
}

/** Returns `null` for odd length or any non-hex-digit character. */
function decodeHex(input: string): Uint8Array | null {
  if (input.length % 2 !== 0) return null;
  for (const character of input) {
    if (!isHexDigit(character)) return null;
  }
  const bytes = new Uint8Array(input.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = parseInt(input.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

/** Only for the fixed challenge constant above, which is a valid hex literal by construction. */
function hexToBytesUnchecked(input: string): Uint8Array {
  const decoded = decodeHex(input);
  if (!decoded) throw new Error(`unreachable: invalid hex literal ${input}`);
  return decoded;
}

function allZero(bytes: Uint8Array): boolean {
  return bytes.every((byte) => byte === 0);
}

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return false;
  }
  return true;
}

function toHexUpper(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"))
    .join("")
    .toUpperCase();
}
