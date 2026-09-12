// As-you-type input validation for the recovery textbox. Deliberately does
// NOT reimplement any parsing/validation rule — every rule lives in
// `pipeline/input-parser.ts`'s `parseTarget` (a verbatim port of
// `src/input.rs`), this file just interprets that function's result/thrown
// error for the recovery-mode selector's benefit.

import { InputParseError, parseTarget, type ParsedTarget, type Target } from "../pipeline/input-parser";
import type { RecoveryMode } from "../archive/db";

export interface ValidationOk {
  ok: true;
  parsed: ParsedTarget;
  /** Whether the parsed target's shape matches the selected recovery mode. */
  modeMatches: boolean;
}

export interface ValidationError {
  ok: false;
  message: string;
}

export type ValidationResult = ValidationOk | ValidationError;

/** Human-readable label for a parsed target's shape, for display next to the textbox. */
export function targetKindLabel(target: Target): string {
  switch (target.kind) {
    case "des":
      return "single 8-byte DES ciphertext";
    case "twoDes":
      return "16-byte (two DES ciphertexts, no des3)";
    case "fullResponse":
      return "full 24-byte NetNTLMv1 response";
  }
}

/**
 * `mode === "single-des"` expects exactly an 8-byte `des` target (skipping
 * des2/des3/NT-hash assembly, per this task's brief); `mode === "full"`
 * expects the full 24-byte response (or an equivalent capture line).
 */
function matchesMode(target: Target, mode: RecoveryMode): boolean {
  return mode === "single-des" ? target.kind === "des" : target.kind === "fullResponse";
}

/** Validate `input` against the recovery mode currently selected in the UI. Empty input is reported as an error, not a special "untouched" state — callers decide whether to suppress the message before the user has typed anything. */
export function validateInput(input: string, mode: RecoveryMode): ValidationResult {
  if (input.trim().length === 0) {
    return { ok: false, message: "enter a target hash or capture line" };
  }
  try {
    const parsed = parseTarget(input);
    return { ok: true, parsed, modeMatches: matchesMode(parsed.target, mode) };
  } catch (error) {
    if (error instanceof InputParseError) {
      return { ok: false, message: error.message };
    }
    throw error;
  }
}
