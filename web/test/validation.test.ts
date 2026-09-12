import { describe, expect, it } from "vitest";
import { targetKindLabel, validateInput } from "../src/ui/validation";

const DES = "AABBCCDD11223344";
const TWO_DES = DES + "5566778899AABBCC";
const FULL = TWO_DES + "0011223344556677";

describe("validateInput", () => {
  it("reports empty input as an error rather than an untouched/neutral state", () => {
    const result = validateInput("   ", "full");
    expect(result.ok).toBe(false);
  });

  it("surfaces input-parser's own error message verbatim on invalid input", () => {
    const result = validateInput("not-hex", "full");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.message).toMatch(/16, 32, or 48/);
    }
  });

  it("accepts a single DES ciphertext and flags mode match for single-des mode", () => {
    const result = validateInput(DES, "single-des");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.parsed.target.kind).toBe("des");
      expect(result.modeMatches).toBe(true);
    }
  });

  it("flags a mode mismatch when a single DES ciphertext is entered in full-response mode", () => {
    const result = validateInput(DES, "full");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.modeMatches).toBe(false);
    }
  });

  it("accepts a full response and flags mode match for full mode", () => {
    const result = validateInput(FULL, "full");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.parsed.target.kind).toBe("fullResponse");
      expect(result.modeMatches).toBe(true);
    }
  });

  it("flags a mode mismatch when a full response is entered in single-des mode", () => {
    const result = validateInput(FULL, "single-des");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.modeMatches).toBe(false);
    }
  });

  it("labels every target kind", () => {
    expect(targetKindLabel({ kind: "des", bytes: new Uint8Array(8) })).toMatch(/single 8-byte/);
    expect(targetKindLabel({ kind: "twoDes", bytes: new Uint8Array(16) })).toMatch(/16-byte/);
    expect(targetKindLabel({ kind: "fullResponse", bytes: new Uint8Array(24) })).toMatch(/full 24-byte/);
  });
});
