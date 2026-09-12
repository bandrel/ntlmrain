import { describe, expect, it } from "vitest";
import { InputParseError, desTargets, k3Ciphertext, parseTarget } from "../src/pipeline/input-parser";

const RESPONSE = "727B4E35F947129EA52B9CDEDAE86934BB23EF89F50FC595";

function hexBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

describe("parseTarget - raw mode", () => {
  it("parses all raw lengths case-insensitively", () => {
    expect(parseTarget("727b4e35f947129e").target.kind).toBe("des");
    expect(parseTarget("727B4E35F947129EA52B9CDEDAE86934").target.kind).toBe("twoDes");
    const parsed = parseTarget(RESPONSE);
    expect(parsed.target.kind).toBe("fullResponse");
    expect(parsed.target.bytes).toEqual(hexBytes(RESPONSE));
    expect(parsed.capture).toBeNull();
  });

  it("strips surrounding and embedded whitespace", () => {
    const parsed = parseTarget("  727B4E35 F947129E\n");
    expect(parsed.target.bytes).toEqual(hexBytes("727B4E35F947129E"));
  });

  it("rejects an invalid raw length", () => {
    expect(() => parseTarget("abcd")).toThrowError(InputParseError);
    try {
      parseTarget("abcd");
      expect.fail("expected InputParseError");
    } catch (error) {
      expect(error).toBeInstanceOf(InputParseError);
      expect((error as InputParseError).kind).toBe("invalid-raw-length");
    }
  });

  it("rejects non-hexadecimal characters", () => {
    expect(() => parseTarget("ZZZZZZZZZZZZZZZZ")).toThrowError(InputParseError);
    try {
      parseTarget("ZZZZZZZZZZZZZZZZ");
      expect.fail("expected InputParseError");
    } catch (error) {
      expect((error as InputParseError).kind).toBe("invalid-hex");
    }
  });
});

describe("parseTarget - capture mode", () => {
  it("parses a Responder-style capture with an LM response", () => {
    const lm = "AABBCCDDEEFF00112233445566778899AABBCCDDEEFF0011";
    const capture = `alice::DOMAIN:${lm}:${RESPONSE}:1122334455667788`;
    const parsed = parseTarget(capture);
    expect(parsed.target.kind).toBe("fullResponse");
    expect(parsed.target.bytes).toEqual(hexBytes(RESPONSE));
    expect(parsed.capture).not.toBeNull();
    expect(parsed.capture?.username).toBe("alice");
    expect(parsed.capture?.domain).toBe("DOMAIN");
    expect(parsed.capture?.lmResponse).toEqual(hexBytes(lm));
  });

  it("parses a capture without an LM response", () => {
    const capture = `alice::DOMAIN:${RESPONSE}:1122334455667788`;
    const parsed = parseTarget(capture);
    expect(parsed.capture?.lmResponse).toBeNull();
  });

  it("rejects an arbitrary (non-fixed) challenge", () => {
    const capture = `alice::DOMAIN::${RESPONSE}:0102030405060708`;
    try {
      parseTarget(capture);
      expect.fail("expected InputParseError");
    } catch (error) {
      expect((error as InputParseError).kind).toBe("unsupported-challenge");
      expect((error as InputParseError).actual).toBe("0102030405060708".toUpperCase());
    }
  });

  it("rejects NTLMv1-ESS captures BEFORE validating the challenge", () => {
    // An arbitrary (also-unsupported) challenge, paired with an
    // NTLMv1-ESS-shaped LM response. If challenge validation ran first this
    // would surface "unsupported-challenge" instead; the native ordering
    // (src/input.rs::parse_capture) checks NTLMv1-ESS first.
    const lm = "0102030405060708" + "00".repeat(16);
    const capture = `alice::DOMAIN:${lm}:${RESPONSE}:0102030405060708`;
    try {
      parseTarget(capture);
      expect.fail("expected InputParseError");
    } catch (error) {
      expect((error as InputParseError).kind).toBe("ntlmv1-ess");
    }
  });

  it("rejects invalid capture shapes and fields", () => {
    try {
      parseTarget("alice:DOMAIN:bad");
      expect.fail("expected InputParseError");
    } catch (error) {
      // Contains no "::", so this actually falls into raw-mode parsing and
      // fails hex validation there (matches native's identical routing).
      expect((error as InputParseError).kind).toBe("invalid-hex");
    }

    try {
      parseTarget("::DOMAIN::0011:1122");
      expect.fail("expected InputParseError");
    } catch (error) {
      expect((error as InputParseError).kind).toBe("invalid-capture-field");
      expect((error as InputParseError).field).toBe("username");
    }
  });

  it("rejects a malformed field count", () => {
    try {
      parseTarget(`alice::DOMAIN:${RESPONSE}:1122334455667788:extra:fields`);
      expect.fail("expected InputParseError");
    } catch (error) {
      expect((error as InputParseError).kind).toBe("invalid-capture");
    }
  });

  it("rejects a wrong-length NT response field", () => {
    try {
      parseTarget("alice::DOMAIN:aabb:1122334455667788");
      expect.fail("expected InputParseError");
    } catch (error) {
      expect((error as InputParseError).kind).toBe("invalid-capture-field");
      expect((error as InputParseError).field).toBe("NT response");
    }
  });
});

describe("desTargets", () => {
  it("returns one 8-byte target for a Des target", () => {
    const parsed = parseTarget("727b4e35f947129e");
    const targets = desTargets(parsed.target);
    expect(targets).toHaveLength(1);
    expect(targets[0]).toEqual(hexBytes("727b4e35f947129e"));
  });

  it("returns two 8-byte targets for TwoDes/FullResponse targets", () => {
    const parsed = parseTarget(RESPONSE);
    const targets = desTargets(parsed.target);
    expect(targets).toHaveLength(2);
    expect(targets[0]).toEqual(hexBytes("727B4E35F947129E"));
    expect(targets[1]).toEqual(hexBytes("A52B9CDEDAE86934"));
  });
});

describe("k3Ciphertext", () => {
  it("is null for Des/TwoDes targets", () => {
    expect(k3Ciphertext(parseTarget("727b4e35f947129e").target)).toBeNull();
    expect(k3Ciphertext(parseTarget("727B4E35F947129EA52B9CDEDAE86934").target)).toBeNull();
  });

  it("is the last 8 bytes for a FullResponse target", () => {
    const parsed = parseTarget(RESPONSE);
    expect(k3Ciphertext(parsed.target)).toEqual(hexBytes("BB23EF89F50FC595"));
  });
});
