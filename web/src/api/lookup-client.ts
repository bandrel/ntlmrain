// `fetch`-based port of `src/remote_lookup.rs`'s blocking HTTP client,
// checked branch-for-branch against that file (not approximated).
//
// Endpoints (relative to `baseUrl`, matching `RemoteLookupClient::new`'s
// `endpoint(...)` calls exactly):
//   POST /api/v1/submissions          - submit an NTLMEND1 artifact
//   POST /api/v1/submissions/status   - poll submission state
//   POST /api/v1/submissions/result   - download the NTLMCAN1 result
//   POST /api/v1/submissions/cancel   - best-effort cancel
//
// Deliberately NOT ported here: `validate_token` (256-bit lowercase hex
// submission-token shape check) and the artifact-format validation
// (`parse_endpoint_file`/`validate_candidate_file`) — those depend on
// Task 2's `crypto-wasm` formats module, which is this client's caller's
// concern (`pipeline/orchestrator.ts`), not this HTTP-transport layer's.

export interface LookupClientConfig {
  /** e.g. `https://lookup.ntlmrain.com`; trailing slashes are tolerated. */
  baseUrl: string;
  /** `undefined` disables Basic authentication; empty strings remain valid values. */
  username?: string;
  password?: string;
  /** Injectable for tests; defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

export interface SubmissionReceipt {
  submissionToken: string;
  pollWithinSeconds: number | null;
}

/** Field-for-field port of `RemoteStatus`. */
export interface StatusResponse {
  state: string;
  recordCount: number;
  processedRecords: number;
  progress: number;
  queuePosition: number | null;
  matchCount: number | null;
  error: string | null;
  pollWithinSeconds: number | null;
  downloadWithinSeconds: number | null;
}

export class LookupServiceError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "LookupServiceError";
    this.status = status;
  }
}

const KNOWN_STATES = new Set(["queued", "running", "ready", "failed"]);

/**
 * `POST {base}/api/v1/submissions`: raw NTLMEND1 bytes,
 * `Content-Type: application/vnd.netntlmv1.endpoints`.
 */
export async function submit(config: LookupClientConfig, endpointBytes: Uint8Array): Promise<SubmissionReceipt> {
  const response = await doFetch(config, "/api/v1/submissions", {
    method: "POST",
    headers: { "Content-Type": "application/vnd.netntlmv1.endpoints" },
    // `new Uint8Array(endpointBytes)` guarantees a plain `ArrayBuffer`-backed
    // view (not `ArrayBufferLike`/`SharedArrayBuffer`), which is what
    // `BodyInit`'s `ArrayBufferView` constraint requires under this
    // TypeScript version's DOM lib typings.
    body: new Uint8Array(endpointBytes),
  });
  const json = (await response.json()) as Record<string, unknown>;
  if (typeof json.submission_token !== "string") {
    throw new LookupServiceError("lookup service response is missing submission_token");
  }
  return {
    submissionToken: json.submission_token,
    pollWithinSeconds: typeof json.poll_within_seconds === "number" ? json.poll_within_seconds : null,
  };
}

/**
 * `POST {base}/api/v1/submissions/status`: `{"submission_token": token}`.
 *
 * Validates every invariant `finish_lookup`'s polling loop validates, in
 * the same order: `progress` finite and in `0..=1`, then `record_count`
 * zero or equal to `expectedRecordCount`, then that `state` is one of the
 * service's known states (an unrecognized state is a hard error, matching
 * native's `other => Err(...)` branch) — and if `state` is `"failed"`,
 * throws immediately with the service-provided error message (or a
 * fallback), matching native's `"failed" => Err(...)` branch.
 */
export async function status(
  config: LookupClientConfig,
  submissionToken: string,
  expectedRecordCount: number,
): Promise<StatusResponse> {
  const response = await doFetch(config, "/api/v1/submissions/status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ submission_token: submissionToken }),
  });
  const json = (await response.json()) as Record<string, unknown>;
  return validateStatus(json, expectedRecordCount);
}

function validateStatus(json: Record<string, unknown>, expectedRecordCount: number): StatusResponse {
  const progress = typeof json.progress === "number" ? json.progress : NaN;
  if (!Number.isFinite(progress) || progress < 0 || progress > 1) {
    throw new LookupServiceError("status progress is outside 0..=1");
  }
  const recordCount = typeof json.record_count === "number" ? json.record_count : 0;
  if (recordCount !== 0 && recordCount !== expectedRecordCount) {
    throw new LookupServiceError("status record count does not match submitted endpoints");
  }
  const state = typeof json.state === "string" ? json.state : "";
  if (!KNOWN_STATES.has(state)) {
    throw new LookupServiceError(`unexpected submission state ${JSON.stringify(state)}`);
  }
  const result: StatusResponse = {
    state,
    recordCount,
    processedRecords: typeof json.processed_records === "number" ? json.processed_records : 0,
    progress,
    queuePosition: typeof json.queue_position === "number" ? json.queue_position : null,
    matchCount: typeof json.match_count === "number" ? json.match_count : null,
    error: typeof json.error === "string" ? json.error : null,
    pollWithinSeconds: typeof json.poll_within_seconds === "number" ? json.poll_within_seconds : null,
    downloadWithinSeconds: typeof json.download_within_seconds === "number" ? json.download_within_seconds : null,
  };
  if (state === "failed") {
    throw new LookupServiceError(result.error ?? "lookup failed");
  }
  return result;
}

export interface PollOptions {
  onStatus?: (status: StatusResponse) => void;
  /** Injectable for tests; defaults to a real `setTimeout`-based sleep. */
  sleep?: (ms: number) => Promise<void>;
}

/**
 * Sleep-then-poll loop, matching `finish_lookup`'s loop exactly: sleeps
 * first, then polls at `min(2000, pollWithinSeconds * 500)` ms intervals
 * (native: `min(config.poll_interval /* default 2s *\/, poll_within_seconds
 * / 2.0)`), falling back to a straight 2000ms interval when
 * `pollWithinSeconds` is absent, non-finite, or non-positive (native's
 * `.filter(|seconds| seconds.is_finite() && *seconds > 0.0)` followed by
 * `.unwrap_or(self.config.poll_interval)`). Returns once `state ===
 * "ready"`.
 */
export async function pollUntilReady(
  config: LookupClientConfig,
  submissionToken: string,
  expectedRecordCount: number,
  pollWithinSeconds: number | null,
  options: PollOptions = {},
): Promise<StatusResponse> {
  const intervalMs =
    pollWithinSeconds !== null && Number.isFinite(pollWithinSeconds) && pollWithinSeconds > 0
      ? Math.min(2000, pollWithinSeconds * 500)
      : 2000;
  const sleep = options.sleep ?? defaultSleep;
  for (;;) {
    await sleep(intervalMs);
    const current = await status(config, submissionToken, expectedRecordCount);
    options.onStatus?.(current);
    if (current.state === "ready") {
      return current;
    }
  }
}

/** `POST {base}/api/v1/submissions/result`: returns raw NTLMCAN1 bytes. */
export async function result(config: LookupClientConfig, submissionToken: string): Promise<Uint8Array> {
  const response = await doFetch(config, "/api/v1/submissions/result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ submission_token: submissionToken }),
  });
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * `POST {base}/api/v1/submissions/cancel`: best-effort, matching native's
 * fire-and-forget `cancel_best_effort` (`let _ = ...send()`) — every error
 * is swallowed.
 */
export async function cancel(config: LookupClientConfig, submissionToken: string): Promise<void> {
  try {
    await doFetch(config, "/api/v1/submissions/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ submission_token: submissionToken }),
    });
  } catch {
    // Best-effort: swallow every failure.
  }
}

// ---------------------------------------------------------------------------
// Transport helpers
// ---------------------------------------------------------------------------

function joinUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/+$/, "")}${path}`;
}

async function doFetch(config: LookupClientConfig, path: string, init: RequestInit): Promise<Response> {
  const fetchImpl = config.fetchImpl ?? fetch;
  const headers = new Headers(init.headers);
  if (config.username !== undefined) {
    headers.set("Authorization", `Basic ${base64Encode(`${config.username}:${config.password ?? ""}`)}`);
  }
  const response = await fetchImpl(joinUrl(config.baseUrl, path), { ...init, headers });
  return requireSuccess(response);
}

/**
 * Port of `require_success`: on a non-2xx response, prefer a JSON
 * `{"detail": "..."}` body's message, else the raw (trimmed) body text if
 * non-empty, else the response's status text (native: `canonical_reason()`)
 * as a last-resort fallback.
 */
async function requireSuccess(response: Response): Promise<Response> {
  if (response.ok) return response;
  const fallback = response.statusText || "lookup request failed";
  const bodyText = await response.text().catch(() => "");
  const parsed = tryParseJson(bodyText);
  let message: string;
  if (parsed !== undefined && typeof parsed === "object" && parsed !== null && typeof (parsed as Record<string, unknown>).detail === "string") {
    message = (parsed as Record<string, unknown>).detail as string;
  } else if (bodyText.trim().length > 0) {
    message = bodyText.trim();
  } else {
    message = fallback;
  }
  throw new LookupServiceError(message, response.status);
}

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * UTF-8-safe base64 encode with no dependency on `btoa` (Latin1-only, and
 * not guaranteed present under every test runner) or `Buffer` (Node-only,
 * and this file's `tsconfig.json` restricts ambient `types` to just
 * `@webgpu/types`/`vite/client`, so `Buffer`'s types are not in scope).
 */
function base64Encode(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let output = "";
  let index = 0;
  for (; index + 3 <= bytes.length; index += 3) {
    const chunk = (bytes[index] << 16) | (bytes[index + 1] << 8) | bytes[index + 2];
    output +=
      BASE64_ALPHABET[(chunk >> 18) & 0x3f] +
      BASE64_ALPHABET[(chunk >> 12) & 0x3f] +
      BASE64_ALPHABET[(chunk >> 6) & 0x3f] +
      BASE64_ALPHABET[chunk & 0x3f];
  }
  const remaining = bytes.length - index;
  if (remaining === 1) {
    const chunk = bytes[index] << 16;
    output += BASE64_ALPHABET[(chunk >> 18) & 0x3f] + BASE64_ALPHABET[(chunk >> 12) & 0x3f] + "==";
  } else if (remaining === 2) {
    const chunk = (bytes[index] << 16) | (bytes[index + 1] << 8);
    output +=
      BASE64_ALPHABET[(chunk >> 18) & 0x3f] +
      BASE64_ALPHABET[(chunk >> 12) & 0x3f] +
      BASE64_ALPHABET[(chunk >> 6) & 0x3f] +
      "=";
  }
  return output;
}
