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
// `validate_token` (256-bit lowercase hex submission-token shape check) IS
// ported here (`validateSubmissionToken`, called from `submit` right after
// parsing the receipt, and again at the top of `cancel` — the same two call
// sites as native's `remote_lookup.rs:204`/`:220`): it has no crypto-wasm
// dependency, unlike the artifact-format validation
// (`parse_endpoint_file`/`validate_candidate_file`), which DOES depend on
// Task 2's `crypto-wasm` formats module and is deliberately left to this
// client's caller (`pipeline/orchestrator.ts`) instead.

export interface LookupClientConfig {
  /** e.g. `https://lookup.ntlmrain.com`; trailing slashes are tolerated. */
  baseUrl: string;
  /** `undefined` disables Basic authentication; empty strings remain valid values. */
  username?: string;
  password?: string;
  /** Injectable for tests; defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
  /**
   * Whole-request deadline in milliseconds, covering the body read as well
   * as the headers. Port of `RemoteLookupConfig::request_timeout`, and like
   * reqwest's `Client::timeout` it bounds the entire exchange rather than
   * just the connect. Defaults to `DEFAULT_REQUEST_TIMEOUT_MS`.
   */
  requestTimeoutMs?: number;
}

/**
 * 30 minutes, matching `RemoteLookupConfig::default`'s
 * `Duration::from_secs(30 * 60)`. Generous on purpose: a submission uploads
 * an endpoint file with hundreds of thousands of records and the result
 * download can be comparably large, so this is a "the service has gone
 * away" backstop, not a latency budget.
 */
export const DEFAULT_REQUEST_TIMEOUT_MS = 30 * 60 * 1000;

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
 * Port of `validate_token`: the capability token must be exactly 64
 * lowercase-hex characters (256 bits). Native calls this immediately after
 * parsing a submission receipt (`remote_lookup.rs:204`) and again at the
 * top of its public `cancel` (`:220`) — this is the security boundary that
 * stops a malicious/compromised lookup service from echoing an arbitrary
 * string back as a "capability token" that then gets replayed into every
 * subsequent status/result/cancel request body.
 */
function validateSubmissionToken(token: string): void {
  const isValid =
    token.length === 64 &&
    Array.from(token).every((character) => (character >= "0" && character <= "9") || (character >= "a" && character <= "f"));
  if (!isValid) {
    throw new LookupServiceError("submission token is not 256-bit lowercase hexadecimal");
  }
}

/**
 * `POST {base}/api/v1/submissions`: raw NTLMEND1 bytes,
 * `Content-Type: application/vnd.netntlmv1.endpoints`.
 */
export async function submit(config: LookupClientConfig, endpointBytes: Uint8Array): Promise<SubmissionReceipt> {
  const json = (await doFetch(
    config,
    "/api/v1/submissions",
    {
      method: "POST",
      headers: { "Content-Type": "application/vnd.netntlmv1.endpoints" },
      // `new Uint8Array(endpointBytes)` guarantees a plain `ArrayBuffer`-backed
      // view (not `ArrayBufferLike`/`SharedArrayBuffer`), which is what
      // `BodyInit`'s `ArrayBufferView` constraint requires under this
      // TypeScript version's DOM lib typings.
      body: new Uint8Array(endpointBytes),
    },
    (response) => response.json(),
  )) as Record<string, unknown>;
  if (typeof json.submission_token !== "string") {
    throw new LookupServiceError("lookup service response is missing submission_token");
  }
  validateSubmissionToken(json.submission_token);
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
  const json = (await doFetch(
    config,
    "/api/v1/submissions/status",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ submission_token: submissionToken }),
    },
    (response) => response.json(),
  )) as Record<string, unknown>;
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
  return doFetch(
    config,
    "/api/v1/submissions/result",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ submission_token: submissionToken }),
    },
    async (response) => new Uint8Array(await response.arrayBuffer()),
  );
}

/**
 * `POST {base}/api/v1/submissions/cancel`: best-effort, matching native's
 * fire-and-forget `cancel_best_effort` (`let _ = ...send()`) — every error
 * is swallowed.
 */
export async function cancel(config: LookupClientConfig, submissionToken: string): Promise<void> {
  try {
    validateSubmissionToken(submissionToken);
    await doFetch(
      config,
      "/api/v1/submissions/cancel",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ submission_token: submissionToken }),
      },
      // Body is irrelevant, but it still has to be drained inside the guarded
      // window rather than left dangling on an aborted-at-timeout request.
      async (response) => void (await response.arrayBuffer()),
    );
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

/**
 * Port of reqwest's whole-request `timeout`. `read` consumes the body while
 * the abort controller is still armed — `result` streams a multi-megabyte
 * NTLMCAN1 payload after the headers land, and that leg is the one most
 * likely to stall, so returning the `Response` and clearing the timer here
 * would leave the download unguarded.
 *
 * The timer is always cleared: a seven-minute lookup polls ~200 times, and
 * a leaked 30-minute timer per request would pile up for the life of the
 * page.
 */
async function doFetch<T>(
  config: LookupClientConfig,
  path: string,
  init: RequestInit,
  read: (response: Response) => Promise<T>,
): Promise<T> {
  const fetchImpl = config.fetchImpl ?? fetch;
  const headers = new Headers(init.headers);
  if (config.username !== undefined) {
    headers.set("Authorization", `Basic ${base64Encode(`${config.username}:${config.password ?? ""}`)}`);
  }
  const timeoutMs = config.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const controller = new AbortController();
  // Distinguishes our own deadline from a caller-supplied abort or a network
  // error that merely looks like one, so only a real timeout is reported as
  // one (native: `error.is_timeout()`).
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetchImpl(joinUrl(config.baseUrl, path), {
      ...init,
      headers,
      signal: controller.signal,
    });
    return await read(await requireSuccess(response));
  } catch (error) {
    if (timedOut) {
      throw new LookupServiceError(`lookup request timed out after ${Math.round(timeoutMs / 1000)}s: ${path}`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
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
