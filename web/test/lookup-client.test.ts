import { describe, expect, it, vi } from "vitest";
import {
  cancel,
  pollUntilReady,
  result,
  status,
  submit,
  type LookupClientConfig,
} from "../src/api/lookup-client";

function jsonResponse(body: unknown, init: { status?: number; statusText?: string } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    statusText: init.statusText,
    headers: { "Content-Type": "application/json" },
  });
}

function config(fetchImpl: typeof fetch, extra: Partial<LookupClientConfig> = {}): LookupClientConfig {
  return { baseUrl: "https://lookup.example.test/", fetchImpl, ...extra };
}

describe("submit", () => {
  it("POSTs the raw endpoint bytes with the correct content type and parses the receipt", async () => {
    const calls: { url: string; init: RequestInit }[] = [];
    const fetchImpl = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: url.toString(), init: init ?? {} });
      return jsonResponse({ submission_token: "a".repeat(64), poll_within_seconds: 4 });
    }) as unknown as typeof fetch;

    const bytes = new Uint8Array([1, 2, 3, 4]);
    const receipt = await submit(config(fetchImpl), bytes);

    expect(receipt).toEqual({ submissionToken: "a".repeat(64), pollWithinSeconds: 4 });
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://lookup.example.test/api/v1/submissions");
    expect(calls[0].init.method).toBe("POST");
    const headers = new Headers(calls[0].init.headers);
    expect(headers.get("Content-Type")).toBe("application/vnd.netntlmv1.endpoints");
    expect(new Uint8Array(calls[0].init.body as ArrayBuffer)).toEqual(bytes);
  });

  it("adds a Basic Authorization header when credentials are configured", async () => {
    let seenAuth: string | null = null;
    const fetchImpl = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      seenAuth = new Headers(init?.headers).get("Authorization");
      return jsonResponse({ submission_token: "b".repeat(64) });
    }) as unknown as typeof fetch;

    await submit(config(fetchImpl, { username: "alice", password: "secret" }), new Uint8Array([1]));

    expect(seenAuth).toBe(`Basic ${btoa("alice:secret")}`);
  });

  it("throws with the JSON detail message on a non-2xx response", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "bad batch" }), { status: 400, statusText: "Bad Request" }),
    ) as unknown as typeof fetch;

    await expect(submit(config(fetchImpl), new Uint8Array([1]))).rejects.toMatchObject({
      message: "bad batch",
      status: 400,
    });
  });

  it("falls back to the raw body text when the body isn't JSON-shaped", async () => {
    const fetchImpl = vi.fn(async () => new Response("plain text failure", { status: 500 })) as unknown as typeof fetch;
    await expect(submit(config(fetchImpl), new Uint8Array([1]))).rejects.toMatchObject({
      message: "plain text failure",
    });
  });

  it("falls back to the status text when the body is empty", async () => {
    const fetchImpl = vi.fn(async () => new Response("", { status: 503, statusText: "Service Unavailable" })) as unknown as typeof fetch;
    await expect(submit(config(fetchImpl), new Uint8Array([1]))).rejects.toMatchObject({
      message: "Service Unavailable",
    });
  });
});

describe("status", () => {
  it("parses a well-formed status response", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({
        state: "running",
        record_count: 3,
        processed_records: 1,
        progress: 0.5,
        queue_position: 2,
      }),
    ) as unknown as typeof fetch;

    const current = await status(config(fetchImpl), "token", 3);
    expect(current).toMatchObject({ state: "running", recordCount: 3, progress: 0.5, queuePosition: 2 });
  });

  it("sends the submission_token as a JSON body", async () => {
    let sentBody: string | undefined;
    const fetchImpl = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      sentBody = init?.body as string;
      return jsonResponse({ state: "queued", progress: 0, record_count: 0 });
    }) as unknown as typeof fetch;

    await status(config(fetchImpl), "abc123", 5);
    expect(JSON.parse(sentBody!)).toEqual({ submission_token: "abc123" });
  });

  it("rejects a progress value outside 0..=1", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "running", progress: 1.5, record_count: 0 })) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 0)).rejects.toThrow(/progress is outside 0..=1/);
  });

  it("rejects a non-finite progress value", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "running", progress: null, record_count: 0 })) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 0)).rejects.toThrow(/progress is outside 0..=1/);
  });

  it("rejects a record_count that doesn't match the submitted endpoint count", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "running", progress: 0, record_count: 4 })) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 3)).rejects.toThrow(/record count does not match/);
  });

  it("accepts a record_count of 0 regardless of the expected count", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "queued", progress: 0, record_count: 0 })) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 99)).resolves.toMatchObject({ state: "queued" });
  });

  it("treats an unrecognized state as a hard error", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "bogus", progress: 0, record_count: 0 })) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 0)).rejects.toThrow(/unexpected submission state/);
  });

  it("throws the server-provided error message when state is failed", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({ state: "failed", progress: 1, record_count: 1, error: "table read failed" }),
    ) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 1)).rejects.toThrow("table read failed");
  });

  it("falls back to a generic message when a failed state has no error field", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "failed", progress: 1, record_count: 0 })) as unknown as typeof fetch;
    await expect(status(config(fetchImpl), "token", 0)).rejects.toThrow("lookup failed");
  });
});

describe("pollUntilReady", () => {
  it("sleeps before the first poll and stops once state is ready", async () => {
    const states = ["queued", "running", "ready"];
    let callIndex = 0;
    const fetchImpl = vi.fn(async () => {
      const state = states[callIndex];
      callIndex += 1;
      return jsonResponse({ state, progress: callIndex / states.length, record_count: 0 });
    }) as unknown as typeof fetch;

    const sleepCalls: number[] = [];
    const sleep = vi.fn(async (ms: number) => {
      sleepCalls.push(ms);
    });

    const seenStates: string[] = [];
    const finalStatus = await pollUntilReady(config(fetchImpl), "token", 0, 4, {
      sleep,
      onStatus: (current) => seenStates.push(current.state),
    });

    expect(finalStatus.state).toBe("ready");
    expect(seenStates).toEqual(["queued", "running", "ready"]);
    // sleep is called once per poll (3 polls here), always before the fetch.
    expect(sleepCalls).toHaveLength(3);
    // min(2000, pollWithinSeconds * 500) = min(2000, 4 * 500) = 2000.
    expect(sleepCalls.every((ms) => ms === 2000)).toBe(true);
  });

  it("uses min(2000, pollWithinSeconds * 500) when that is smaller than 2000", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "ready", progress: 1, record_count: 0 })) as unknown as typeof fetch;
    const sleep = vi.fn(async () => {});
    await pollUntilReady(config(fetchImpl), "token", 0, 1, { sleep });
    expect(sleep).toHaveBeenCalledWith(500); // min(2000, 1 * 500) = 500
  });

  it("falls back to a straight 2000ms interval when pollWithinSeconds is absent/non-finite/non-positive", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ state: "ready", progress: 1, record_count: 0 })) as unknown as typeof fetch;
    for (const pollWithinSeconds of [null, NaN, 0, -5]) {
      const sleep = vi.fn(async () => {});
      // eslint-disable-next-line no-await-in-loop
      await pollUntilReady(config(fetchImpl), "token", 0, pollWithinSeconds, { sleep });
      expect(sleep).toHaveBeenCalledWith(2000);
    }
  });
});

describe("result", () => {
  it("returns the raw response bytes", async () => {
    const expected = new Uint8Array([0x4e, 0x54, 0x4c, 0x4d]);
    const fetchImpl = vi.fn(async () => new Response(expected)) as unknown as typeof fetch;
    const bytes = await result(config(fetchImpl), "token");
    expect(bytes).toEqual(expected);
  });

  it("sends the submission_token as a JSON body", async () => {
    let sentBody: string | undefined;
    const fetchImpl = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      sentBody = init?.body as string;
      return new Response(new Uint8Array([1]));
    }) as unknown as typeof fetch;
    await result(config(fetchImpl), "xyz");
    expect(JSON.parse(sentBody!)).toEqual({ submission_token: "xyz" });
  });
});

describe("cancel", () => {
  it("swallows network errors (best-effort, fire-and-forget)", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("network down");
    }) as unknown as typeof fetch;
    await expect(cancel(config(fetchImpl), "token")).resolves.toBeUndefined();
  });

  it("swallows a non-2xx response", async () => {
    const fetchImpl = vi.fn(async () => new Response("", { status: 500 })) as unknown as typeof fetch;
    await expect(cancel(config(fetchImpl), "token")).resolves.toBeUndefined();
  });

  it("still issues the request when it succeeds", async () => {
    const fetchImpl = vi.fn(async () => new Response("", { status: 204 })) as unknown as typeof fetch;
    await cancel(config(fetchImpl), "token");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});
