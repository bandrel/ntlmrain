// Last-resort reporting for failures that escape the local `catch` blocks.
//
// `RunController` marks a run failed when the orchestrator promise rejects,
// but anything thrown outside that chain -- an event handler, a stray
// `void`-ed promise, a WebGPU device loss -- previously reached the console
// and nowhere else. The page went on showing a stage that was already dead,
// which is indistinguishable from "slow" until you open devtools.
//
// Pure (no DOM, no `window` reference of its own), so it unit-tests against
// a fake target.

/**
 * The subset of `window` this binds to. Events arrive as `unknown` and are
 * narrowed at the point of use: `ErrorEvent` and `PromiseRejectionEvent`
 * share no field, and the DOM's `EventListener` signature does not describe
 * either precisely enough to destructure from.
 */
export interface ErrorEventTarget {
  addEventListener(type: string, listener: (event: unknown) => void): void;
  removeEventListener(type: string, listener: (event: unknown) => void): void;
}

/**
 * Messages that originate in a browser extension rather than this page.
 *
 * Chrome surfaces these on the page's console and, depending on the
 * extension, its `unhandledrejection` handler too. They are not actionable
 * by this application and reporting them as a run failure actively
 * misleads: the first one below appears whenever any installed extension
 * returns `true` from a `chrome.runtime.onMessage` listener and then never
 * replies, which has nothing to do with a lookup.
 */
const EXTENSION_NOISE: RegExp[] = [
  /message channel closed before a response was received/i,
  /A listener indicated an asynchronous response by returning true/i,
  /Extension context invalidated/i,
  /^chrome-extension:\/\//i,
  /^moz-extension:\/\//i,
];

export function isExtensionNoise(message: string): boolean {
  return EXTENSION_NOISE.some((pattern) => pattern.test(message));
}

function messageOf(value: unknown): string {
  if (value instanceof Error) return value.message;
  if (typeof value === "string" && value.length > 0) return value;
  if (value === undefined || value === null) return "unknown error";
  return String(value);
}

/**
 * Route uncaught errors and unhandled rejections to `report`, skipping
 * extension noise. Returns a disposer that detaches both listeners.
 */
export function installGlobalErrorReporter(
  target: ErrorEventTarget,
  report: (message: string) => void,
): () => void {
  const onRejection = (event: unknown) => {
    const message = messageOf((event as { reason?: unknown } | null)?.reason);
    if (!isExtensionNoise(message)) report(message);
  };
  const onError = (event: unknown) => {
    // `error` is absent for cross-origin script failures, where the spec
    // collapses everything to the opaque "Script error." string; fall back
    // to the event's own message so something still reaches the user.
    const typed = event as { error?: unknown; message?: unknown } | null;
    const message = messageOf(typed?.error ?? typed?.message);
    if (!isExtensionNoise(message)) report(message);
  };

  target.addEventListener("unhandledrejection", onRejection);
  target.addEventListener("error", onError);

  return () => {
    target.removeEventListener("unhandledrejection", onRejection);
    target.removeEventListener("error", onError);
  };
}
