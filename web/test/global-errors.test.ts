import { describe, expect, it, vi } from "vitest";
import { installGlobalErrorReporter, isExtensionNoise } from "../src/ui/global-errors";

/** Minimal stand-in for the bits of `window` the reporter binds to. */
function fakeTarget() {
  const listeners = new Map<string, ((event: unknown) => void)[]>();
  return {
    addEventListener(type: string, listener: (event: unknown) => void) {
      listeners.set(type, [...(listeners.get(type) ?? []), listener]);
    },
    removeEventListener(type: string, listener: (event: unknown) => void) {
      listeners.set(type, (listeners.get(type) ?? []).filter((each) => each !== listener));
    },
    emit(type: string, event: unknown) {
      for (const listener of listeners.get(type) ?? []) listener(event);
    },
    count(type: string) {
      return (listeners.get(type) ?? []).length;
    },
  };
}

describe("isExtensionNoise", () => {
  // The message that prompted this: a Chrome extension's content script
  // returned `true` from a `chrome.runtime.onMessage` listener and never
  // replied. A page cannot produce it, and surfacing it as "Run failed"
  // would blame the app for a browser add-on.
  it("recognises the chrome.runtime message-channel error", () => {
    expect(
      isExtensionNoise(
        "A listener indicated an asynchronous response by returning true, but the message channel closed before a response was received",
      ),
    ).toBe(true);
  });

  it("recognises an invalidated extension context", () => {
    expect(isExtensionNoise("Extension context invalidated.")).toBe(true);
  });

  it("does not swallow a genuine application error", () => {
    expect(isExtensionNoise("lookup request timed out after 1800s: /api/v1/submissions/result")).toBe(false);
    expect(isExtensionNoise("WebGPU device lost")).toBe(false);
  });
});

describe("installGlobalErrorReporter", () => {
  it("reports the reason of an unhandled rejection", () => {
    const target = fakeTarget();
    const report = vi.fn();
    installGlobalErrorReporter(target, report);

    target.emit("unhandledrejection", { reason: new Error("verify stage exploded") });

    expect(report).toHaveBeenCalledWith("verify stage exploded");
  });

  it("reports an uncaught error event", () => {
    const target = fakeTarget();
    const report = vi.fn();
    installGlobalErrorReporter(target, report);

    target.emit("error", { error: new Error("null is not an object") });

    expect(report).toHaveBeenCalledWith("null is not an object");
  });

  it("stays silent for browser-extension noise", () => {
    const target = fakeTarget();
    const report = vi.fn();
    installGlobalErrorReporter(target, report);

    target.emit("unhandledrejection", {
      reason: new Error(
        "A listener indicated an asynchronous response by returning true, but the message channel closed before a response was received",
      ),
    });

    expect(report).not.toHaveBeenCalled();
  });

  // Rejections are not obliged to carry an `Error`; `Promise.reject("nope")`
  // and `Promise.reject()` both reach here.
  it("copes with a non-Error reason", () => {
    const target = fakeTarget();
    const report = vi.fn();
    installGlobalErrorReporter(target, report);

    target.emit("unhandledrejection", { reason: "plain string failure" });
    target.emit("unhandledrejection", { reason: undefined });

    expect(report).toHaveBeenNthCalledWith(1, "plain string failure");
    expect(report).toHaveBeenNthCalledWith(2, "unknown error");
  });

  it("detaches both listeners when disposed", () => {
    const target = fakeTarget();
    const report = vi.fn();

    const dispose = installGlobalErrorReporter(target, report);
    expect(target.count("unhandledrejection")).toBe(1);
    expect(target.count("error")).toBe(1);

    dispose();

    expect(target.count("unhandledrejection")).toBe(0);
    expect(target.count("error")).toBe(0);
    target.emit("unhandledrejection", { reason: new Error("after dispose") });
    expect(report).not.toHaveBeenCalled();
  });
});
