// DOM rendering for the recovery page. Every function here takes already-
// looked-up elements and plain data — no GPU/orchestrator-*wiring* imports;
// `main.ts` is the only place that builds the real ports/device/archive
// plumbing. `key-format.ts` (byte7-index -> plaintext/key, via Task 2's
// crypto-wasm) is a pure, already-initialized-by-the-time-we-render
// dependency, not a wiring one, so it's fine to use directly here.

import type { DesSlot } from "../pipeline/orchestrator";
import type { RunSnapshot, SlotSnapshot } from "./run-controller";
import type { ValidationResult } from "./validation";
import { targetKindLabel } from "./validation";
import { ProgressMeter, type MeterSnapshot } from "./progress-meter";
import { recoveredKeysFromIndices, toHex } from "./key-format";
import type { RunRecord, RunSummary } from "../archive/db";

// ---------------------------------------------------------------------------
// Input feedback
// ---------------------------------------------------------------------------

export function renderInputFeedback(el: HTMLElement, result: ValidationResult, touched: boolean): void {
  if (!touched) {
    el.textContent = "";
    el.removeAttribute("data-state");
    return;
  }
  if (!result.ok) {
    el.textContent = result.message;
    el.setAttribute("data-state", "error");
    return;
  }
  const kind = targetKindLabel(result.parsed.target);
  if (!result.modeMatches) {
    el.textContent = `Parsed as ${kind}, which does not match the selected target shape.`;
    el.setAttribute("data-state", "warn");
    return;
  }
  el.textContent = `Parsed as ${kind}.`;
  el.setAttribute("data-state", "ok");
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

interface StageElements {
  root: HTMLElement;
  bar: HTMLElement;
  line: HTMLElement;
}

interface SlotElements {
  root: HTMLElement;
  precompute: StageElements;
  lookup: StageElements;
  verify: StageElements;
  meters: { precompute: ProgressMeter; lookup: ProgressMeter; verify: ProgressMeter };
}

function stageElements(slotRoot: HTMLElement, stage: string): StageElements {
  const root = slotRoot.querySelector<HTMLElement>(`.nr-stage[data-stage="${stage}"]`);
  if (!root) throw new Error(`slot template is missing a "${stage}" stage`);
  const bar = root.querySelector<HTMLElement>(".nr-bar-fill");
  const line = root.querySelector<HTMLElement>(".nr-stage-line");
  if (!bar || !line) throw new Error(`"${stage}" stage is missing its bar/line elements`);
  return { root, bar, line };
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function formatCount(value: number): string {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function renderStage(elements: StageElements, snapshot: MeterSnapshot, unitLabel: string): void {
  elements.root.setAttribute("data-active", "true");
  elements.bar.style.width = `${snapshot.percent.toFixed(1)}%`;
  const rateText = snapshot.rate !== null ? `${formatCount(snapshot.rate)} ${unitLabel}/s` : "measuring rate…";
  const etaText = snapshot.etaMs !== null ? `ETA ${formatDuration(snapshot.etaMs)}` : "ETA —";
  elements.line.textContent =
    `${formatCount(snapshot.done)} / ${formatCount(snapshot.total)} ${unitLabel} ` +
    `(${snapshot.percent.toFixed(1)}%) · elapsed ${formatDuration(snapshot.elapsedMs)} · ${rateText} · ${etaText}`;
}

/** Owns one slot's (des1 or des2) three stage meters and re-renders them as new snapshots arrive. */
export class ProgressSlotView {
  private readonly elements: SlotElements;

  constructor(container: HTMLElement, template: HTMLTemplateElement, which: DesSlot) {
    const fragment = template.content.cloneNode(true) as DocumentFragment;
    const root = fragment.firstElementChild as HTMLElement;
    const title = root.querySelector<HTMLElement>(".nr-slot-title");
    if (title) title.textContent = which === "des1" ? "DES1" : "DES2";
    container.appendChild(root);

    this.elements = {
      root,
      precompute: stageElements(root, "precompute"),
      lookup: stageElements(root, "lookup"),
      verify: stageElements(root, "verify"),
      meters: { precompute: new ProgressMeter(), lookup: new ProgressMeter(), verify: new ProgressMeter() },
    };
  }

  render(slot: SlotSnapshot): void {
    if (slot.precompute) {
      const snapshot = this.elements.meters.precompute.update(slot.precompute.stepsDone, slot.precompute.stepsTotal);
      renderStage(this.elements.precompute, snapshot, "DES steps");
    }
    if (slot.lookupEvent) {
      // Read the counts off the retained `lookupStatus`, not off the current
      // event: `submitted` and `downloaded` carry no status, and reading
      // through them used to blank the bar to "0 / 0 endpoints (0.0%)".
      const status = slot.lookupStatus;
      const total = status ? status.recordCount : 0;
      // A completed lookup is 100% by definition -- the bytes are downloaded.
      // The last `ready` status can still report fewer records than it has,
      // since the server serves a checkpointed counter once a job leaves the
      // running state, so trusting it here would strand the bar near the end.
      const done = slot.lookupComplete ? total : (status?.processedRecords ?? 0);
      const snapshot = this.elements.meters.lookup.update(done, total);
      renderStage(this.elements.lookup, snapshot, "endpoints");
      this.elements.lookup.line.textContent += ` · ${slot.lookupEvent.type}`;
    }
    if (slot.verify) {
      const snapshot = this.elements.meters.verify.update(Number(slot.verify.stepsDone), Number(slot.verify.stepsTotal));
      renderStage(this.elements.verify, snapshot, "DES steps");
      const candidateNote = ` · ${slot.verify.candidatesDone}/${slot.verify.candidatesTotal} candidates, ${slot.verify.verifiedKeys} verified`;
      this.elements.verify.line.textContent += candidateNote;
    }
  }

  remove(): void {
    this.elements.root.remove();
  }
}

export class ProgressView {
  private readonly slots = new Map<DesSlot, ProgressSlotView>();

  constructor(
    private readonly panel: HTMLElement,
    private readonly container: HTMLElement,
    private readonly template: HTMLTemplateElement,
  ) {}

  reset(): void {
    for (const slot of this.slots.values()) slot.remove();
    this.slots.clear();
    this.container.innerHTML = "";
  }

  render(snapshot: RunSnapshot): void {
    const active = snapshot.status !== "idle";
    this.panel.hidden = !active;
    if (!active) return;

    this.ensureSlot("des1").render(snapshot.slots.des1);
    if (snapshot.slots.des2) {
      this.ensureSlot("des2").render(snapshot.slots.des2);
    }
  }

  private ensureSlot(which: DesSlot): ProgressSlotView {
    let slot = this.slots.get(which);
    if (!slot) {
      slot = new ProgressSlotView(this.container, this.template, which);
      this.slots.set(which, slot);
    }
    return slot;
  }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/**
 * Every string interpolated here can originate outside this code (a
 * malicious/compromised lookup service's error `detail`, or crypto-wasm
 * output derived from attacker-influenced wire bytes), so this builds real
 * DOM nodes via `textContent` throughout rather than any `innerHTML`
 * template-string concatenation — the latter would be a stored-XSS vector
 * the moment an untrusted string reached it.
 */
function roleList(label: string, values: string[]): HTMLElement {
  const dl = document.createElement("dl");
  dl.className = "nr-results-role";
  const dt = document.createElement("dt");
  dt.textContent = label;
  dl.appendChild(dt);
  for (const value of values) {
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.appendChild(dd);
  }
  return dl;
}

function paragraph(className: string, text: string): HTMLElement {
  const p = document.createElement("p");
  p.className = className;
  p.textContent = text;
  return p;
}

export function renderResults(panel: HTMLElement, body: HTMLElement, snapshot: RunSnapshot): void {
  body.innerHTML = "";
  if (snapshot.status !== "done" && snapshot.status !== "no-match" && snapshot.status !== "error") {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  if (snapshot.status === "error") {
    body.appendChild(paragraph("nr-no-match", `Run failed: ${snapshot.errorMessage ?? "unknown error"}`));
    return;
  }
  if (snapshot.status === "no-match" || !snapshot.result) {
    body.appendChild(paragraph("nr-no-match", "No match found in this table."));
    return;
  }

  const result = snapshot.result;
  const des1 = recoveredKeysFromIndices(result.des1Keys);
  body.appendChild(roleList("DES1 plaintext(s)", des1.map((entry) => toHex(entry.plaintext))));
  body.appendChild(roleList("DES1 key(s)", des1.map((entry) => toHex(entry.key))));
  if (result.des2Keys.length > 0) {
    const des2 = recoveredKeysFromIndices(result.des2Keys);
    body.appendChild(roleList("DES2 plaintext(s)", des2.map((entry) => toHex(entry.plaintext))));
    body.appendChild(roleList("DES2 key(s)", des2.map((entry) => toHex(entry.key))));
  }
  if (result.pt3 !== null) {
    body.appendChild(roleList("DES3 plaintext", [toHex(result.pt3)]));
  }
  if (result.ntHashes.length > 0) {
    body.appendChild(roleList("Assembled NT hash(es)", result.ntHashes.map((entry) => toHex(entry.ntHash))));
  }
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

export function renderArchiveList(
  listEl: HTMLElement,
  summaries: RunSummary[],
  selectedRunId: string | null,
  onSelect: (runId: string) => void,
): void {
  listEl.innerHTML = "";
  if (summaries.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = "No runs archived yet.";
    empty.style.cursor = "default";
    listEl.appendChild(empty);
    return;
  }
  for (const summary of summaries) {
    const item = document.createElement("li");
    item.dataset.selected = String(summary.run_id === selectedRunId);

    // `archive/db.ts` reads these back with a blind cast and no runtime
    // shape check, so treat every field here as untrusted, the same as
    // `renderArchiveDetail`/`renderResults` below — no `innerHTML` template-
    // string concatenation, ever, even for fields that happen to be
    // numeric/enum/date-derived today.
    const when = new Date(summary.created_at).toLocaleString();
    const topLine = document.createElement("div");
    topLine.textContent = `${when} · ${summary.input_mode}`;

    const bottomLine = document.createElement("div");
    if (summary.matched) {
      const hashSpan = document.createElement("span");
      hashSpan.className = "nr-archive-run-hash";
      hashSpan.textContent = summary.nt_hash_hex ?? "matched";
      bottomLine.appendChild(hashSpan);
    } else {
      bottomLine.textContent = "no match";
    }

    item.appendChild(topLine);
    item.appendChild(bottomLine);
    item.addEventListener("click", () => onSelect(summary.run_id));
    listEl.appendChild(item);
  }
}

export function renderArchiveDetail(detailEl: HTMLElement, record: RunRecord | null): void {
  detailEl.innerHTML = "";
  if (!record) {
    const hint = document.createElement("p");
    hint.className = "nr-hint";
    hint.textContent = "Select a run to see its full detail.";
    detailEl.appendChild(hint);
    return;
  }
  // `record.input.raw` is the user's own pasted target/capture text, so this
  // MUST go through `textContent` (never `innerHTML`) — a malicious capture
  // line is otherwise a stored-XSS vector the moment a past run is viewed.
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(record, jsonReplacer, 2);
  detailEl.appendChild(pre);
}

function jsonReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? value.toString() : value;
}
