// DOM rendering for the recovery page. Every function here takes already-
// looked-up elements and plain data — no orchestrator/GPU/archive imports —
// so `main.ts` is the only place that wires this to the real pipeline.

import type { DesSlot } from "../pipeline/orchestrator";
import type { RunSnapshot, SlotSnapshot } from "./run-controller";
import type { ValidationResult } from "./validation";
import { targetKindLabel } from "./validation";
import { ProgressMeter, type MeterSnapshot } from "./progress-meter";
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
      const status = slot.lookupEvent.status;
      const done = status ? status.processedRecords : 0;
      const total = status ? status.recordCount : 0;
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

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function bigintHex(value: bigint, byteLength: number): string {
  return value.toString(16).padStart(byteLength * 2, "0");
}

export function renderResults(panel: HTMLElement, body: HTMLElement, snapshot: RunSnapshot): void {
  if (snapshot.status !== "done" && snapshot.status !== "no-match" && snapshot.status !== "error") {
    panel.hidden = true;
    body.innerHTML = "";
    return;
  }
  panel.hidden = false;

  if (snapshot.status === "error") {
    body.innerHTML = `<p class="nr-no-match">Run failed: ${snapshot.errorMessage ?? "unknown error"}</p>`;
    return;
  }
  if (snapshot.status === "no-match" || !snapshot.result) {
    body.innerHTML = `<p class="nr-no-match">No match found in this table.</p>`;
    return;
  }

  const result = snapshot.result;
  const parts: string[] = [];
  parts.push(
    `<dl class="nr-results-role"><dt>DES1 key(s)</dt>${result.des1Keys
      .map((key) => `<dd>${bigintHex(key, 7)}</dd>`)
      .join("")}</dl>`,
  );
  if (result.des2Keys.length > 0) {
    parts.push(
      `<dl class="nr-results-role"><dt>DES2 key(s)</dt>${result.des2Keys
        .map((key) => `<dd>${bigintHex(key, 7)}</dd>`)
        .join("")}</dl>`,
    );
  }
  if (result.pt3 !== null) {
    parts.push(`<dl class="nr-results-role"><dt>DES3 plaintext</dt><dd>${hex(result.pt3)}</dd></dl>`);
  }
  if (result.ntHashes.length > 0) {
    parts.push(
      `<dl class="nr-results-role"><dt>Assembled NT hash(es)</dt>${result.ntHashes
        .map((entry) => `<dd>${hex(entry.ntHash)}</dd>`)
        .join("")}</dl>`,
    );
  }
  body.innerHTML = parts.join("");
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
    const when = new Date(summary.created_at).toLocaleString();
    const hashLabel = summary.matched
      ? `<span class="nr-archive-run-hash">${summary.nt_hash_hex ?? "matched"}</span>`
      : "no match";
    item.innerHTML = `<div>${when} · ${summary.input_mode}</div><div>${hashLabel}</div>`;
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
