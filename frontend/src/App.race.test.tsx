import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// ---------------------------------------------------------------------------
// Regression test for the "Merge models never becomes choosable" race.
//
// App.refreshLocalModels() is called once per completed download, and each
// call fires an independent, un-awaited fetch("/api/models/local"). Nothing
// guarantees those requests resolve in the order they were sent. This test
// deliberately resolves them *out of order* - the request kicked off after
// Specimen A downloads (which only reflects "A is on disk" at the moment it
// was sent) is held open and released *after* the later request kicked off
// after Specimen B downloads (which reflects "A and B are on disk"). A fix
// that merely relies on fetches resolving in send order (like the original
// bug, and like a naive fake-fetch test with instant/in-order resolution)
// will fail this test: the stale A-only response arrives last and wins,
// dropping B from state and re-locking the Merge models button.
// ---------------------------------------------------------------------------

interface MockJob {
  id: string;
  type: "download" | "merge";
  status: "succeeded" | "failed";
  log: string[];
  error?: string;
  result?: unknown;
}

type LocalModel = { id: string; files: string[]; totalBytes: number; hasSafeTensor: boolean };

let jobs: Record<string, MockJob>;
let localModels: LocalModel[];
let jobCounter = 0;

const CATALOG = [
  { id: "roneneldan/TinyStories-1M", label: "TinyStories 1M", description: "d", approxSize: "~4 MB", family: "GPT-Neo" },
  { id: "roneneldan/TinyStories-8M", label: "TinyStories 8M", description: "d", approxSize: "~30 MB", family: "GPT-Neo" },
];

class FakeEventSource {
  static all: FakeEventSource[] = [];
  url: string;
  onmessage: ((evt: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners: Record<string, Array<(evt: { data: string }) => void>> = {};

  constructor(url: string) {
    this.url = url;
    FakeEventSource.all.push(this);
    setTimeout(() => this.emit(), 0);
  }

  addEventListener(type: string, cb: (evt: { data: string }) => void) {
    (this.listeners[type] ||= []).push(cb);
  }

  close() {
    /* no-op */
  }

  private emit() {
    const match = this.url.match(/\/api\/jobs\/([^/]+)\/events/);
    const jobId = match?.[1];
    if (!jobId) return;
    const job = jobs[jobId];
    if (!job) return;
    for (const line of job.log) {
      this.onmessage?.({ data: line });
    }
    for (const cb of this.listeners["done"] || []) {
      cb({ data: job.status });
    }
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * A deferred /api/models/local call: captures a snapshot of `localModels`
 * at the moment the request was *sent* (mirroring how the real backend
 * would answer based on disk state at request time), but doesn't resolve
 * until the test explicitly releases it. This lets the test control
 * resolution order independently of send order.
 */
interface PendingLocalModelsCall {
  snapshot: LocalModel[];
  release: () => void;
}

function installRaceyFetch(): PendingLocalModelsCall[] {
  const pendingLocalModelsCalls: PendingLocalModelsCall[] = [];

  global.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const method = (init?.method || "GET").toUpperCase();

    if (url.endsWith("/api/models") && method === "GET") {
      return jsonResponse(CATALOG);
    }

    if (url.endsWith("/api/models/local") && method === "GET") {
      const snapshot = [...localModels];
      return new Promise<Response>((resolve) => {
        pendingLocalModelsCalls.push({
          snapshot,
          release: () => resolve(jsonResponse(snapshot)),
        });
      });
    }

    if (url.endsWith("/api/jobs/download") && method === "POST") {
      const body = JSON.parse(String(init?.body));
      const id = `job${++jobCounter}`;
      jobs[id] = {
        id,
        type: "download",
        status: "succeeded",
        log: [`Downloading ${body.modelId}...`, "Download complete."],
      };
      localModels.push({
        id: body.modelId,
        files: ["model.safetensors", "config.json"],
        totalBytes: 1234,
        hasSafeTensor: true,
      });
      return jsonResponse({ jobId: id }, 202);
    }

    const jobMatch = url.match(/\/api\/jobs\/([^/]+)$/);
    if (jobMatch && method === "GET") {
      const job = jobs[jobMatch[1]];
      if (!job) return jsonResponse({ error: "not found" }, 404);
      return jsonResponse({
        id: job.id,
        type: job.type,
        status: job.status,
        log: job.log,
        error: job.error,
        result: job.result,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
    }

    throw new Error(`Unhandled fetch in test: ${method} ${url}`);
  }) as unknown as typeof fetch;

  return pendingLocalModelsCalls;
}

beforeEach(() => {
  jobs = {};
  localModels = [];
  jobCounter = 0;
  // @ts-expect-error -- test-only global override, real DOM lacks EventSource
  global.EventSource = FakeEventSource;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Splice local-model refresh race condition", () => {
  it("keeps both specimens in state even when the A-refresh resolves after the B-refresh", async () => {
    const calls = installRaceyFetch();
    const user = userEvent.setup();
    render(<App />);

    // Initial mount kicks off refreshLocalModels() once, before any
    // downloads. Release it with an empty snapshot so the app finishes
    // its first render normally.
    await waitFor(() => expect(calls.length).toBe(1));
    calls[0].release();

    await waitFor(() => {
      expect(screen.getAllByText(/TinyStories 1M/).length).toBeGreaterThan(0);
    });

    const specimenCards = document.querySelectorAll(".specimen-card");
    const cardA = within(specimenCards[0] as HTMLElement);
    const cardB = within(specimenCards[1] as HTMLElement);

    // --- Download Specimen A: this fires refresh call #2, snapshot = [A] ---
    await user.click(cardA.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardA.getByText(/downloaded/i)).toBeInTheDocument());
    await waitFor(() => expect(calls.length).toBe(2));
    expect(calls[1].snapshot.map((m) => m.id)).toEqual(["roneneldan/TinyStories-1M"]);

    // Both dropdowns default to the same first catalog entry, so pick the
    // second model for Specimen B - otherwise "downloading B" would just
    // re-download A under a different slot and never exercise two distinct
    // ids landing in local models.
    await user.selectOptions(
      cardB.getByLabelText(/choose a huggingface model/i),
      "roneneldan/TinyStories-8M"
    );

    // --- Download Specimen B: this fires refresh call #3, snapshot = [A, B] ---
    await user.click(cardB.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardB.getByText(/downloaded/i)).toBeInTheDocument());
    await waitFor(() => expect(calls.length).toBe(3));
    expect(calls[2].snapshot.map((m) => m.id)).toEqual([
      "roneneldan/TinyStories-1M",
      "roneneldan/TinyStories-8M",
    ]);

    // Now resolve OUT OF ORDER: the newer [A, B] response lands first, then
    // the older, now-stale [A]-only response lands second. A correct
    // implementation must not let the stale response overwrite the fresher
    // state that already reflects both downloads.
    calls[2].release();
    calls[1].release();

    // The core regression check: Merge models must become enabled and, since
    // the stale response arrived last, must *stay* enabled rather than
    // flipping back to disabled once the late response is processed.
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /merge models/i })).not.toBeDisabled();
    });

    // Give any further microtasks a chance to run, then confirm the button
    // is still enabled - this is what would fail under the original bug,
    // where the late-arriving stale [A] response overwrites [A, B] and
    // isDownloaded(downloadedB) flips back to false.
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.getByRole("button", { name: /merge models/i })).not.toBeDisabled();
  });
});
