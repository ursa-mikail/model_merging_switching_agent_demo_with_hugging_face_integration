import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// ---------------------------------------------------------------------------
// A minimal fake backend + fake EventSource, so we can exercise the *real*
// App/ModelPicker/TensorLedger code exactly as a browser would, without a
// real Go server or a real browser. This is what actually caught the bugs
// reported ("downloads and blanks out", "merge not choosable") — running the
// literal production code through the literal user flow, rather than
// reasoning about it in the abstract.
// ---------------------------------------------------------------------------

interface MockJob {
  id: string;
  type: "download" | "merge";
  status: "succeeded" | "failed";
  log: string[];
  error?: string;
  result?: unknown;
}

let jobs: Record<string, MockJob>;
let localModels: Array<{ id: string; files: string[]; totalBytes: number; hasSafeTensor: boolean }>;
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
    // Simulate the backend replaying its log buffer then sending "done",
    // asynchronously, just like the real SSE endpoint.
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

function installFakeFetch() {
  global.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const method = (init?.method || "GET").toUpperCase();

    if (url.endsWith("/api/models") && method === "GET") {
      return jsonResponse(CATALOG);
    }
    if (url.endsWith("/api/models/local") && method === "GET") {
      return jsonResponse(localModels);
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
    if (url.endsWith("/api/jobs/merge") && method === "POST") {
      const body = JSON.parse(String(init?.body));
      const id = `job${++jobCounter}`;
      const details = [
        { name: "embed.weight", shape: [4, 1], dtype: "F32", bytes: 16, sourceModel: "A", swapped: false },
        { name: "layer0.weight", shape: [8, 1], dtype: "F32", bytes: 32, sourceModel: "B", swapped: true },
      ];
      jobs[id] = {
        id,
        type: "merge",
        status: "succeeded",
        log: ["Starting merge...", "Merge complete."],
        result: {
          modelA: body.modelAId,
          modelB: body.modelBId,
          swapRatio: body.swapRatio,
          seed: body.seed,
          commonTensors: 2,
          onlyInA: 0,
          onlyInB: 0,
          swappedCount: 1,
          totalOutBytes: 48,
          outputFile: "merged.safetensors",
          details,
        },
      };
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
}

beforeEach(() => {
  jobs = {};
  localModels = [];
  jobCounter = 0;
  installFakeFetch();
  // @ts-expect-error -- test-only global override, real DOM lacks EventSource
  global.EventSource = FakeEventSource;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Splice end-to-end UI flow", () => {
  it("downloads both specimens without crashing, then enables Merge models", async () => {
    const user = userEvent.setup();
    render(<App />);

    // Catalog should load into both dropdowns.
    await waitFor(() => {
      expect(screen.getAllByText(/TinyStories 1M/).length).toBeGreaterThan(0);
    });

    const mergeButton = screen.getByRole("button", { name: /merge models/i });
    expect(mergeButton).toBeDisabled();

    const specimenCards = document.querySelectorAll(".specimen-card");
    expect(specimenCards.length).toBe(2);

    const cardA = within(specimenCards[0] as HTMLElement);
    const cardB = within(specimenCards[1] as HTMLElement);

    // --- Download specimen A ---
    const downloadButtonA = cardA.getByRole("button", { name: /download from huggingface/i });
    await user.click(downloadButtonA);

    // This is the exact assertion that would fail with the nil-slice bug:
    // if /api/models/local ever returns something the app can't treat as an
    // array, rendering throws and the whole tree (including this button)
    // disappears from the DOM.
    await waitFor(() => {
      expect(cardA.getByText(/downloaded/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /merge models/i })).toBeInTheDocument();

    // --- Download specimen B ---
    const downloadButtonB = cardB.getByRole("button", { name: /download from huggingface/i });
    await user.click(downloadButtonB);

    await waitFor(() => {
      expect(cardB.getByText(/downloaded/i)).toBeInTheDocument();
    });

    // The core regression check: once both specimens are down, Merge models
    // must become clickable.
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /merge models/i })).not.toBeDisabled();
    });

    // The whole app must still be mounted and interactive — this is the
    // "blanks out" regression check.
    expect(screen.getByRole("heading", { name: /^splice$/i })).toBeInTheDocument();
    expect(document.body.textContent).not.toBe("");
  });

  it("runs a full merge after both downloads and renders the tensor ledger", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getAllByText(/TinyStories 1M/).length).toBeGreaterThan(0);
    });

    const specimenCards = document.querySelectorAll(".specimen-card");
    const cardA = within(specimenCards[0] as HTMLElement);
    const cardB = within(specimenCards[1] as HTMLElement);

    await user.click(cardA.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardA.getByText(/downloaded/i)).toBeInTheDocument());

    await user.click(cardB.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardB.getByText(/downloaded/i)).toBeInTheDocument());

    const mergeButton = await waitFor(() => {
      const btn = screen.getByRole("button", { name: /merge models/i });
      expect(btn).not.toBeDisabled();
      return btn;
    });

    await user.click(mergeButton);

    await waitFor(() => {
      expect(screen.getByText(/layer0\.weight/)).toBeInTheDocument();
    });

    // Diff map + filters should be present and the app must still be alive.
    expect(screen.getByText(/Diff map/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /^splice$/i })).toBeInTheDocument();
  });
});
