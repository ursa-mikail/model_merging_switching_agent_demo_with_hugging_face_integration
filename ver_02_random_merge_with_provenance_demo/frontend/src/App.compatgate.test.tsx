import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// ---------------------------------------------------------------------------
// Regression test for "the app let you pick two models that could never be
// merged, and only found out after a doomed job failed."
//
// Root cause: the old UI grouped models by a cosmetic "family" label
// (GPT-Neo, GPT-2, ...) without checking real tensor shapes, so even
// same-family pairs (e.g. TinyStories-1M vs TinyStories-33M, different
// widths) could slip through and produce a zero-common-tensor merge.
//
// The fix: GET /api/models/compat inspects real tensor shapes live, and the
// frontend uses it to restrict the *other* dropdown to only confirmed-
// compatible options as soon as one model is chosen - with a visible
// explanation of why. This test asserts the doomed pairing is filtered out
// of the dropdown before it can ever be selected, and that the merge bar
// explains why once a real pair is verified compatible.
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
  { id: "roneneldan/TinyStories-1M", label: "TinyStories 1M", description: "d", approxSize: "~4 MB", family: "text-generation" },
  { id: "roneneldan/TinyStories-8M", label: "TinyStories 8M", description: "d", approxSize: "~30 MB", family: "text-generation" },
  { id: "sshleifer/tiny-gpt2", label: "Tiny GPT-2", description: "d", approxSize: "~2 MB", family: "text-generation" },
];

// Mirrors the real backend's tensor-shape logic well enough for this test:
// the two TinyStories checkpoints are treated as sharing matching tensors,
// tiny-gpt2 shares none with either of them.
function tinyStoriesBucket(id: string): "tinystories" | "other" {
  return id.includes("TinyStories") ? "tinystories" : "other";
}

function compatEntryFor(base: string, candidate: string) {
  const same = tinyStoriesBucket(base) === tinyStoriesBucket(candidate);
  return {
    id: candidate,
    compatible: same,
    checked: true,
    commonTensors: same ? 12 : 0,
    sharedNames: same ? 12 : 0,
    candidateTensorCount: 12,
    reason: same
      ? "All 12 shared tensors match exactly in shape and dtype — safe to merge."
      : "No tensor names in common at all — these are different model architectures.",
    source: "local",
  };
}

class FakeEventSource {
  url: string;
  onmessage: ((evt: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners: Record<string, Array<(evt: { data: string }) => void>> = {};

  constructor(url: string) {
    this.url = url;
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
    const rawUrl = typeof input === "string" ? input : input.toString();
    const method = (init?.method || "GET").toUpperCase();
    const url = new URL(rawUrl, "http://localhost");

    if (url.pathname === "/api/models" && method === "GET") return jsonResponse(CATALOG);
    if (url.pathname === "/api/models/local" && method === "GET") return jsonResponse(localModels);

    if (url.pathname === "/api/models/compat" && method === "GET") {
      const base = url.searchParams.get("base") || "";
      const candidateIds = new Set<string>([...CATALOG.map((c) => c.id), ...localModels.map((m) => m.id)]);
      candidateIds.delete(base);
      const results = [...candidateIds].map((id) => compatEntryFor(base, id));
      return jsonResponse({ base, baseTensorCount: 12, baseSource: "local", results });
    }

    if (url.pathname === "/api/jobs/download" && method === "POST") {
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

    if (url.pathname === "/api/jobs/merge" && method === "POST") {
      const body = JSON.parse(String(init?.body));
      const id = `job${++jobCounter}`;
      const same = tinyStoriesBucket(body.modelAId) === tinyStoriesBucket(body.modelBId);
      jobs[id] = same
        ? {
            id,
            type: "merge",
            status: "succeeded",
            log: ["Loading tensor indices...", "12 shared tensors found.", "Splicing complete."],
            result: {
              modelA: body.modelAId,
              modelB: body.modelBId,
              swapRatio: body.swapRatio,
              seed: body.seed,
              commonTensors: 12,
              onlyInA: 0,
              onlyInB: 0,
              swappedCount: 4,
              totalOutBytes: 1024,
              outputFile: "merged.safetensors",
              details: [],
            },
          }
        : {
            id,
            type: "merge",
            status: "failed",
            log: ["Loading tensor indices...", "Comparing tensors..."],
            error: "model A and model B share no tensors with matching name+shape+dtype",
          };
      return jsonResponse({ jobId: id }, 202);
    }

    const jobMatch = url.pathname.match(/\/api\/jobs\/([^/]+)$/);
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

    throw new Error(`Unhandled fetch in test: ${method} ${url.pathname}`);
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

describe("Splice live compatibility gating", () => {
  it("filters the other dropdown to compatible models (with an explanation) instead of allowing a doomed pairing", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getAllByText(/TinyStories 1M/).length).toBeGreaterThan(0);
    });

    const specimenCards = document.querySelectorAll(".specimen-card");
    const cardA = within(specimenCards[0] as HTMLElement);
    const cardB = within(specimenCards[1] as HTMLElement);

    // Explicitly choose Specimen A - this is what turns on gating of B.
    await user.selectOptions(cardA.getByLabelText(/choose a huggingface model/i), "roneneldan/TinyStories-1M");

    // Specimen B's dropdown should narrow to only the confirmed-compatible
    // option, and explain why via the live compat banner.
    await waitFor(() => {
      expect(cardB.getByText(/live check/i)).toBeInTheDocument();
    });
    const bSelect = cardB.getByLabelText(/choose a huggingface model/i) as HTMLSelectElement;
    const bOptionValues = Array.from(bSelect.options).map((o) => o.value);
    expect(bOptionValues).toContain("roneneldan/TinyStories-8M");
    expect(bOptionValues).not.toContain("sshleifer/tiny-gpt2");

    // Download both (B auto-selected the one compatible option).
    await user.click(cardA.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardA.getByText(/downloaded/i)).toBeInTheDocument());
    await user.click(cardB.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardB.getByText(/downloaded/i)).toBeInTheDocument());

    // The merge bar should show a positive, explained compatibility status
    // and an enabled button - the pair was verified live, not guessed.
    await waitFor(() => {
      expect(screen.getAllByText(/shared tensors match exactly/i).length).toBeGreaterThan(0);
    });
    const mergeButton = await waitFor(() => {
      const btn = screen.getByRole("button", { name: /merge models/i });
      expect(btn).not.toBeDisabled();
      return btn;
    });

    await user.click(mergeButton);

    await waitFor(() => {
      expect(screen.getByText(/tensor ledger/i)).toBeInTheDocument();
      expect(screen.getByText(/download merged\.safetensors/i)).toBeInTheDocument();
    });
  });

  it("still blocks the merge button if an incompatible pair is reached via custom repo ids", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getAllByText(/TinyStories 1M/).length).toBeGreaterThan(0);
    });

    const specimenCards = document.querySelectorAll(".specimen-card");
    const cardA = within(specimenCards[0] as HTMLElement);
    const cardB = within(specimenCards[1] as HTMLElement);

    await user.selectOptions(cardA.getByLabelText(/choose a huggingface model/i), "roneneldan/TinyStories-1M");
    await user.click(cardA.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardA.getByText(/downloaded/i)).toBeInTheDocument());

    // Bypass the filtered dropdown entirely with a custom, incompatible id.
    await user.type(cardB.getByPlaceholderText(/facebook\/opt-125m/i), "sshleifer/tiny-gpt2");
    await user.click(cardB.getByRole("button", { name: /download from huggingface/i }));
    await waitFor(() => expect(cardB.getByText(/downloaded/i)).toBeInTheDocument());

    // The final, authoritative live check on the exact downloaded pair
    // should still catch this and keep the merge button disabled.
    await waitFor(() => {
      expect(screen.getByText(/different model architectures/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /merge models/i })).toBeDisabled();
  });
});
