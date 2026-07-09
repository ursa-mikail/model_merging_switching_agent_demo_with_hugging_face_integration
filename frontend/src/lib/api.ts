import type {
  CatalogEntry,
  LocalModelInfo,
  JobRecord,
  StartJobResponse,
  ApiErrorBody,
  CompatResponse,
} from "../types/api";

// In production (docker-compose), nginx serves the frontend and proxies
// /api/* to the backend container, so a relative base works everywhere.
const BASE = "/api";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as ApiErrorBody;
      if (body?.error) message = body.error;
    } catch {
      // response wasn't JSON; keep the generic message
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function fetchCatalog(): Promise<CatalogEntry[]> {
  const res = await fetch(`${BASE}/models`);
  return asJson<CatalogEntry[]>(res);
}

export async function fetchLocalModels(): Promise<LocalModelInfo[]> {
  const res = await fetch(`${BASE}/models/local`);
  const data = await asJson<LocalModelInfo[] | null>(res);
  // Defensive: treat a null/non-array response (e.g. an empty Go slice that
  // serialized as JSON `null`) the same as an empty list, so callers can
  // always safely call array methods on the result.
  return Array.isArray(data) ? data : [];
}

/**
 * Live tensor-shape compatibility check: which other models can actually be
 * merged with `baseId`? This inspects real safetensors headers (locally if
 * already downloaded, otherwise via a small HTTP range request against the
 * Hub) rather than guessing from a cosmetic "family" label.
 */
export async function fetchCompat(baseId: string): Promise<CompatResponse> {
  const res = await fetch(`${BASE}/models/compat?base=${encodeURIComponent(baseId)}`);
  return asJson<CompatResponse>(res);
}

export async function startDownload(modelId: string): Promise<StartJobResponse> {
  const res = await fetch(`${BASE}/jobs/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelId }),
  });
  return asJson<StartJobResponse>(res);
}

export async function startMerge(
  modelAId: string,
  modelBId: string,
  swapRatio: number,
  seed: number
): Promise<StartJobResponse> {
  const res = await fetch(`${BASE}/jobs/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelAId, modelBId, swapRatio, seed }),
  });
  return asJson<StartJobResponse>(res);
}

export async function fetchJob<T = unknown>(jobId: string): Promise<JobRecord<T>> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  return asJson<JobRecord<T>>(res);
}

export function mergedDownloadUrl(jobId: string): string {
  return `${BASE}/merged/${jobId}/download`;
}

/**
 * Subscribes to a job's Server-Sent Events log stream. Calls onLine for
 * every log line (including ones that happened before subscribing, since the
 * backend replays its buffer), and onDone once the job finishes.
 * Returns an unsubscribe function.
 */
export function subscribeJobEvents(
  jobId: string,
  onLine: (line: string) => void,
  onDone: (status: string) => void
): () => void {
  const source = new EventSource(`${BASE}/jobs/${jobId}/events`);

  source.onmessage = (evt) => {
    onLine(evt.data);
  };
  source.addEventListener("done", (evt) => {
    const msgEvt = evt as MessageEvent<string>;
    onDone(msgEvt.data);
    source.close();
  });
  source.onerror = () => {
    // EventSource auto-retries; if the job already finished the server will
    // have sent "done" before closing, so an error here after that is harmless.
  };

  return () => source.close();
}
