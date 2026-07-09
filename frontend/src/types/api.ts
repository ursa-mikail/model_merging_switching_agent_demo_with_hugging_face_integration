// Types mirroring the JSON shapes produced by the Go backend.
// Keep these in sync with backend/internal/{models,merge,api}.

export interface CatalogEntry {
  id: string;
  label: string;
  description: string;
  approxSize: string;
  family: string;
}

export interface LocalModelInfo {
  id: string;
  files: string[];
  totalBytes: number;
  hasSafeTensor: boolean;
}

export type JobStatus = "pending" | "running" | "succeeded" | "failed";

// A single candidate's live compatibility result against some base model.
// Produced by GET /api/models/compat?base=<id>, which inspects real
// safetensors tensor shapes (locally if the model is already downloaded,
// otherwise via a small HTTP range request against the Hub) rather than
// guessing from a cosmetic "family" label.
export interface CompatEntry {
  id: string;
  compatible: boolean;
  checked: boolean;
  commonTensors: number;
  sharedNames: number;
  candidateTensorCount: number;
  reason: string;
  source: "local" | "remote" | "error";
}

export interface CompatResponse {
  base: string;
  baseTensorCount: number;
  baseSource: "local" | "remote";
  results: CompatEntry[];
}

export interface DownloadResult {
  repoId: string;
  dir: string;
  files: string[];
  totalBytes: number;
  safetensors: string[];
}

export interface TensorSwapDetail {
  name: string;
  shape: number[];
  dtype: string;
  bytes: number;
  sourceModel: "A" | "B";
  swapped: boolean;
}

export interface MergeReport {
  modelA: string;
  modelB: string;
  swapRatio: number;
  seed: number;
  commonTensors: number;
  onlyInA: number;
  onlyInB: number;
  swappedCount: number;
  totalOutBytes: number;
  outputFile: string;
  details: TensorSwapDetail[];
}

export interface JobRecord<TResult = unknown> {
  id: string;
  type: "download" | "merge";
  status: JobStatus;
  log: string[];
  error?: string;
  result?: TResult;
  createdAt: string;
  updatedAt: string;
}

export interface StartJobResponse {
  jobId: string;
}

export interface ApiErrorBody {
  error: string;
}
