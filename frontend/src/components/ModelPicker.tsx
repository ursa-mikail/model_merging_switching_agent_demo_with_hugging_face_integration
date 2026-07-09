import { useEffect, useRef, useState } from "react";
import type { CatalogEntry, CompatEntry, JobStatus } from "../types/api";
import { startDownload, subscribeJobEvents, fetchJob } from "../lib/api";

/** Describes why (and how) this slot's dropdown is currently being
 * restricted to models compatible with the *other* slot's selection. */
interface Restriction {
  baseId: string;
  baseLabel: string;
  loading: boolean;
  error: string | null;
  compatibleCount: number;
  totalCount: number;
}

interface Props {
  slot: "A" | "B";
  /** The options to actually render in the dropdown - already filtered by
   * the parent when `restriction` is active. */
  options: CatalogEntry[];
  /** Size of the full, unfiltered catalog (used in the "3 of 58" copy). */
  fullCatalogSize: number;
  selectedId: string;
  /** Fired only for explicit, user-driven dropdown changes. */
  onSelect: (id: string) => void;
  customId: string;
  onCustomIdChange: (value: string) => void;
  isDownloaded: boolean;
  onDownloaded: (modelId: string) => void;
  onError: (message: string) => void;
  restriction: Restriction | null;
  /** Live compatibility result for the currently selected catalog model,
   * against the other slot's base model - only present when `restriction`
   * is active and the check has completed. */
  compatEntry: CompatEntry | null;
}

export default function ModelPicker({
  slot,
  options,
  fullCatalogSize,
  selectedId,
  onSelect,
  customId,
  onCustomIdChange,
  isDownloaded,
  onDownloaded,
  onError,
  restriction,
  compatEntry,
}: Props) {
  const [status, setStatus] = useState<JobStatus | "idle">("idle");
  const [log, setLog] = useState<string[]>([]);
  const consoleRef = useRef<HTMLDivElement>(null);
  const unsubscribeRef = useRef<() => void>();

  useEffect(() => {
    return () => unsubscribeRef.current?.();
  }, []);

  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [log]);

  const effectiveModelId = customId.trim() || selectedId;
  const selectedEntry = options.find((c) => c.id === selectedId);
  const busy = status === "running" || status === "pending";
  const usingCustomId = customId.trim().length > 0;

  async function handleDownload() {
    const modelId = effectiveModelId;
    if (!modelId) {
      onError("Pick a model from the dropdown or type a HuggingFace repo id.");
      return;
    }
    setStatus("pending");
    setLog([`Requesting download of ${modelId}...`]);

    try {
      const { jobId } = await startDownload(modelId);
      setStatus("running");
      unsubscribeRef.current?.();
      unsubscribeRef.current = subscribeJobEvents(
        jobId,
        (line) => setLog((prev) => [...prev, line]),
        async (finalStatus) => {
          if (finalStatus === "succeeded") {
            setStatus("succeeded");
            onDownloaded(modelId);
          } else {
            setStatus("failed");
            try {
              const job = await fetchJob(jobId);
              onError(job.error || "Download failed for an unknown reason.");
            } catch {
              onError(`Download of ${modelId} failed.`);
            }
          }
        }
      );
    } catch (err) {
      setStatus("failed");
      const message = err instanceof Error ? err.message : String(err);
      setLog((prev) => [...prev, `ERROR: ${message}`]);
      onError(message);
    }
  }

  const dotClass =
    status === "running" || status === "pending"
      ? "running"
      : status === "succeeded"
      ? "done"
      : status === "failed"
      ? "error"
      : "idle";

  const statusText =
    status === "idle"
      ? isDownloaded
        ? "ready (previously downloaded)"
        : "not downloaded yet"
      : status === "pending"
      ? "queuing job..."
      : status === "running"
      ? "downloading..."
      : status === "succeeded"
      ? "downloaded"
      : "failed";

  return (
    <div className={`specimen-card slot-${slot.toLowerCase()}`}>
      <div className="specimen-label">
        <span className="chip">{slot}</span>
        Specimen {slot}
      </div>

      {restriction && (
        <div className="compat-banner">
          {restriction.loading ? (
            <span className="compat-checking">
              🔬 Checking live compatibility against Specimen {slot === "A" ? "B" : "A"} (
              {restriction.baseLabel})… fetching real tensor shapes, not guessing from a family label.
            </span>
          ) : restriction.error ? (
            <span className="compat-bad">
              ⚠️ Live compatibility check failed ({restriction.error}). Showing the full catalog — compatibility
              will still be verified before merging.
            </span>
          ) : restriction.compatibleCount === 0 ? (
            <span className="compat-bad">
              🔬 No cataloged model shares matching tensor shapes with <strong>{restriction.baseLabel}</strong>{" "}
              (live check, 0 of {restriction.totalCount}). You can still type a custom repo id below — it'll be
              verified for real before merging.
            </span>
          ) : (
            <span className="compat-good">
              🔬 Showing {restriction.compatibleCount} of {restriction.totalCount} catalog models with tensor
              shapes confirmed to match <strong>{restriction.baseLabel}</strong> (live check, not a family guess).
            </span>
          )}
        </div>
      )}

      <div className="field-group">
        <label className="field-label" htmlFor={`select-${slot}`}>
          Choose a HuggingFace model
        </label>
        <select
          id={`select-${slot}`}
          value={options.some((o) => o.id === selectedId) ? selectedId : ""}
          onChange={(e) => onSelect(e.target.value)}
          disabled={busy || (restriction != null && !restriction.loading && options.length === 0)}
        >
          {options.length === 0 && (
            <option value="" disabled>
              {restriction?.loading ? "checking compatibility…" : "no compatible models found"}
            </option>
          )}
          {options.map((entry) => (
            <option key={entry.id} value={entry.id}>
              {entry.label} · {entry.approxSize}
            </option>
          ))}
        </select>
      </div>

      {restriction && !usingCustomId && compatEntry && (
        <div className={`compat-reason ${compatEntry.compatible ? "compat-good" : "compat-bad"}`}>
          {compatEntry.compatible ? "✅" : "⚠️"} {compatEntry.reason}
        </div>
      )}

      <div className="or-divider">or type any public repo id</div>

      <div className="field-group">
        <input
          type="text"
          placeholder="e.g. facebook/opt-125m"
          value={customId}
          onChange={(e) => onCustomIdChange(e.target.value)}
          disabled={busy}
        />
      </div>

      <div className="model-meta">
        {usingCustomId ? (
          <span>
            Using custom repo id — make sure it publishes .safetensors weights. Compatibility bypasses the catalog
            filter here, but will still be checked for real before merging.
          </span>
        ) : selectedEntry ? (
          <>
            <span className="size-badge">{selectedEntry.approxSize}</span>
            <span className="size-badge">{selectedEntry.family}</span>
            <br />
            {selectedEntry.description}
          </>
        ) : fullCatalogSize === 0 ? (
          <span>Loading the live catalog from HuggingFace…</span>
        ) : null}
      </div>

      <button className="primary-btn" onClick={handleDownload} disabled={busy}>
        {busy ? "Downloading..." : isDownloaded ? "Re-download" : "Download from HuggingFace"}
      </button>

      <div className="status-line">
        <span className={`status-dot ${dotClass}`} />
        <span>{statusText}</span>
      </div>

      <div className="console" ref={consoleRef}>
        {log.map((line, i) => (
          <div key={i} className={line.startsWith("ERROR") ? "log-error" : undefined}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
