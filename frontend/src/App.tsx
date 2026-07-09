import { useEffect, useMemo, useRef, useState } from "react";
import ModelPicker from "./components/ModelPicker";
import MergePanel from "./components/MergePanel";
import TensorLedger from "./components/TensorLedger";
import Toast from "./components/Toast";
import { fetchCatalog, fetchCompat, fetchLocalModels, startMerge, subscribeJobEvents, fetchJob } from "./lib/api";
import type { CatalogEntry, CompatResponse, LocalModelInfo, MergeReport } from "./types/api";

type Slot = "A" | "B";

export default function App() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Which model is currently chosen in each dropdown (independent of
  // whether it's been downloaded yet), plus any custom repo id typed in.
  const [selectedA, setSelectedA] = useState("");
  const [selectedB, setSelectedB] = useState("");
  const [customA, setCustomA] = useState("");
  const [customB, setCustomB] = useState("");

  // Which slot's selection is currently *gating* the other slot's dropdown.
  // null until the person makes their first explicit choice - both sides
  // show the full live catalog until then.
  const [gatingSlot, setGatingSlot] = useState<Slot | null>(null);

  const [compat, setCompat] = useState<CompatResponse | null>(null);
  const [compatLoading, setCompatLoading] = useState(false);
  const [compatError, setCompatError] = useState<string | null>(null);

  const [downloadedA, setDownloadedA] = useState<string | null>(null);
  const [downloadedB, setDownloadedB] = useState<string | null>(null);

  const [swapRatio, setSwapRatio] = useState(0.3);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1_000_000));

  const [merging, setMerging] = useState(false);
  const [mergeJobId, setMergeJobId] = useState<string | null>(null);
  const [mergeLog, setMergeLog] = useState<string[]>([]);
  const [mergeReport, setMergeReport] = useState<MergeReport | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // See refreshLocalModels below: guards against a late-arriving response
  // from an older /api/models/local request clobbering a newer one.
  const latestLocalModelsRequestId = useRef(0);
  // Same idea for the live compatibility check: guards against a stale
  // /api/models/compat response (for a base model the person has since
  // changed away from) landing after a newer request and clobbering it.
  const latestCompatRequestId = useRef(0);

  useEffect(() => {
    fetchCatalog()
      .then((entries) => {
        setCatalog(entries);
        // Backfill both dropdowns to the top catalog entry once it loads.
        // This is a *default*, not a user choice, so it deliberately does
        // NOT set gatingSlot - both sides stay showing the full catalog
        // until the person actually picks something.
        setSelectedA((cur) => cur || entries[0]?.id || "");
        setSelectedB((cur) => cur || entries[0]?.id || "");
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
    refreshLocalModels();
  }, []);

  function refreshLocalModels() {
    const requestId = ++latestLocalModelsRequestId.current;
    fetchLocalModels()
      .then((models) => {
        if (requestId === latestLocalModelsRequestId.current) {
          setLocalModels(models);
        }
      })
      .catch(() => {
        /* non-fatal: local model list is informational only */
      });
  }

  function isDownloaded(id: string | null): boolean {
    if (!id) return false;
    if (!Array.isArray(localModels)) return false;
    return localModels.some((m) => m.id === id && m.hasSafeTensor);
  }

  // Explicit, user-driven selection in one slot's dropdown. This is what
  // switches on live gating of the *other* slot - picking a model in the
  // catalog dropdown is a strong enough signal of intent to start filtering
  // the other side down to real, verified-compatible options.
  function handleSelect(slot: Slot, id: string) {
    if (slot === "A") setSelectedA(id);
    else setSelectedB(id);
    setGatingSlot(slot);
  }

  const baseSlot = gatingSlot;
  const otherSlot: Slot | null = gatingSlot === "A" ? "B" : gatingSlot === "B" ? "A" : null;
  const baseId = baseSlot === "A" ? selectedA : baseSlot === "B" ? selectedB : null;

  // Fetch (or re-fetch) the live compatibility check whenever the gating
  // model changes.
  useEffect(() => {
    if (!baseId) {
      setCompat(null);
      setCompatError(null);
      setCompatLoading(false);
      return;
    }
    const requestId = ++latestCompatRequestId.current;
    setCompatLoading(true);
    setCompatError(null);
    fetchCompat(baseId)
      .then((res) => {
        if (requestId === latestCompatRequestId.current) {
          setCompat(res);
        }
      })
      .catch((err) => {
        if (requestId === latestCompatRequestId.current) {
          setCompatError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (requestId === latestCompatRequestId.current) {
          setCompatLoading(false);
        }
      });
  }, [baseId]);

  const compatMatchesBase = !!compat && compat.base === baseId;

  const compatById = useMemo(() => {
    const map = new Map<string, CompatResponse["results"][number]>();
    if (compatMatchesBase) {
      for (const r of compat!.results) map.set(r.id, r);
    }
    return map;
  }, [compat, compatMatchesBase]);

  const allowedIds = useMemo(() => {
    if (!compatMatchesBase) return null;
    const set = new Set<string>();
    for (const r of compat!.results) {
      if (r.compatible) set.add(r.id);
    }
    return set;
  }, [compat, compatMatchesBase]);

  // If the other slot's current selection just fell outside the allowed
  // set (because the gating model changed), auto-jump it to the first
  // still-allowed catalog entry so the UI never sits on a stale, now-
  // invalid choice.
  useEffect(() => {
    if (!otherSlot || !allowedIds || allowedIds.size === 0) return;
    const currentOtherId = otherSlot === "A" ? selectedA : selectedB;
    if (currentOtherId && allowedIds.has(currentOtherId)) return;
    const firstAllowed = catalog.find((c) => allowedIds.has(c.id));
    if (!firstAllowed) return;
    if (otherSlot === "A") setSelectedA(firstAllowed.id);
    else setSelectedB(firstAllowed.id);
  }, [allowedIds, otherSlot, selectedA, selectedB, catalog]);

  const optionsFor = (slot: Slot): CatalogEntry[] => {
    if (otherSlot !== slot || !allowedIds) return catalog;
    return catalog.filter((c) => allowedIds.has(c.id));
  };

  const restrictionFor = (slot: Slot) => {
    if (otherSlot !== slot || !baseId) return null;
    const baseLabel = catalog.find((c) => c.id === baseId)?.label ?? baseId;
    return {
      baseId,
      baseLabel,
      loading: compatLoading,
      error: compatError,
      compatibleCount: allowedIds?.size ?? 0,
      totalCount: catalog.length,
    };
  };

  const selectedIdFor = (slot: Slot) => (slot === "A" ? selectedA : selectedB);
  const compatEntryForSelection = (slot: Slot) => {
    if (otherSlot !== slot) return null;
    const id = selectedIdFor(slot);
    return compatById.get(id) ?? null;
  };

  async function handleMerge() {
    if (!downloadedA || !downloadedB) return;
    setMerging(true);
    setMergeReport(null);
    setMergeLog([`Starting merge: ${downloadedA} + ${downloadedB} (ratio ${Math.round(swapRatio * 100)}%, seed ${seed})`]);

    try {
      const { jobId } = await startMerge(downloadedA, downloadedB, swapRatio, seed);
      setMergeJobId(jobId);
      subscribeJobEvents(
        jobId,
        (line) => setMergeLog((prev) => [...prev, line]),
        async (finalStatus) => {
          setMerging(false);
          if (finalStatus === "succeeded") {
            try {
              const job = await fetchJob<MergeReport>(jobId);
              setMergeReport(job.result ?? null);
            } catch (err) {
              setErrorMessage(err instanceof Error ? err.message : String(err));
            }
          } else {
            try {
              const job = await fetchJob(jobId);
              setErrorMessage(job.error || "Merge failed for an unknown reason.");
            } catch {
              setErrorMessage("Merge failed for an unknown reason.");
            }
          }
        }
      );
    } catch (err) {
      setMerging(false);
      setErrorMessage(err instanceof Error ? err.message : String(err));
    }
  }

  const bothDownloaded = isDownloaded(downloadedA) && isDownloaded(downloadedB) && !!downloadedA && !!downloadedB;

  // Final, authoritative gate on the exact downloaded pair. In normal use
  // this just confirms what the restricted dropdown already guaranteed, but
  // it also covers the "typed a custom repo id" escape hatch, where no
  // dropdown filtering ever applied.
  let mergeStatus: { kind: "ok" | "warning" | "info"; text: string } | null = null;
  let mergeBlocked = false;
  if (bothDownloaded && baseSlot && otherSlot) {
    const otherDownloadedId = otherSlot === "A" ? downloadedA : downloadedB;
    const entry = compatMatchesBase ? compatById.get(otherDownloadedId!) : undefined;
    if (entry) {
      if (entry.compatible) {
        mergeStatus = { kind: "ok", text: `✅ ${entry.reason}` };
      } else {
        mergeBlocked = true;
        mergeStatus = { kind: "warning", text: `⚠️ ${entry.reason}` };
      }
    } else {
      mergeStatus = {
        kind: "info",
        text: "ℹ️ This exact pair hasn't been live-checked yet (likely a custom repo id) — the merge will fail safely if the tensors don't actually match.",
      };
    }
  }

  const canMerge = bothDownloaded && !mergeBlocked;

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">🧬</div>
          <div>
            <h1>Splice</h1>
            <p>Download two HuggingFace models, then randomly recombine their weights.</p>
          </div>
        </div>
        <div className="header-meta">
          <div>
            <span className="dot" /> backend online
          </div>
          <div>go agent · safetensors engine</div>
        </div>
      </header>

      <div className="disclosure">
        <strong>How this works:</strong> Splice pulls its model list live from the HuggingFace Hub, downloads real
        public safetensors checkpoints, then runs a simplified, educational stand-in for <em>mergekit</em>: for every
        tensor the two models share (same name, shape, and dtype), it flips a weighted coin and keeps either the
        Specimen A or Specimen B version. Compatibility between models is verified live, by inspecting actual tensor
        shapes &mdash; not guessed from an architecture-family label, since same-family models (e.g. TinyStories 1M
        vs 33M) can still have completely different tensor widths. This is <strong>not</strong> a real merge algorithm
        like SLERP, TIES, or DARE &mdash; it's a fast, visual way to see model weights being downloaded, inspected,
        and recombined end to end.
      </div>

      {loadError && (
        <div className="disclosure" style={{ borderLeftColor: "var(--accent-danger)" }}>
          Could not reach the backend API: {loadError}. Make sure the backend container is running.
        </div>
      )}

      <div className="specimen-row">
        <ModelPicker
          slot="A"
          options={optionsFor("A")}
          fullCatalogSize={catalog.length}
          selectedId={selectedA}
          onSelect={(id) => handleSelect("A", id)}
          customId={customA}
          onCustomIdChange={setCustomA}
          isDownloaded={isDownloaded(downloadedA)}
          onDownloaded={(id) => {
            setDownloadedA(id);
            refreshLocalModels();
          }}
          onError={setErrorMessage}
          restriction={restrictionFor("A")}
          compatEntry={compatEntryForSelection("A")}
        />
        <ModelPicker
          slot="B"
          options={optionsFor("B")}
          fullCatalogSize={catalog.length}
          selectedId={selectedB}
          onSelect={(id) => handleSelect("B", id)}
          customId={customB}
          onCustomIdChange={setCustomB}
          isDownloaded={isDownloaded(downloadedB)}
          onDownloaded={(id) => {
            setDownloadedB(id);
            refreshLocalModels();
          }}
          onError={setErrorMessage}
          restriction={restrictionFor("B")}
          compatEntry={compatEntryForSelection("B")}
        />
      </div>

      <MergePanel
        swapRatio={swapRatio}
        onSwapRatioChange={setSwapRatio}
        seed={seed}
        onSeedChange={setSeed}
        onRandomizeSeed={() => setSeed(Math.floor(Math.random() * 1_000_000))}
        onMerge={handleMerge}
        canMerge={canMerge}
        merging={merging}
        status={mergeStatus}
      />

      <TensorLedger report={mergeReport} jobId={mergeJobId} mergeLog={mergeLog} />

      <footer className="app-footer">
        hf-mergekit-demo &middot; TypeScript + React frontend &middot; Go backend &middot; for education, not production merges
      </footer>

      {errorMessage && <Toast message={errorMessage} onDismiss={() => setErrorMessage(null)} />}
    </div>
  );
}
