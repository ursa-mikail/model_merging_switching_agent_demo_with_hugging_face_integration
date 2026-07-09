import { useMemo, useState } from "react";
import type { MergeReport } from "../types/api";
import { mergedDownloadUrl } from "../lib/api";
import DiffMap from "./DiffMap";

interface Props {
  report: MergeReport | null;
  jobId: string | null;
  mergeLog: string[];
}

type Filter = "all" | "swapped" | "kept";

function humanBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let val = n / 1024;
  let i = 0;
  while (val >= 1024 && i < units.length - 1) {
    val /= 1024;
    i++;
  }
  return `${val.toFixed(2)} ${units[i]}`;
}

export default function TensorLedger({ report, jobId, mergeLog }: Props) {
  const [filter, setFilter] = useState<Filter>("all");

  const filteredDetails = useMemo(() => {
    if (!report) return [];
    if (filter === "swapped") return report.details.filter((d) => d.swapped);
    if (filter === "kept") return report.details.filter((d) => !d.swapped);
    return report.details;
  }, [report, filter]);

  return (
    <div className="results-section">
      <div className="results-header">
        <div>
          <h2>Tensor ledger</h2>
          <div className="sub">
            Every shared tensor's fate: kept from Specimen A, or spliced in from Specimen B.
          </div>
        </div>
      </div>

      {!report ? (
        <div className="empty-state">
          <div className="emoji">🧫</div>
          {mergeLog.length > 0 ? (
            <div className="console" style={{ textAlign: "left", maxWidth: 560, margin: "0 auto" }}>
              {mergeLog.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
          ) : (
            <p>Download two models and hit &ldquo;Merge models&rdquo; to see the splice results here.</p>
          )}
        </div>
      ) : (
        <>
          <div className="stat-strip">
            <div className="stat-cell accent-a">
              <div className="stat-label">Common tensors</div>
              <div className="stat-value">{report.commonTensors}</div>
            </div>
            <div className="stat-cell accent-b">
              <div className="stat-label">Swapped to B</div>
              <div className="stat-value">{report.swappedCount}</div>
            </div>
            <div className="stat-cell accent-merge">
              <div className="stat-label">Swap ratio</div>
              <div className="stat-value">{Math.round(report.swapRatio * 100)}%</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Output size</div>
              <div className="stat-value">{humanBytes(report.totalOutBytes)}</div>
            </div>
          </div>

          <div className="diffmap-wrap">
            <DiffMap details={report.details} />
          </div>

          <div className="ledger-toolbar">
            <div className="filter-tabs">
              <button
                type="button"
                className={`filter-tab ${filter === "all" ? "active" : ""}`}
                onClick={() => setFilter("all")}
              >
                All <span className="filter-count">{report.details.length}</span>
              </button>
              <button
                type="button"
                className={`filter-tab tab-b ${filter === "swapped" ? "active" : ""}`}
                onClick={() => setFilter("swapped")}
              >
                Spliced from B <span className="filter-count">{report.swappedCount}</span>
              </button>
              <button
                type="button"
                className={`filter-tab tab-a ${filter === "kept" ? "active" : ""}`}
                onClick={() => setFilter("kept")}
              >
                Kept from A <span className="filter-count">{report.details.length - report.swappedCount}</span>
              </button>
            </div>
          </div>

          <div className="ledger-wrap">
            <table className="ledger">
              <thead>
                <tr>
                  <th>Tensor</th>
                  <th>Shape</th>
                  <th>Dtype</th>
                  <th>Size</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {filteredDetails.map((d, i) => (
                  <tr
                    key={d.name}
                    className={d.swapped ? "row-swapped" : "row-kept"}
                    style={{ animationDelay: `${Math.min(i, 40) * 12}ms` }}
                  >
                    <td className="tensor-name">
                      {d.swapped && <span className="diff-marker">●</span>}
                      {d.name}
                    </td>
                    <td>[{d.shape.join(", ")}]</td>
                    <td>{d.dtype}</td>
                    <td>{humanBytes(d.bytes)}</td>
                    <td>
                      <span className={`source-badge source-${d.sourceModel.toLowerCase()}`}>
                        {d.sourceModel}
                        {d.swapped && <span className="swap-flag">SPLICED</span>}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="results-footer">
            <span className="report-note">
              seed={report.seed} · model A tensors reused: {report.commonTensors - report.swappedCount} · model-A-only
              tensors kept: {report.onlyInA}
            </span>
            {jobId && (
              <a className="download-link" href={mergedDownloadUrl(jobId)} download="merged.safetensors">
                ⬇ Download merged.safetensors
              </a>
            )}
          </div>
        </>
      )}
    </div>
  );
}

