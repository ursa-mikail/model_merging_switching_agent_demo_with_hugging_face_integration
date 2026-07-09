import { useState } from "react";
import type { TensorSwapDetail } from "../types/api";

interface Props {
  details: TensorSwapDetail[];
}

/**
 * Renders every tensor as a small colored cell in a grid — cyan for "kept
 * from Specimen A", amber for "spliced in from Specimen B". This gives an
 * instant, at-a-glance picture of the merge pattern, independent of the
 * detailed row-by-row table below it.
 */
export default function DiffMap({ details }: Props) {
  const [hovered, setHovered] = useState<TensorSwapDetail | null>(null);

  if (details.length === 0) return null;

  const swappedCount = details.filter((d) => d.swapped).length;

  return (
    <div className="diffmap">
      <div className="diffmap-header">
        <div className="diffmap-title">Diff map</div>
        <div className="diffmap-legend">
          <span className="legend-item">
            <span className="legend-swatch swatch-a" /> kept &middot; Specimen A
          </span>
          <span className="legend-item">
            <span className="legend-swatch swatch-b" /> spliced &middot; Specimen B
          </span>
          <span className="legend-count">
            {swappedCount} / {details.length} tensors spliced
          </span>
        </div>
      </div>

      <div className="diffmap-grid" onMouseLeave={() => setHovered(null)}>
        {details.map((d) => (
          <div
            key={d.name}
            className={`diffmap-cell ${d.swapped ? "cell-b" : "cell-a"}`}
            onMouseEnter={() => setHovered(d)}
            title={`${d.name}\n[${d.shape.join(", ")}] ${d.dtype}\nsource: ${d.swapped ? "B (spliced)" : "A (kept)"}`}
          />
        ))}
      </div>

      <div className="diffmap-tooltip-line">
        {hovered ? (
          <>
            <span className={`source-badge source-${hovered.sourceModel.toLowerCase()}`}>
              {hovered.sourceModel}
              {hovered.swapped && <span className="swap-flag">SPLICED</span>}
            </span>{" "}
            <code>{hovered.name}</code> &nbsp;[{hovered.shape.join(", ")}] {hovered.dtype}
          </>
        ) : (
          <span className="diffmap-hint">hover any cell to inspect that tensor</span>
        )}
      </div>
    </div>
  );
}
