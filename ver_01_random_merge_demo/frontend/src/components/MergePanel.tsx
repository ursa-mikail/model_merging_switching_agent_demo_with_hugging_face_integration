interface Status {
  kind: "ok" | "warning" | "info";
  text: string;
}

interface Props {
  swapRatio: number;
  onSwapRatioChange: (v: number) => void;
  seed: number;
  onSeedChange: (v: number) => void;
  onRandomizeSeed: () => void;
  onMerge: () => void;
  canMerge: boolean;
  merging: boolean;
  status?: Status | null;
}

export default function MergePanel({
  swapRatio,
  onSwapRatioChange,
  seed,
  onSeedChange,
  onRandomizeSeed,
  onMerge,
  canMerge,
  merging,
  status,
}: Props) {
  return (
    <div className="merge-bar">
      <div className="merge-bar-title">
        <span className="splice-icon">🧬</span>
        Splice controls
      </div>

      <div className="ratio-control">
        <div className="ratio-row">
          <span>Swap ratio (chance each shared tensor is taken from B)</span>
          <span className="ratio-value">{Math.round(swapRatio * 100)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(swapRatio * 100)}
          onChange={(e) => onSwapRatioChange(Number(e.target.value) / 100)}
        />
      </div>

      <div className="seed-control">
        <label className="field-label" htmlFor="seed-input">
          Random seed
        </label>
        <div className="seed-input-row">
          <input
            id="seed-input"
            type="number"
            value={seed}
            onChange={(e) => onSeedChange(Number(e.target.value))}
          />
          <button type="button" className="icon-btn" title="Randomize seed" onClick={onRandomizeSeed}>
            🎲
          </button>
        </div>
      </div>

      {status && (
        <div
          className={
            status.kind === "warning" ? "merge-warning" : status.kind === "ok" ? "merge-status-ok" : "merge-status-info"
          }
          role="status"
        >
          {status.text}
        </div>
      )}

      <button
        className="merge-btn"
        onClick={onMerge}
        disabled={!canMerge || merging}
        title={status?.kind === "warning" ? status.text : undefined}
      >
        {merging ? "Splicing..." : "Merge models"}
      </button>
    </div>
  );
}
