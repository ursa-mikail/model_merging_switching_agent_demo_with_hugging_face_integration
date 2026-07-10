// Package merge implements a simplified, educational stand-in for mergekit's
// weight-merging algorithms. Real mergekit supports strategies like SLERP,
// TIES, and DARE. This demo implements a much simpler "random tensor swap":
// for every tensor that exists in both source models with matching shape and
// dtype, we flip a weighted coin and take that tensor from model B instead of
// model A. The result is a Frankenstein checkpoint that is architecturally
// valid (same config, same tensor shapes) but behaviorally scrambled — a fun,
// fast way to demonstrate the download -> merge -> inspect pipeline without
// needing GPU-heavy interpolation math.
package merge

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"hfmergekit/internal/store"
)

// TensorSwapDetail records what happened to a single tensor during the merge.
type TensorSwapDetail struct {
	Name        string  `json:"name"`
	Shape       []int64 `json:"shape"`
	Dtype       string  `json:"dtype"`
	Bytes       int64   `json:"bytes"`
	SourceModel string  `json:"sourceModel"` // "A" or "B"
	Swapped     bool    `json:"swapped"`
}

// Report summarizes the outcome of a merge job.
type Report struct {
	ModelA        string             `json:"modelA"`
	ModelB        string             `json:"modelB"`
	SwapRatio     float64            `json:"swapRatio"`
	Seed          int64              `json:"seed"`
	CommonTensors int                `json:"commonTensors"`
	OnlyInA       int                `json:"onlyInA"`
	OnlyInB       int                `json:"onlyInB"`
	SwappedCount  int                `json:"swappedCount"`
	TotalOutBytes int64              `json:"totalOutBytes"`
	OutputFile    string             `json:"outputFile"`
	Details       []TensorSwapDetail `json:"details"`
}

// ProgressFunc receives human-readable log lines during the merge.
type ProgressFunc func(line string)

// Options configures a merge run.
type Options struct {
	ModelADir string
	ModelBDir string
	// ModelAID/ModelBID are the original HuggingFace repo ids (e.g.
	// "org/model-name"), used for provenance in the report and the merged
	// file's embedded metadata. If empty, ModelADir/ModelBDir are used
	// instead (less readable, but keeps this backward compatible for
	// direct callers that only have local paths).
	ModelAID  string
	ModelBID  string
	OutDir    string
	SwapRatio float64 // 0..1, fraction of common tensors to take from B
	Seed      int64   // 0 means "use a random seed"
}

// Run performs the random tensor-swap merge and writes merged.safetensors
// plus a JSON report into OutDir.
func Run(opts Options, progress ProgressFunc) (*Report, error) {
	if progress == nil {
		progress = func(string) {}
	}
	if opts.SwapRatio < 0 {
		opts.SwapRatio = 0
	}
	if opts.SwapRatio > 1 {
		opts.SwapRatio = 1
	}
	seed := opts.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	progress(fmt.Sprintf("Loading tensor index for model A (%s)...", opts.ModelADir))
	idxA, _, err := store.LoadModelIndex(opts.ModelADir)
	if err != nil {
		return nil, fmt.Errorf("model A: %w", err)
	}
	progress(fmt.Sprintf("  -> %d tensors found in model A", len(idxA)))

	progress(fmt.Sprintf("Loading tensor index for model B (%s)...", opts.ModelBDir))
	idxB, _, err := store.LoadModelIndex(opts.ModelBDir)
	if err != nil {
		return nil, fmt.Errorf("model B: %w", err)
	}
	progress(fmt.Sprintf("  -> %d tensors found in model B", len(idxB)))

	// Deterministic ordering for reproducibility given the same seed.
	var allNames []string
	seen := map[string]bool{}
	for n := range idxA {
		allNames = append(allNames, n)
		seen[n] = true
	}
	for n := range idxB {
		if !seen[n] {
			allNames = append(allNames, n)
		}
	}
	sort.Strings(allNames)

	modelALabel := opts.ModelAID
	if modelALabel == "" {
		modelALabel = opts.ModelADir
	}
	modelBLabel := opts.ModelBID
	if modelBLabel == "" {
		modelBLabel = opts.ModelBDir
	}

	report := &Report{
		ModelA:    modelALabel,
		ModelB:    modelBLabel,
		SwapRatio: opts.SwapRatio,
		Seed:      seed,
	}

	order := make([]string, 0, len(allNames))
	details := make([]TensorSwapDetail, 0, len(allNames))

	progress("Comparing tensors and rolling the dice on swaps...")
	for _, name := range allNames {
		locA, inA := idxA[name]
		locB, inB := idxB[name]

		switch {
		case inA && inB:
			sameShape := shapesEqual(locA.Info.Shape, locB.Info.Shape)
			sameDtype := locA.Info.Dtype == locB.Info.Dtype
			if !sameShape || !sameDtype {
				// Shape/dtype mismatch: keep A's tensor untouched, can't swap safely.
				report.OnlyInA++ // counted as "kept as-is from A" for reporting simplicity
				order = append(order, name)
				details = append(details, TensorSwapDetail{
					Name: name, Shape: locA.Info.Shape, Dtype: locA.Info.Dtype,
					Bytes:       locA.Info.DataOffsets[1] - locA.Info.DataOffsets[0],
					SourceModel: "A", Swapped: false,
				})
				continue
			}
			report.CommonTensors++
			takeB := rng.Float64() < opts.SwapRatio
			order = append(order, name)
			chosenLoc := locA
			source := "A"
			if takeB {
				chosenLoc = locB
				source = "B"
				report.SwappedCount++
			}
			details = append(details, TensorSwapDetail{
				Name: name, Shape: chosenLoc.Info.Shape, Dtype: chosenLoc.Info.Dtype,
				Bytes:       chosenLoc.Info.DataOffsets[1] - chosenLoc.Info.DataOffsets[0],
				SourceModel: source, Swapped: takeB,
			})
		case inA:
			report.OnlyInA++
			order = append(order, name)
			details = append(details, TensorSwapDetail{
				Name: name, Shape: locA.Info.Shape, Dtype: locA.Info.Dtype,
				Bytes:       locA.Info.DataOffsets[1] - locA.Info.DataOffsets[0],
				SourceModel: "A", Swapped: false,
			})
		case inB:
			// Present only in B: not included in the merged output, since the
			// output follows model A's architecture/config. Recorded for the report.
			report.OnlyInB++
		}
	}

	if report.CommonTensors == 0 {
		return nil, fmt.Errorf("model A and model B share no tensors with matching name+shape+dtype; pick two models of the same architecture family to merge")
	}

	progress(fmt.Sprintf("Swapped %d of %d common tensors (ratio=%.2f, seed=%d).", report.SwappedCount, report.CommonTensors, opts.SwapRatio, seed))

	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return nil, fmt.Errorf("creating output directory: %w", err)
	}

	outPath := filepath.Join(opts.OutDir, "merged.safetensors")
	progress("Writing merged.safetensors ...")

	getter := func(name string) ([]byte, string, []int64, error) {
		locA, inA := idxA[name]
		locB, inB := idxB[name]
		// Determine which one we picked, matching the decision above.
		for _, d := range details {
			if d.Name == name {
				if d.Swapped && inB {
					data, err := store.ReadTensorBytes(locB.Header, name)
					return data, d.Dtype, d.Shape, err
				}
				if inA {
					data, err := store.ReadTensorBytes(locA.Header, name)
					return data, d.Dtype, d.Shape, err
				}
			}
		}
		return nil, "", nil, fmt.Errorf("internal error: no decision recorded for %q", name)
	}

	meta := map[string]string{
		"merged_by":  "hf-mergekit-demo (Splice)",
		"model_a":    modelALabel,
		"model_b":    modelBLabel,
		"swap_ratio": fmt.Sprintf("%.4f", opts.SwapRatio),
		"seed":       fmt.Sprintf("%d", seed),
		"created_at": time.Now().UTC().Format(time.RFC3339),
		// Deliberately embedded so this can't be silently stripped or
		// mistaken for an original, untouched checkpoint: every tensor here
		// is either model_a's or model_b's byte-for-byte original weight,
		// chosen by an unweighted per-tensor coin flip - not a trained,
		// evaluated, or safety-reviewed model. See CAVEAT.md.
		"disclaimer": "Randomly spliced from two HuggingFace checkpoints for educational purposes; not evaluated, not safety-reviewed, not fit for deployment without independent testing.",
	}

	if err := store.WriteSafetensors(outPath, order, getter, meta); err != nil {
		return nil, fmt.Errorf("writing merged safetensors: %w", err)
	}

	fi, err := os.Stat(outPath)
	if err == nil {
		report.TotalOutBytes = fi.Size()
	}
	report.OutputFile = outPath
	report.Details = details

	// Copy config.json from model A if present, so the merged checkpoint is
	// architecturally loadable (same config as its base).
	if data, err := os.ReadFile(filepath.Join(opts.ModelADir, "config.json")); err == nil {
		_ = os.WriteFile(filepath.Join(opts.OutDir, "config.json"), data, 0o644)
	}

	if err := writeReportJSON(opts.OutDir, report); err != nil {
		progress(fmt.Sprintf("Warning: could not write report.json: %v", err))
	}

	progress(fmt.Sprintf("Merge complete. Output: %s (%s)", outPath, humanBytes(report.TotalOutBytes)))
	return report, nil
}

func shapesEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func writeReportJSON(dir string, report *Report) error {
	f, err := os.Create(filepath.Join(dir, "report.json"))
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}
