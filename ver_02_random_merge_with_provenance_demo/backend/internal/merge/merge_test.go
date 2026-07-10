package merge

import (
	"os"
	"path/filepath"
	"testing"

	"hfmergekit/internal/store"
)

func makeFakeModel(t *testing.T, dir string, valueByte byte) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	tensors := map[string][]byte{
		"embed.weight":  make([]byte, 16),
		"layer0.weight": make([]byte, 32),
		"layer0.bias":   make([]byte, 8),
	}
	for name := range tensors {
		for i := range tensors[name] {
			tensors[name][i] = valueByte
		}
	}
	shapes := map[string][]int64{
		"embed.weight":  {4, 1},
		"layer0.weight": {8, 1},
		"layer0.bias":   {2, 1},
	}
	var names []string
	for n := range tensors {
		names = append(names, n)
	}
	err := store.WriteSafetensors(filepath.Join(dir, "model.safetensors"), names,
		func(name string) ([]byte, string, []int64, error) {
			return tensors[name], "F32", shapes[name], nil
		}, nil)
	if err != nil {
		t.Fatalf("writing fake model: %v", err)
	}
	cfg := []byte(`{"model_type":"fake"}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRunEmbedsProvenanceMetadata(t *testing.T) {
	base := t.TempDir()
	dirA := filepath.Join(base, "modelA")
	dirB := filepath.Join(base, "modelB")
	outDir := filepath.Join(base, "out")

	makeFakeModel(t, dirA, 0xAA)
	makeFakeModel(t, dirB, 0xBB)

	report, err := Run(Options{
		ModelADir: dirA,
		ModelBDir: dirB,
		ModelAID:  "org/model-a",
		ModelBID:  "org/model-b",
		OutDir:    outDir,
		SwapRatio: 0.5,
		Seed:      42,
	}, func(string) {})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The report should carry the human-readable repo ids, not the local
	// filesystem paths, when they're provided.
	if report.ModelA != "org/model-a" || report.ModelB != "org/model-b" {
		t.Fatalf("expected report to use repo ids for provenance, got ModelA=%q ModelB=%q", report.ModelA, report.ModelB)
	}

	fh, err := store.ReadHeader(report.OutputFile)
	if err != nil {
		t.Fatalf("ReadHeader on merged output: %v", err)
	}

	// This is the load-bearing assertion for CAVEAT.md's claim that
	// provenance travels *inside* the merged file itself, not just in a
	// separate, easy-to-drop report.
	want := map[string]string{
		"model_a":    "org/model-a",
		"model_b":    "org/model-b",
		"swap_ratio": "0.5000",
		"seed":       "42",
	}
	for k, v := range want {
		if fh.Metadata[k] != v {
			t.Fatalf("expected embedded metadata[%q] = %q, got %q (all metadata: %+v)", k, v, fh.Metadata[k], fh.Metadata)
		}
	}
	if fh.Metadata["disclaimer"] == "" {
		t.Fatalf("expected a non-empty embedded disclaimer explaining this is a spliced, unevaluated model")
	}
	if fh.Metadata["created_at"] == "" {
		t.Fatalf("expected a non-empty embedded created_at timestamp")
	}
}

func TestRunFullSwap(t *testing.T) {
	base := t.TempDir()
	dirA := filepath.Join(base, "modelA")
	dirB := filepath.Join(base, "modelB")
	outDir := filepath.Join(base, "out")

	makeFakeModel(t, dirA, 0xAA)
	makeFakeModel(t, dirB, 0xBB)

	var logLines []string
	report, err := Run(Options{
		ModelADir: dirA,
		ModelBDir: dirB,
		OutDir:    outDir,
		SwapRatio: 1.0, // force every common tensor to come from B
		Seed:      42,
	}, func(line string) { logLines = append(logLines, line) })
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if report.CommonTensors != 3 {
		t.Fatalf("expected 3 common tensors, got %d", report.CommonTensors)
	}
	if report.SwappedCount != 3 {
		t.Fatalf("expected all 3 tensors swapped with ratio=1.0, got %d", report.SwappedCount)
	}
	if len(logLines) == 0 {
		t.Fatalf("expected progress log lines to be recorded")
	}

	// Verify the output file actually contains B's bytes for every tensor.
	fh, err := store.ReadHeader(filepath.Join(outDir, "merged.safetensors"))
	if err != nil {
		t.Fatalf("reading merged output: %v", err)
	}
	for name := range fh.Tensors {
		data, err := store.ReadTensorBytes(fh, name)
		if err != nil {
			t.Fatalf("reading tensor %s: %v", name, err)
		}
		for _, b := range data {
			if b != 0xBB {
				t.Fatalf("tensor %s: expected all bytes 0xBB (from model B), got 0x%X", name, b)
			}
		}
	}

	// config.json should have been copied from model A.
	if _, err := os.Stat(filepath.Join(outDir, "config.json")); err != nil {
		t.Fatalf("expected config.json to be copied to output dir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "report.json")); err != nil {
		t.Fatalf("expected report.json to be written: %v", err)
	}
}

func TestRunZeroSwapKeepsModelA(t *testing.T) {
	base := t.TempDir()
	dirA := filepath.Join(base, "modelA")
	dirB := filepath.Join(base, "modelB")
	outDir := filepath.Join(base, "out")

	makeFakeModel(t, dirA, 0x11)
	makeFakeModel(t, dirB, 0x22)

	report, err := Run(Options{
		ModelADir: dirA,
		ModelBDir: dirB,
		OutDir:    outDir,
		SwapRatio: 0.0,
		Seed:      7,
	}, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if report.SwappedCount != 0 {
		t.Fatalf("expected 0 swaps with ratio=0.0, got %d", report.SwappedCount)
	}

	fh, err := store.ReadHeader(filepath.Join(outDir, "merged.safetensors"))
	if err != nil {
		t.Fatalf("reading merged output: %v", err)
	}
	for name := range fh.Tensors {
		data, err := store.ReadTensorBytes(fh, name)
		if err != nil {
			t.Fatalf("reading tensor %s: %v", name, err)
		}
		for _, b := range data {
			if b != 0x11 {
				t.Fatalf("tensor %s: expected all bytes 0x11 (from model A), got 0x%X", name, b)
			}
		}
	}
}

func TestRunMismatchedArchitectureErrors(t *testing.T) {
	base := t.TempDir()
	dirA := filepath.Join(base, "modelA")
	dirB := filepath.Join(base, "modelB")

	if err := os.MkdirAll(dirA, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dirB, 0o755); err != nil {
		t.Fatal(err)
	}

	err := store.WriteSafetensors(filepath.Join(dirA, "model.safetensors"), []string{"only.in.a"},
		func(name string) ([]byte, string, []int64, error) {
			return []byte{1, 2, 3, 4}, "F32", []int64{1}, nil
		}, nil)
	if err != nil {
		t.Fatal(err)
	}
	err = store.WriteSafetensors(filepath.Join(dirB, "model.safetensors"), []string{"only.in.b"},
		func(name string) ([]byte, string, []int64, error) {
			return []byte{5, 6, 7, 8}, "F32", []int64{1}, nil
		}, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Run(Options{
		ModelADir: dirA,
		ModelBDir: dirB,
		OutDir:    filepath.Join(base, "out"),
		SwapRatio: 0.5,
	}, nil)
	if err == nil {
		t.Fatalf("expected an error when models share no common tensors")
	}
}
