package compat

import (
	"os"
	"path/filepath"
	"testing"

	"hfmergekit/internal/store"
)

func writeFakeModel(t *testing.T, dir string, tensors map[string][]int64, dtype string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var names []string
	for n := range tensors {
		names = append(names, n)
	}
	err := store.WriteSafetensors(filepath.Join(dir, "model.safetensors"), names,
		func(name string) ([]byte, string, []int64, error) {
			shape := tensors[name]
			size := int64(4)
			for _, d := range shape {
				size *= d
			}
			return make([]byte, size), dtype, shape, nil
		}, nil)
	if err != nil {
		t.Fatalf("writing fake model: %v", err)
	}
}

func TestLocalSignatureMatchesFileContents(t *testing.T) {
	dir := t.TempDir()
	writeFakeModel(t, dir, map[string][]int64{
		"embed.weight": {4, 4},
		"layer0.bias":  {4},
	}, "F32")

	sig, err := LocalSignature(dir)
	if err != nil {
		t.Fatalf("LocalSignature: %v", err)
	}
	if len(sig) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(sig))
	}
	ts, ok := sig["embed.weight"]
	if !ok {
		t.Fatalf("expected embed.weight in signature")
	}
	if ts.Dtype != "F32" || len(ts.Shape) != 2 || ts.Shape[0] != 4 || ts.Shape[1] != 4 {
		t.Fatalf("unexpected shape/dtype for embed.weight: %+v", ts)
	}
}

func TestCompareFullyCompatible(t *testing.T) {
	a := Signature{
		"embed.weight": {Dtype: "F32", Shape: []int64{4, 4}},
		"layer0.bias":  {Dtype: "F32", Shape: []int64{4}},
	}
	b := Signature{
		"embed.weight": {Dtype: "F32", Shape: []int64{4, 4}},
		"layer0.bias":  {Dtype: "F32", Shape: []int64{4}},
	}
	res := Compare(a, b)
	if !res.Compatible {
		t.Fatalf("expected compatible, got: %+v", res)
	}
	if res.MatchingTensors != 2 || res.CommonNames != 2 {
		t.Fatalf("expected 2/2 matching tensors, got %+v", res)
	}
}

func TestCompareSameNamesDifferentShapesIsIncompatible(t *testing.T) {
	// This is exactly the TinyStories-1M vs TinyStories-33M scenario: same
	// tensor *names* (same architecture family), but different widths, so
	// nothing actually lines up.
	a := Signature{
		"embed.weight": {Dtype: "F32", Shape: []int64{64, 64}},
	}
	b := Signature{
		"embed.weight": {Dtype: "F32", Shape: []int64{768, 768}},
	}
	res := Compare(a, b)
	if res.Compatible {
		t.Fatalf("expected incompatible (shape mismatch), got: %+v", res)
	}
	if res.CommonNames != 1 || res.MatchingTensors != 0 {
		t.Fatalf("expected 1 shared name but 0 matches, got %+v", res)
	}
	if res.Reason == "" {
		t.Fatalf("expected a non-empty explanation")
	}
}

func TestCompareDifferentArchitecturesIsIncompatible(t *testing.T) {
	a := Signature{"transformer.wte.weight": {Dtype: "F32", Shape: []int64{50257, 768}}}
	b := Signature{"bert.embeddings.word_embeddings.weight": {Dtype: "F32", Shape: []int64{30522, 128}}}
	res := Compare(a, b)
	if res.Compatible {
		t.Fatalf("expected incompatible (no shared names), got: %+v", res)
	}
	if res.CommonNames != 0 {
		t.Fatalf("expected 0 shared names, got %d", res.CommonNames)
	}
}

func TestCompareDifferentDtypeIsIncompatible(t *testing.T) {
	a := Signature{"w": {Dtype: "F32", Shape: []int64{4, 4}}}
	b := Signature{"w": {Dtype: "F16", Shape: []int64{4, 4}}}
	res := Compare(a, b)
	if res.Compatible {
		t.Fatalf("expected incompatible (dtype mismatch), got: %+v", res)
	}
}
