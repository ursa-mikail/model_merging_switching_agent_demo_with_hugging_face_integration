package store

import (
	"bytes"
	"path/filepath"
	"testing"
)

func writeTestSafetensors(t *testing.T, path string, tensors map[string][]byte, shapes map[string][]int64) {
	t.Helper()
	var names []string
	for n := range tensors {
		names = append(names, n)
	}
	err := WriteSafetensors(path, names, func(name string) ([]byte, string, []int64, error) {
		return tensors[name], "F32", shapes[name], nil
	}, nil)
	if err != nil {
		t.Fatalf("WriteSafetensors: %v", err)
	}
}

func TestWriteReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")

	tensors := map[string][]byte{
		"layer.weight": bytes.Repeat([]byte{0x01, 0x02, 0x03, 0x04}, 4), // 16 bytes = 4 float32
		"layer.bias":   bytes.Repeat([]byte{0xAA, 0xBB}, 2),             // 4 bytes
	}
	shapes := map[string][]int64{
		"layer.weight": {2, 2},
		"layer.bias":   {2},
	}
	writeTestSafetensors(t, path, tensors, shapes)

	fh, err := ReadHeader(path)
	if err != nil {
		t.Fatalf("ReadHeader: %v", err)
	}
	if len(fh.Tensors) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(fh.Tensors))
	}

	for name, want := range tensors {
		got, err := ReadTensorBytes(fh, name)
		if err != nil {
			t.Fatalf("ReadTensorBytes(%s): %v", name, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("tensor %s: got %v want %v", name, got, want)
		}
		ti := fh.Tensors[name]
		if ti.Dtype != "F32" {
			t.Fatalf("tensor %s: expected dtype F32, got %s", name, ti.Dtype)
		}
	}
}

func TestLoadModelIndexShardedAndSingle(t *testing.T) {
	dir := t.TempDir()

	// Two separate safetensors files simulating shards.
	writeTestSafetensors(t, filepath.Join(dir, "shard1.safetensors"), map[string][]byte{
		"a": {1, 2, 3, 4},
	}, map[string][]int64{"a": {1}})
	writeTestSafetensors(t, filepath.Join(dir, "shard2.safetensors"), map[string][]byte{
		"b": {5, 6, 7, 8},
	}, map[string][]int64{"b": {1}})

	idx, headers, err := LoadModelIndex(dir)
	if err != nil {
		t.Fatalf("LoadModelIndex: %v", err)
	}
	if len(headers) != 2 {
		t.Fatalf("expected 2 headers, got %d", len(headers))
	}
	if _, ok := idx["a"]; !ok {
		t.Fatalf("expected tensor 'a' in index")
	}
	if _, ok := idx["b"]; !ok {
		t.Fatalf("expected tensor 'b' in index")
	}
}

func TestReadHeaderMissingFile(t *testing.T) {
	_, err := ReadHeader("/nonexistent/path/model.safetensors")
	if err == nil {
		t.Fatalf("expected error for missing file")
	}
}

func TestLoadModelIndexNoFiles(t *testing.T) {
	dir := t.TempDir()
	_, _, err := LoadModelIndex(dir)
	if err == nil {
		t.Fatalf("expected error for directory with no safetensors files")
	}
}
