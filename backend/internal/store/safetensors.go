// Package store implements a small, dependency-free reader/writer for the
// safetensors file format used by HuggingFace model checkpoints.
//
// Format recap (see https://github.com/huggingface/safetensors):
//
//	[8 bytes]      little-endian uint64 N = length of the JSON header
//	[N bytes]      JSON header: { tensorName: {dtype, shape, data_offsets:[start,end]}, ... }
//	[remaining]    raw tensor bytes, referenced by data_offsets relative to
//	               the start of this data section
package store

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// TensorInfo describes one tensor's metadata as stored in a safetensors header.
type TensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// FileHeader is the parsed header of a single safetensors file, plus the
// byte offset in the file where the data section begins.
type FileHeader struct {
	Path      string
	Tensors   map[string]TensorInfo
	Metadata  map[string]string
	DataStart int64 // absolute byte offset in the file where data begins
}

// ReadHeader parses the safetensors header of the file at path without
// reading the (potentially huge) tensor data itself.
func ReadHeader(path string) (*FileHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lenBuf [8]byte
	if _, err := f.ReadAt(lenBuf[:], 0); err != nil {
		return nil, fmt.Errorf("reading safetensors length prefix: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])

	headerBytes := make([]byte, headerLen)
	if _, err := f.ReadAt(headerBytes, 8); err != nil {
		return nil, fmt.Errorf("reading safetensors header: %w", err)
	}

	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("parsing safetensors header JSON: %w", err)
	}

	fh := &FileHeader{
		Path:      path,
		Tensors:   map[string]TensorInfo{},
		Metadata:  map[string]string{},
		DataStart: 8 + int64(headerLen),
	}

	for name, msg := range raw {
		if name == "__metadata__" {
			_ = json.Unmarshal(msg, &fh.Metadata)
			continue
		}
		var ti TensorInfo
		if err := json.Unmarshal(msg, &ti); err != nil {
			return nil, fmt.Errorf("parsing tensor %q: %w", name, err)
		}
		fh.Tensors[name] = ti
	}

	return fh, nil
}

// ReadTensorBytes reads the raw bytes for a single tensor from its file.
func ReadTensorBytes(fh *FileHeader, name string) ([]byte, error) {
	ti, ok := fh.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not present in %s", name, fh.Path)
	}
	f, err := os.Open(fh.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	size := ti.DataOffsets[1] - ti.DataOffsets[0]
	buf := make([]byte, size)
	if _, err := f.ReadAt(buf, fh.DataStart+ti.DataOffsets[0]); err != nil {
		return nil, fmt.Errorf("reading tensor %q data: %w", name, err)
	}
	return buf, nil
}

// TensorLocation records which file (and shard, if any) a tensor lives in.
// It is used to build a unified view across single-file and sharded models.
type TensorLocation struct {
	Header *FileHeader
	Info   TensorInfo
}

// ModelTensorIndex maps tensor name -> its location, aggregated across every
// *.safetensors file found in a model directory.
type ModelTensorIndex map[string]TensorLocation

// LoadModelIndex scans dir for *.safetensors files and builds a unified
// tensor-name -> location index across all of them (handling both
// single-file checkpoints and sharded ones).
func LoadModelIndex(dir string) (ModelTensorIndex, []*FileHeader, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, nil, err
	}
	if len(matches) == 0 {
		return nil, nil, fmt.Errorf("no .safetensors files found in %s", dir)
	}
	sort.Strings(matches)

	index := ModelTensorIndex{}
	var headers []*FileHeader
	for _, path := range matches {
		fh, err := ReadHeader(path)
		if err != nil {
			return nil, nil, fmt.Errorf("reading header of %s: %w", path, err)
		}
		headers = append(headers, fh)
		for name, ti := range fh.Tensors {
			index[name] = TensorLocation{Header: fh, Info: ti}
		}
	}
	return index, headers, nil
}

// WriteSafetensors writes a new single-file safetensors checkpoint given an
// ordered list of tensor names and a lookup function that returns the raw
// bytes + dtype/shape for each name.
func WriteSafetensors(outPath string, order []string, get func(name string) (data []byte, dtype string, shape []int64, err error), extraMetadata map[string]string) error {
	type entry struct {
		name  string
		data  []byte
		dtype string
		shape []int64
	}

	entries := make([]entry, 0, len(order))
	for _, name := range order {
		data, dtype, shape, err := get(name)
		if err != nil {
			return fmt.Errorf("fetching tensor %q: %w", name, err)
		}
		entries = append(entries, entry{name, data, dtype, shape})
	}

	headerMap := map[string]interface{}{}
	if len(extraMetadata) > 0 {
		headerMap["__metadata__"] = extraMetadata
	}

	var offset int64
	for _, e := range entries {
		start := offset
		end := offset + int64(len(e.data))
		headerMap[e.name] = TensorInfo{
			Dtype:       e.dtype,
			Shape:       e.shape,
			DataOffsets: [2]int64{start, end},
		}
		offset = end
	}

	headerBytes, err := json.Marshal(headerMap)
	if err != nil {
		return fmt.Errorf("marshaling safetensors header: %w", err)
	}

	// safetensors requires the header to be padded so the data section is
	// aligned; padding with spaces inside the JSON is safe because it occurs
	// only in whitespace between tokens is NOT guaranteed, so instead we pad
	// by adding a trailing space inside the JSON object via a metadata pad key
	// only if needed. Simpler: many implementations just pad with spaces
	// after the closing brace is not allowed (must be valid JSON exactly of
	// declared length). We pad by inserting extra spaces into a dummy
	// metadata field before marshaling would be more complex, so instead we
	// simply do not pad — 8-byte alignment is a recommendation, not a
	// requirement, so an unpadded header is valid.

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(headerBytes); err != nil {
		return err
	}

	var buf bytes.Buffer
	for _, e := range entries {
		buf.Write(e.data)
	}
	if _, err := f.Write(buf.Bytes()); err != nil {
		return err
	}
	return nil
}
