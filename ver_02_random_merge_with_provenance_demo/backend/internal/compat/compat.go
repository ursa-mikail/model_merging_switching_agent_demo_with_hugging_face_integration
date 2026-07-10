// Package compat implements *live* tensor-shape compatibility checks
// between HuggingFace models, replacing the old "cosmetic family label"
// heuristic (e.g. assuming any two "GPT-Neo" models are mergeable). Two
// models are only truly mergeable if they share tensors with matching
// name, shape, and dtype - and the only way to know that for sure is to
// look at the actual safetensors headers.
//
// For models that have already been downloaded, that's a fast local file
// read (see LocalSignature). For models still sitting on the HuggingFace
// Hub, we avoid downloading the (potentially large) weights just to check
// compatibility: a safetensors file's header - the JSON blob describing
// every tensor's name/shape/dtype - lives in the first few KB of the file,
// so we fetch only that via an HTTP Range request (see RemoteSignature).
package compat

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"hfmergekit/internal/models"
	"hfmergekit/internal/store"
)

// TensorShape is the minimal signature of a tensor used for compatibility
// checks: two tensors are considered "the same" if both fields match.
type TensorShape struct {
	Dtype string
	Shape []int64
}

// Signature maps tensor name -> shape signature for an entire model.
type Signature map[string]TensorShape

// LocalSignature builds a Signature by reading the safetensors header(s) of
// a model already downloaded to disk. This is exact and fast - no network
// call, and no need to read the (potentially huge) tensor data itself.
func LocalSignature(dir string) (Signature, error) {
	idx, _, err := store.LoadModelIndex(dir)
	if err != nil {
		return nil, err
	}
	sig := make(Signature, len(idx))
	for name, loc := range idx {
		sig[name] = TensorShape{Dtype: loc.Info.Dtype, Shape: loc.Info.Shape}
	}
	return sig, nil
}

const (
	remoteSuccessTTL = 30 * time.Minute
	remoteErrorTTL   = 2 * time.Minute
	headerProbeBytes = 65536 // most safetensors headers fit comfortably within 64KB
	maxShardsFetched = 8     // bound cost for huge sharded checkpoints
	httpTimeout      = 8 * time.Second
	shardConcurrency = 4
)

var httpClient = &http.Client{Timeout: httpTimeout}

type cacheEntry struct {
	sig       Signature
	err       error
	fetchedAt time.Time
}

var (
	remoteCache   = map[string]cacheEntry{}
	remoteCacheMu sync.Mutex
)

// RemoteSignature fetches (or serves from an in-memory cache) the tensor
// shape signature of a public HuggingFace repo *without downloading the
// full weights*. Results are cached briefly (longer on success, shorter on
// failure) so repeatedly checking the same catalog entry doesn't hammer the
// Hub on every request.
func RemoteSignature(repoID string) (Signature, error) {
	remoteCacheMu.Lock()
	if e, ok := remoteCache[repoID]; ok {
		ttl := remoteSuccessTTL
		if e.err != nil {
			ttl = remoteErrorTTL
		}
		if time.Since(e.fetchedAt) < ttl {
			remoteCacheMu.Unlock()
			return e.sig, e.err
		}
	}
	remoteCacheMu.Unlock()

	sig, err := fetchRemoteSignature(repoID)

	remoteCacheMu.Lock()
	remoteCache[repoID] = cacheEntry{sig: sig, err: err, fetchedAt: time.Now()}
	remoteCacheMu.Unlock()

	return sig, err
}

func fetchRemoteSignature(repoID string) (Signature, error) {
	info, err := models.FetchRepoInfo(repoID)
	if err != nil {
		return nil, err
	}
	if info.Private || info.Disabled {
		return nil, fmt.Errorf("model %q is private or disabled", repoID)
	}

	var shardFiles []string
	for _, s := range info.Siblings {
		if strings.HasSuffix(s.RFilename, ".safetensors") {
			shardFiles = append(shardFiles, s.RFilename)
		}
	}
	if len(shardFiles) == 0 {
		return nil, fmt.Errorf("no .safetensors files found in %q", repoID)
	}
	sort.Strings(shardFiles)
	if len(shardFiles) > maxShardsFetched {
		shardFiles = shardFiles[:maxShardsFetched]
	}

	type shardResult struct {
		sig Signature
		err error
	}
	results := make([]shardResult, len(shardFiles))
	var wg sync.WaitGroup
	sem := make(chan struct{}, shardConcurrency)
	for i, fname := range shardFiles {
		wg.Add(1)
		go func(i int, fname string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			url := "https://huggingface.co/" + repoID + "/resolve/main/" + fname
			sig, err := fetchHeaderOverHTTP(url)
			results[i] = shardResult{sig: sig, err: err}
		}(i, fname)
	}
	wg.Wait()

	combined := Signature{}
	var lastErr error
	okCount := 0
	for _, r := range results {
		if r.err != nil {
			lastErr = r.err
			continue
		}
		okCount++
		for name, ts := range r.sig {
			combined[name] = ts
		}
	}
	if okCount == 0 {
		return nil, fmt.Errorf("could not read tensor headers for %q: %w", repoID, lastErr)
	}
	return combined, nil
}

// fetchHeaderOverHTTP downloads just the safetensors header for the file at
// url (a few KB of JSON), using HTTP Range requests so the (potentially
// multi-GB) tensor data itself is never transferred.
func fetchHeaderOverHTTP(url string) (Signature, error) {
	afterLenPrefix, headerLen, err := probeHeader(url)
	if err != nil {
		return nil, err
	}
	if int64(len(afterLenPrefix)) < headerLen {
		rest, err := fetchHeaderContinuation(url, int64(len(afterLenPrefix)), headerLen-1)
		if err != nil {
			return nil, err
		}
		afterLenPrefix = append(afterLenPrefix, rest...)
	}
	headerBytes := afterLenPrefix[:headerLen]

	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("parsing safetensors header: %w", err)
	}
	sig := Signature{}
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var ti store.TensorInfo
		if err := json.Unmarshal(msg, &ti); err != nil {
			continue
		}
		sig[name] = TensorShape{Dtype: ti.Dtype, Shape: ti.Shape}
	}
	return sig, nil
}

// probeHeader fetches the first headerProbeBytes bytes of the file at url
// and returns everything after the 8-byte length prefix, plus the declared
// header length. If the declared header is longer than what we probed,
// fetchHeaderContinuation is used to get the rest.
func probeHeader(url string) (afterLenPrefix []byte, headerLen int64, err error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("User-Agent", "hf-mergekit-demo/1.0")
	req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", headerProbeBytes-1))

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, 0, fmt.Errorf("fetching header (HTTP %d): %s", resp.StatusCode, string(body))
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, headerProbeBytes))
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 8 {
		return nil, 0, fmt.Errorf("file too small to contain a safetensors header")
	}
	headerLen = int64(binary.LittleEndian.Uint64(data[:8]))
	if headerLen <= 0 || headerLen > 64*1024*1024 {
		return nil, 0, fmt.Errorf("implausible safetensors header length: %d", headerLen)
	}
	return data[8:], headerLen, nil
}

// fetchHeaderContinuation fetches header bytes [start, end] (inclusive,
// relative to the start of the header, i.e. offset by the file's leading
// 8-byte length prefix) when the initial probe didn't cover the whole
// header.
func fetchHeaderContinuation(url string, start, end int64) ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "hf-mergekit-demo/1.0")
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", 8+start, 8+end))
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return nil, fmt.Errorf("fetching header continuation (HTTP %d)", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

// Result summarizes how compatible two tensor signatures are for merging.
type Result struct {
	CommonNames     int    // tensor names present in both signatures
	MatchingTensors int    // of those, how many also match in shape+dtype
	BaseCount       int    // total tensors in the base signature
	CandidateCount  int    // total tensors in the candidate signature
	Compatible      bool   // true iff MatchingTensors > 0
	Reason          string // human-readable explanation, always populated
}

// Compare reports how many tensors two model signatures actually share.
// This is the single source of truth for "can these two models be merged?"
// - both the catalog-wide compatibility filter and the final pre-merge
// gate call this same function, so the UI and the backend can never
// disagree about what counts as compatible.
func Compare(base, candidate Signature) Result {
	res := Result{BaseCount: len(base), CandidateCount: len(candidate)}
	for name, bts := range base {
		cts, ok := candidate[name]
		if !ok {
			continue
		}
		res.CommonNames++
		if bts.Dtype == cts.Dtype && shapesEqual(bts.Shape, cts.Shape) {
			res.MatchingTensors++
		}
	}
	res.Compatible = res.MatchingTensors > 0

	switch {
	case res.MatchingTensors > 0 && res.MatchingTensors == res.CommonNames:
		res.Reason = fmt.Sprintf("All %d shared tensors match exactly in shape and dtype — safe to merge.", res.MatchingTensors)
	case res.MatchingTensors > 0:
		res.Reason = fmt.Sprintf("%d of %d shared tensor names match exactly in shape and dtype — mergeable, though some shared names differ in shape.", res.MatchingTensors, res.CommonNames)
	case res.CommonNames > 0:
		res.Reason = fmt.Sprintf("%d tensor name(s) overlap, but none match in shape/dtype (likely a different width, e.g. hidden size) — cannot merge safely.", res.CommonNames)
	default:
		res.Reason = "No tensor names in common at all — these are different model architectures."
	}
	return res
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
