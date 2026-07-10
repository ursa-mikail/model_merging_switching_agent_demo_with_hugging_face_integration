package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hfmergekit/internal/store"
)

// writeFakeLocalModel writes a minimal single-file safetensors checkpoint
// directly into a server's data directory, as if `modelID` had already been
// downloaded, so tests can exercise compatibility logic without any network
// access.
func writeFakeLocalModel(t *testing.T, srv *Server, modelID string, tensors map[string][]int64, dtype string) {
	t.Helper()
	dir := filepath.Join(srv.DataDir, "models", encodeModelDir(modelID))
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
		t.Fatalf("writing fake local model %s: %v", modelID, err)
	}
}

// TestLocalModelsEmptyIsArrayNotNull guards against a real regression: a Go
// `var out []T` left unassigned marshals to JSON `null`, which crashes any
// frontend code that calls array methods (e.g. `.some(...)`) on the response
// without a null check. The endpoint must always return `[]`, never `null`,
// even when no models have been downloaded yet.
func TestLocalModelsEmptyIsArrayNotNull(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)

	req := httptest.NewRequest(http.MethodGet, "/api/models/local", nil)
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	body := rec.Body.Bytes()

	// Guard against the exact regression: literal JSON null in the body.
	trimmed := string(body)
	for len(trimmed) > 0 && (trimmed[len(trimmed)-1] == '\n' || trimmed[len(trimmed)-1] == ' ') {
		trimmed = trimmed[:len(trimmed)-1]
	}
	if trimmed == "null" {
		t.Fatalf("regression: /api/models/local returned literal JSON null instead of []")
	}

	var parsed []map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("expected valid JSON array, got error: %v (body: %s)", err, body)
	}
	if parsed == nil {
		t.Fatalf("expected a non-nil (empty) slice after unmarshaling, got nil")
	}
	if len(parsed) != 0 {
		t.Fatalf("expected 0 local models in a fresh data dir, got %d", len(parsed))
	}
}

func TestHealthEndpoint(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)

	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestModelsEndpointReturnsCatalog(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)

	req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var parsed []map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(parsed) == 0 {
		t.Fatalf("expected a non-empty curated catalog")
	}
}

func TestCompatRequiresBaseParam(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)

	req := httptest.NewRequest(http.MethodGet, "/api/models/compat", nil)
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 without a base param, got %d", rec.Code)
	}
}

func TestCompatUsesLocalFilesForDownloadedModels(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)

	// Two locally "downloaded" models with identical tensor shapes should
	// be reported compatible using only local files, with no network call.
	writeFakeLocalModel(t, srv, "demo/model-a", map[string][]int64{
		"embed.weight": {4, 4},
		"layer0.bias":  {4},
	}, "F32")
	writeFakeLocalModel(t, srv, "demo/model-b", map[string][]int64{
		"embed.weight": {4, 4},
		"layer0.bias":  {4},
	}, "F32")
	// A third model with the same tensor *names* but different shapes -
	// the classic "same family, different width" trap.
	writeFakeLocalModel(t, srv, "demo/model-c-different-width", map[string][]int64{
		"embed.weight": {8, 8},
		"layer0.bias":  {8},
	}, "F32")

	req := httptest.NewRequest(http.MethodGet, "/api/models/compat?base=demo/model-a", nil)
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (body: %s)", rec.Code, rec.Body.String())
	}

	var resp compatResponsePayload
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v (body: %s)", err, rec.Body.String())
	}
	if resp.BaseTensorCount != 2 {
		t.Fatalf("expected base to report 2 tensors, got %d", resp.BaseTensorCount)
	}
	if resp.BaseSource != "local" {
		t.Fatalf("expected base source to be 'local' for an already-downloaded model, got %q", resp.BaseSource)
	}

	var foundB, foundC bool
	for _, r := range resp.Results {
		switch r.ID {
		case "demo/model-b":
			foundB = true
			if !r.Checked || !r.Compatible {
				t.Fatalf("expected model-b to be checked and compatible, got %+v", r)
			}
			if r.Source != "local" {
				t.Fatalf("expected model-b to be checked via local files, got source=%q", r.Source)
			}
			if r.CommonTensors != 2 {
				t.Fatalf("expected 2 common tensors with model-b, got %d", r.CommonTensors)
			}
		case "demo/model-c-different-width":
			foundC = true
			if !r.Checked {
				t.Fatalf("expected model-c to be checked, got %+v", r)
			}
			if r.Compatible {
				t.Fatalf("expected model-c to be INCOMPATIBLE (same names, different shapes), got %+v", r)
			}
			if r.SharedNames == 0 {
				t.Fatalf("expected shared tensor names to be detected even though shapes differ")
			}
			if r.Reason == "" {
				t.Fatalf("expected a human-readable reason explaining the incompatibility")
			}
		}
	}
	if !foundB {
		t.Fatalf("expected demo/model-b in results")
	}
	if !foundC {
		t.Fatalf("expected demo/model-c-different-width in results")
	}
}

func TestCompatExcludesBaseFromItsOwnResults(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)
	writeFakeLocalModel(t, srv, "demo/only-model", map[string][]int64{"w": {2, 2}}, "F32")

	req := httptest.NewRequest(http.MethodGet, "/api/models/compat?base=demo/only-model", nil)
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var resp compatResponsePayload
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, r := range resp.Results {
		if r.ID == "demo/only-model" {
			t.Fatalf("base model should not appear in its own candidate results")
		}
	}
}

func TestDownloadConcurrencyCapReturns429(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)
	srv.MaxConcurrentDownloads = 1

	// Manually register a "pending" download job to simulate one already in
	// flight, without actually hitting the network.
	srv.Jobs.Create("download")

	req := httptest.NewRequest(http.MethodPost, "/api/jobs/download",
		strings.NewReader(`{"modelId":"org/some-model"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 once the download concurrency cap is hit, got %d (body: %s)", rec.Code, rec.Body.String())
	}
}

func TestDownloadConcurrencyCapDisabledWhenZero(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)
	srv.MaxConcurrentDownloads = 0 // unlimited
	srv.Jobs.Create("download")
	srv.Jobs.Create("download")

	req := httptest.NewRequest(http.MethodPost, "/api/jobs/download",
		strings.NewReader(`{"modelId":"org/some-model"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code == http.StatusTooManyRequests {
		t.Fatalf("expected the concurrency cap to be disabled when set to 0, got 429")
	}
}

func TestMergeWithoutDownloadedModelsErrorsClearly(t *testing.T) {
	dir := t.TempDir()
	srv := NewServer(dir)

	body := `{"modelAId":"nonexistent/a","modelBId":"nonexistent/b","swapRatio":0.5}`
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/merge", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 when models aren't downloaded yet, got %d (body: %s)", rec.Code, rec.Body.String())
	}
}
