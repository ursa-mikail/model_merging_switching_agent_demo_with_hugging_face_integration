// Package api wires up the HTTP handlers for the hf-mergekit-demo backend.
package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"hfmergekit/internal/compat"
	"hfmergekit/internal/download"
	"hfmergekit/internal/jobs"
	"hfmergekit/internal/merge"
	"hfmergekit/internal/models"
)

// Server holds shared dependencies for all handlers.
type Server struct {
	DataDir string
	Jobs    *jobs.Manager

	// MaxConcurrentDownloads/MaxConcurrentMerges cap how many download/merge
	// jobs can be pending or running at once. This is a basic, built-in
	// mitigation against resource-exhaustion abuse if this backend is ever
	// reachable by untrusted clients - see CAVEAT.md. A value <= 0 means
	// unlimited.
	MaxConcurrentDownloads int
	MaxConcurrentMerges    int
}

const (
	defaultMaxConcurrentDownloads = 4
	defaultMaxConcurrentMerges    = 2
)

// NewServer creates a Server rooted at dataDir (models/ and merged/ live under it).
func NewServer(dataDir string) *Server {
	_ = os.MkdirAll(filepath.Join(dataDir, "models"), 0o755)
	_ = os.MkdirAll(filepath.Join(dataDir, "merged"), 0o755)
	return &Server{
		DataDir:                dataDir,
		Jobs:                   jobs.NewManager(),
		MaxConcurrentDownloads: defaultMaxConcurrentDownloads,
		MaxConcurrentMerges:    defaultMaxConcurrentMerges,
	}
}

// Routes builds the ServeMux with all routes registered.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("GET /api/models", s.handleListModels)
	mux.HandleFunc("GET /api/models/local", s.handleListLocalModels)
	mux.HandleFunc("GET /api/models/compat", s.handleCompat)

	mux.HandleFunc("POST /api/jobs/download", s.handleStartDownload)
	mux.HandleFunc("POST /api/jobs/merge", s.handleStartMerge)
	mux.HandleFunc("GET /api/jobs/{id}", s.handleJobStatus)
	mux.HandleFunc("GET /api/jobs/{id}/events", s.handleJobEvents)
	mux.HandleFunc("GET /api/jobs", s.handleListJobs)

	mux.HandleFunc("GET /api/merged/{id}/download", s.handleDownloadMerged)

	return withCORS(withLogging(mux))
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s (%s)", r.Method, r.URL.Path, time.Since(start))
	})
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	// GetCatalog fetches live from the HuggingFace Hub (cached briefly), so
	// the dropdown always reflects what's actually on the Hub rather than a
	// hardcoded list. See internal/models.GetCatalog for the fallback story.
	writeJSON(w, http.StatusOK, models.GetCatalog())
}

// localModelInfo describes a model already present on disk.
type localModelInfo struct {
	ID            string   `json:"id"`
	Files         []string `json:"files"`
	TotalBytes    int64    `json:"totalBytes"`
	HasSafeTensor bool     `json:"hasSafeTensor"`
}

func (s *Server) handleListLocalModels(w http.ResponseWriter, r *http.Request) {
	root := filepath.Join(s.DataDir, "models")
	entries, err := os.ReadDir(root)
	if err != nil {
		writeJSON(w, http.StatusOK, []localModelInfo{})
		return
	}

	out := make([]localModelInfo, 0)
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		// Model ids may contain a "/" which we encode as "__" on disk.
		id := strings.ReplaceAll(e.Name(), "__", "/")
		dir := filepath.Join(root, e.Name())
		files, _ := os.ReadDir(dir)
		info := localModelInfo{ID: id, Files: make([]string, 0)}
		for _, f := range files {
			if f.IsDir() {
				continue
			}
			fi, err := f.Info()
			if err == nil {
				info.TotalBytes += fi.Size()
			}
			info.Files = append(info.Files, f.Name())
			if strings.HasSuffix(f.Name(), ".safetensors") {
				info.HasSafeTensor = true
			}
		}
		out = append(out, info)
	}
	writeJSON(w, http.StatusOK, out)
}

// encodeModelDir turns a HF repo id like "org/name" into a filesystem-safe
// directory name.
func encodeModelDir(repoID string) string {
	return strings.ReplaceAll(repoID, "/", "__")
}

// compatEntryPayload is the JSON shape returned for one candidate model in
// a compatibility check against some base model.
type compatEntryPayload struct {
	ID                   string `json:"id"`
	Compatible           bool   `json:"compatible"`
	Checked              bool   `json:"checked"` // false if we couldn't inspect this model at all
	CommonTensors        int    `json:"commonTensors"`
	SharedNames          int    `json:"sharedNames"`
	CandidateTensorCount int    `json:"candidateTensorCount"`
	Reason               string `json:"reason"`
	Source               string `json:"source"` // "local" | "remote" | "error"
}

type compatResponsePayload struct {
	Base            string               `json:"base"`
	BaseTensorCount int                  `json:"baseTensorCount"`
	BaseSource      string               `json:"baseSource"`
	Results         []compatEntryPayload `json:"results"`
}

// handleCompat answers "given this base model, which other models can it
// actually be merged with?" - the live tensor-shape check that replaced the
// old cosmetic "architecture family" guess. It checks the base model
// against every catalog entry plus every already-downloaded local model, so
// the frontend can filter the *other* dropdown down to only real,
// currently-verified-compatible options and explain why.
func (s *Server) handleCompat(w http.ResponseWriter, r *http.Request) {
	base := strings.TrimSpace(r.URL.Query().Get("base"))
	if base == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("base query parameter is required"))
		return
	}

	baseSig, baseSource, err := s.signatureFor(base)
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("could not inspect base model %q: %w", base, err))
		return
	}

	candidateIDs := s.compatCandidateIDs(base)
	results := make([]compatEntryPayload, len(candidateIDs))

	var wg sync.WaitGroup
	const workers = 10
	jobsCh := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobsCh {
				results[i] = s.computeCompatEntry(candidateIDs[i], baseSig)
			}
		}()
	}
	for i := range candidateIDs {
		jobsCh <- i
	}
	close(jobsCh)
	wg.Wait()

	writeJSON(w, http.StatusOK, compatResponsePayload{
		Base:            base,
		BaseTensorCount: len(baseSig),
		BaseSource:      baseSource,
		Results:         results,
	})
}

// signatureFor returns a model's tensor-shape signature, preferring an
// already-downloaded local copy (exact, no network) and falling back to a
// live remote header fetch (see internal/compat) when it hasn't been
// downloaded yet.
func (s *Server) signatureFor(id string) (compat.Signature, string, error) {
	dir := filepath.Join(s.DataDir, "models", encodeModelDir(id))
	if hasSafetensorsFiles(dir) {
		if sig, err := compat.LocalSignature(dir); err == nil {
			return sig, "local", nil
		}
		// fall through to a remote check if the local files are somehow bad
	}
	sig, err := compat.RemoteSignature(id)
	return sig, "remote", err
}

func hasSafetensorsFiles(dir string) bool {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return len(matches) > 0
}

// compatCandidateIDs is every model we can meaningfully check against
// `base`: the full live catalog plus any already-downloaded local models
// (which may have been fetched via a custom repo id that isn't in the
// catalog at all), minus the base itself.
func (s *Server) compatCandidateIDs(base string) []string {
	seen := map[string]bool{base: true}
	var ids []string

	for _, c := range models.GetCatalog() {
		if !seen[c.ID] {
			seen[c.ID] = true
			ids = append(ids, c.ID)
		}
	}

	root := filepath.Join(s.DataDir, "models")
	entries, _ := os.ReadDir(root)
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		id := strings.ReplaceAll(e.Name(), "__", "/")
		if !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	return ids
}

func (s *Server) computeCompatEntry(id string, baseSig compat.Signature) compatEntryPayload {
	sig, source, err := s.signatureFor(id)
	if err != nil {
		return compatEntryPayload{
			ID:         id,
			Compatible: false,
			Checked:    false,
			Reason:     fmt.Sprintf("Could not inspect this model's tensors (%v).", err),
			Source:     "error",
		}
	}
	cmp := compat.Compare(baseSig, sig)
	return compatEntryPayload{
		ID:                   id,
		Compatible:           cmp.Compatible,
		Checked:              true,
		CommonTensors:        cmp.MatchingTensors,
		SharedNames:          cmp.CommonNames,
		CandidateTensorCount: cmp.CandidateCount,
		Reason:               cmp.Reason,
		Source:               source,
	}
}

type downloadRequest struct {
	ModelID string `json:"modelId"`
}

func (s *Server) handleStartDownload(w http.ResponseWriter, r *http.Request) {
	var req downloadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON body: %w", err))
		return
	}
	req.ModelID = strings.TrimSpace(req.ModelID)
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("modelId is required"))
		return
	}

	if s.MaxConcurrentDownloads > 0 && s.Jobs.ActiveCount("download") >= s.MaxConcurrentDownloads {
		writeError(w, http.StatusTooManyRequests, fmt.Errorf(
			"too many downloads already in progress (limit=%d) — wait for one to finish before starting another",
			s.MaxConcurrentDownloads))
		return
	}

	job := s.Jobs.Create("download")
	destDir := filepath.Join(s.DataDir, "models", encodeModelDir(req.ModelID))

	go func() {
		job.MarkRunning()
		job.Log("Starting download job for %s", req.ModelID)
		res, err := download.Download(req.ModelID, destDir, func(line string) {
			job.Log("%s", line)
		})
		if err != nil {
			job.Log("ERROR: %v", err)
			job.Fail(err)
			return
		}
		job.Succeed(res)
	}()

	writeJSON(w, http.StatusAccepted, map[string]string{"jobId": job.ID})
}

type mergeRequest struct {
	ModelAID  string  `json:"modelAId"`
	ModelBID  string  `json:"modelBId"`
	SwapRatio float64 `json:"swapRatio"`
	Seed      int64   `json:"seed"`
}

func (s *Server) handleStartMerge(w http.ResponseWriter, r *http.Request) {
	var req mergeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON body: %w", err))
		return
	}
	req.ModelAID = strings.TrimSpace(req.ModelAID)
	req.ModelBID = strings.TrimSpace(req.ModelBID)
	if req.ModelAID == "" || req.ModelBID == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("modelAId and modelBId are both required"))
		return
	}

	dirA := filepath.Join(s.DataDir, "models", encodeModelDir(req.ModelAID))
	dirB := filepath.Join(s.DataDir, "models", encodeModelDir(req.ModelBID))
	if _, err := os.Stat(dirA); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model A (%s) has not been downloaded yet", req.ModelAID))
		return
	}
	if _, err := os.Stat(dirB); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model B (%s) has not been downloaded yet", req.ModelBID))
		return
	}

	if s.MaxConcurrentMerges > 0 && s.Jobs.ActiveCount("merge") >= s.MaxConcurrentMerges {
		writeError(w, http.StatusTooManyRequests, fmt.Errorf(
			"too many merges already in progress (limit=%d) — wait for one to finish before starting another",
			s.MaxConcurrentMerges))
		return
	}

	job := s.Jobs.Create("merge")
	outDir := filepath.Join(s.DataDir, "merged", job.ID)

	go func() {
		job.MarkRunning()
		job.Log("Starting merge job: A=%s B=%s ratio=%.2f", req.ModelAID, req.ModelBID, req.SwapRatio)
		report, err := merge.Run(merge.Options{
			ModelADir: dirA,
			ModelBDir: dirB,
			ModelAID:  req.ModelAID,
			ModelBID:  req.ModelBID,
			OutDir:    outDir,
			SwapRatio: req.SwapRatio,
			Seed:      req.Seed,
		}, func(line string) {
			job.Log("%s", line)
		})
		if err != nil {
			job.Log("ERROR: %v", err)
			job.Fail(err)
			return
		}
		job.Succeed(report)
	}()

	writeJSON(w, http.StatusAccepted, map[string]string{"jobId": job.ID})
}

func (s *Server) handleJobStatus(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.Jobs.Get(id)
	if !ok {
		writeError(w, http.StatusNotFound, fmt.Errorf("job %s not found", id))
		return
	}
	status, logLines := job.Snapshot()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"id":        job.ID,
		"type":      job.Type,
		"status":    status,
		"log":       logLines,
		"error":     job.Error,
		"result":    job.Result,
		"createdAt": job.CreatedAt,
		"updatedAt": job.UpdatedAt,
	})
}

func (s *Server) handleListJobs(w http.ResponseWriter, r *http.Request) {
	all := s.Jobs.List()
	type summary struct {
		ID        string      `json:"id"`
		Type      string      `json:"type"`
		Status    jobs.Status `json:"status"`
		CreatedAt time.Time   `json:"createdAt"`
	}
	out := make([]summary, 0, len(all))
	for _, j := range all {
		status, _ := j.Snapshot()
		out = append(out, summary{ID: j.ID, Type: j.Type, Status: status, CreatedAt: j.CreatedAt})
	}
	writeJSON(w, http.StatusOK, out)
}

// handleJobEvents streams the job's log lines as Server-Sent Events, closing
// automatically once the job finishes.
func (s *Server) handleJobEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.Jobs.Get(id)
	if !ok {
		http.Error(w, "job not found", http.StatusNotFound)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Replay everything logged so far.
	_, existing := job.Snapshot()
	for _, line := range existing {
		fmt.Fprintf(w, "data: %s\n\n", sseEscape(line))
	}
	flusher.Flush()

	status, _ := job.Snapshot()
	if status == jobs.StatusSucceeded || status == jobs.StatusFailed {
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", status)
		flusher.Flush()
		return
	}

	ch, unsubscribe := job.Subscribe()
	defer unsubscribe()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case line := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", sseEscape(line))
			flusher.Flush()
		case <-job.Done():
			status, _ := job.Snapshot()
			fmt.Fprintf(w, "event: done\ndata: %s\n\n", status)
			flusher.Flush()
			return
		case <-ticker.C:
			fmt.Fprintf(w, ": keep-alive\n\n")
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func sseEscape(line string) string {
	return strings.ReplaceAll(line, "\n", " ")
}

func (s *Server) handleDownloadMerged(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.Jobs.Get(id)
	if !ok {
		http.Error(w, "job not found", http.StatusNotFound)
		return
	}
	status, _ := job.Snapshot()
	if status != jobs.StatusSucceeded {
		http.Error(w, "merge job has not succeeded", http.StatusConflict)
		return
	}
	path := filepath.Join(s.DataDir, "merged", job.ID, "merged.safetensors")
	if _, err := os.Stat(path); err != nil {
		http.Error(w, "merged file not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Disposition", "attachment; filename=\"merged.safetensors\"")
	http.ServeFile(w, r, path)
}
