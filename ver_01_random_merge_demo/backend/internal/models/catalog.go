// Package models provides the catalog of HuggingFace models offered in the
// UI dropdowns, plus the HTTP client used to query the public HuggingFace
// Hub API for repository file listings.
//
// The catalog itself is fetched *live* from the HuggingFace Hub on a
// refresh interval (see GetCatalog) rather than hand-curated: previously
// this package shipped a fixed, hardcoded list of ~10 models. That worked
// as a demo but meant the dropdown never reflected what's actually
// available on the Hub. GetCatalog now pulls the current top-downloaded
// public models with safetensors weights directly from the Hub API, and
// falls back to a small built-in seed list (fallbackCatalog) only if that
// network call fails (e.g. offline dev, HF Hub outage) so the UI never
// shows an empty dropdown.
//
// Note: the "Family" field below is purely informational (shown as a small
// badge in the UI) and is NOT used to decide which models can be merged
// together. Two models can share a cosmetic family label (e.g. both
// "GPT-Neo") while having completely different tensor widths, and vice
// versa. Real mergeability is determined by the internal/compat package,
// which inspects actual tensor shapes.
package models

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// CatalogEntry describes one selectable model in the frontend dropdown.
type CatalogEntry struct {
	ID          string `json:"id"`          // HuggingFace repo id, e.g. "roneneldan/TinyStories-1M"
	Label       string `json:"label"`       // human friendly display name
	Description string `json:"description"` // short blurb shown in the UI
	ApproxSize  string `json:"approxSize"`  // rough size/params, informational only
	Family      string `json:"family"`      // pipeline/library tag, informational only (NOT used for merge gating)
}

// fallbackCatalog is used only when a live fetch from the HuggingFace Hub
// fails (no network, Hub outage, etc.) and we have no previously cached
// result to fall back on. All of these are small, public, safetensors-based
// checkpoints known to work well for a fast demo.
var fallbackCatalog = []CatalogEntry{
	{
		ID:          "roneneldan/TinyStories-1M",
		Label:       "TinyStories 1M",
		Description: "Tiny GPT-Neo style model trained on the TinyStories dataset. Ideal for fast demos.",
		ApproxSize:  "~4 MB",
		Family:      "GPT-Neo",
	},
	{
		ID:          "roneneldan/TinyStories-8M",
		Label:       "TinyStories 8M",
		Description: "Slightly larger TinyStories checkpoint, same family as the 1M variant.",
		ApproxSize:  "~30 MB",
		Family:      "GPT-Neo",
	},
	{
		ID:          "roneneldan/TinyStories-33M",
		Label:       "TinyStories 33M",
		Description: "Largest of the TinyStories family, still small enough for a quick download.",
		ApproxSize:  "~130 MB",
		Family:      "GPT-Neo",
	},
	{
		ID:          "EleutherAI/pythia-70m",
		Label:       "Pythia 70M",
		Description: "Smallest model in EleutherAI's Pythia suite, trained on The Pile.",
		ApproxSize:  "~165 MB",
		Family:      "GPT-NeoX",
	},
	{
		ID:          "EleutherAI/pythia-160m",
		Label:       "Pythia 160M",
		Description: "Next size up in the Pythia suite; same architecture as pythia-70m.",
		ApproxSize:  "~350 MB",
		Family:      "GPT-NeoX",
	},
	{
		ID:          "sshleifer/tiny-gpt2",
		Label:       "Tiny GPT-2",
		Description: "Randomly initialized, tiny GPT-2 configuration commonly used in tests.",
		ApproxSize:  "~2 MB",
		Family:      "GPT-2",
	},
	{
		ID:          "distilgpt2",
		Label:       "DistilGPT-2",
		Description: "Distilled version of GPT-2, ~2x faster, ~2/3 the parameters.",
		ApproxSize:  "~330 MB",
		Family:      "GPT-2",
	},
	{
		ID:          "gpt2",
		Label:       "GPT-2 (small, 124M)",
		Description: "The original OpenAI GPT-2 small checkpoint.",
		ApproxSize:  "~550 MB",
		Family:      "GPT-2",
	},
	{
		ID:          "hf-internal-testing/tiny-random-gpt2",
		Label:       "Tiny Random GPT-2 (internal-testing)",
		Description: "Randomly initialized tiny GPT-2 used across HF's own test suite.",
		ApproxSize:  "~2 MB",
		Family:      "GPT-2",
	},
	{
		ID:          "prajjwal1/bert-tiny",
		Label:       "BERT Tiny",
		Description: "2-layer, 128-hidden BERT model — great for a fast encoder-side demo.",
		ApproxSize:  "~17 MB",
		Family:      "BERT",
	},
}

// HFSibling is one file entry returned by the HuggingFace Hub API for a repo.
type HFSibling struct {
	RFilename string `json:"rfilename"`
}

// HFRepoInfo is the subset of the HF Hub API's model-info response we need.
type HFRepoInfo struct {
	ID        string      `json:"id"`
	Siblings  []HFSibling `json:"siblings"`
	Private   bool        `json:"private"`
	Disabled  bool        `json:"disabled"`
	Downloads int         `json:"downloads"`
}

const hfAPIBase = "https://huggingface.co/api/models/"

// FetchRepoInfo queries the public HuggingFace Hub API for the list of files
// in a repository. No auth token is required for public repos.
func FetchRepoInfo(repoID string) (*HFRepoInfo, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest(http.MethodGet, hfAPIBase+repoID, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "hf-mergekit-demo/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("contacting huggingface.co: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("model %q not found on HuggingFace Hub", repoID)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("huggingface API returned %d: %s", resp.StatusCode, string(body))
	}

	var info HFRepoInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decoding huggingface API response: %w", err)
	}
	return &info, nil
}

// ---------------------------------------------------------------------------
// Live catalog fetching
// ---------------------------------------------------------------------------

// hfListItem is the subset of fields we need from the HF Hub's model search
// API (GET /api/models). "gated" can be JSON `false`, `true`, or a string
// like "auto"/"manual" for gated repos, so it's captured as raw JSON and
// interpreted manually rather than typed as bool.
type hfListItem struct {
	ID          string          `json:"id"`
	Private     bool            `json:"private"`
	Disabled    bool            `json:"disabled"`
	Gated       json.RawMessage `json:"gated"`
	Downloads   int64           `json:"downloads"`
	Likes       int64           `json:"likes"`
	Tags        []string        `json:"tags"`
	PipelineTag string          `json:"pipeline_tag"`
	LibraryName string          `json:"library_name"`
	SafeTensors *struct {
		Total int64 `json:"total"`
	} `json:"safetensors"`
}

func (it hfListItem) isGated() bool {
	if len(it.Gated) == 0 {
		return false
	}
	s := strings.TrimSpace(string(it.Gated))
	return s != "" && s != "false" && s != "null"
}

func (it hfListItem) hasSafetensorsTag() bool {
	for _, t := range it.Tags {
		if strings.EqualFold(t, "safetensors") {
			return true
		}
	}
	return it.SafeTensors != nil && it.SafeTensors.Total > 0
}

const (
	hfListURL       = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=200&full=true"
	catalogMaxSize  = 60               // keep the dropdown from becoming unusably long
	catalogTTL      = 10 * time.Minute // how long a successful fetch is considered fresh
	catalogHTTPWait = 20 * time.Second
)

// FetchCatalog queries the public HuggingFace Hub search API for the
// current top-downloaded public models that publish safetensors weights,
// and maps them into CatalogEntry values. This intentionally does NOT
// download or inspect any weights - it's a metadata-only call - so it stays
// fast even though it's on the hot path for GET /api/models.
func FetchCatalog() ([]CatalogEntry, error) {
	client := &http.Client{Timeout: catalogHTTPWait}
	req, err := http.NewRequest(http.MethodGet, hfListURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "hf-mergekit-demo/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("contacting huggingface.co: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("huggingface model search API returned %d: %s", resp.StatusCode, string(body))
	}

	var items []hfListItem
	if err := json.NewDecoder(resp.Body).Decode(&items); err != nil {
		return nil, fmt.Errorf("decoding huggingface model search response: %w", err)
	}

	sort.SliceStable(items, func(i, j int) bool { return items[i].Downloads > items[j].Downloads })

	entries := make([]CatalogEntry, 0, catalogMaxSize)
	seen := map[string]bool{}
	for _, it := range items {
		if it.Private || it.Disabled || it.isGated() || it.ID == "" {
			continue
		}
		if !it.hasSafetensorsTag() {
			continue
		}
		if seen[it.ID] {
			continue
		}
		seen[it.ID] = true
		entries = append(entries, toCatalogEntry(it))
		if len(entries) >= catalogMaxSize {
			break
		}
	}

	if len(entries) == 0 {
		return nil, fmt.Errorf("huggingface model search returned no eligible safetensors models")
	}
	return entries, nil
}

func toCatalogEntry(it hfListItem) CatalogEntry {
	family := it.PipelineTag
	if family == "" {
		family = it.LibraryName
	}
	if family == "" {
		family = "unknown"
	}

	approxSize := "size unknown"
	if it.SafeTensors != nil && it.SafeTensors.Total > 0 {
		approxSize = "~" + humanCount(it.SafeTensors.Total) + " params"
	}

	desc := family
	if it.LibraryName != "" && it.LibraryName != family {
		desc += " · " + it.LibraryName
	}
	desc += fmt.Sprintf(" · %s downloads", humanCount(it.Downloads))

	return CatalogEntry{
		ID:          it.ID,
		Label:       prettifyRepoID(it.ID),
		Description: desc,
		ApproxSize:  approxSize,
		Family:      family,
	}
}

// prettifyRepoID turns "roneneldan/TinyStories-1M" into "TinyStories 1M
// (roneneldan)" for display, while leaving the underlying repo id (used for
// all API calls) untouched.
func prettifyRepoID(id string) string {
	parts := strings.SplitN(id, "/", 2)
	name := id
	org := ""
	if len(parts) == 2 {
		org, name = parts[0], parts[1]
	}
	name = strings.ReplaceAll(name, "_", " ")
	name = strings.ReplaceAll(name, "-", " ")
	if org != "" {
		return fmt.Sprintf("%s (%s)", name, org)
	}
	return name
}

func humanCount(n int64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1_000_000_000)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// ---------------------------------------------------------------------------
// Cached accessor
// ---------------------------------------------------------------------------

var catalogCache = struct {
	mu        sync.Mutex
	entries   []CatalogEntry
	fetchedAt time.Time
}{}

// GetCatalog returns the current catalog, refreshing it from the live
// HuggingFace Hub API at most once per catalogTTL. If a refresh fails, the
// last known-good cached result is returned (or fallbackCatalog if there
// isn't one yet), so a transient network hiccup never blanks the dropdown.
func GetCatalog() []CatalogEntry {
	catalogCache.mu.Lock()
	fresh := len(catalogCache.entries) > 0 && time.Since(catalogCache.fetchedAt) < catalogTTL
	cached := catalogCache.entries
	catalogCache.mu.Unlock()
	if fresh {
		return cached
	}

	entries, err := FetchCatalog()
	if err != nil {
		if len(cached) > 0 {
			return cached
		}
		return fallbackCatalog
	}

	catalogCache.mu.Lock()
	catalogCache.entries = entries
	catalogCache.fetchedAt = time.Now()
	catalogCache.mu.Unlock()
	return entries
}
