// Package download implements a minimal HuggingFace Hub downloader. It fetches
// the repo's file listing via the public Hub API, then streams the files we
// care about (safetensors weight shards, their index, and config.json) to a
// local directory, reporting progress through a callback.
package download

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"hfmergekit/internal/models"
)

// ProgressFunc is called with human-readable log lines as the download proceeds.
type ProgressFunc func(line string)

// Result summarizes a completed download.
type Result struct {
	RepoID      string   `json:"repoId"`
	Dir         string   `json:"dir"`
	Files       []string `json:"files"`
	TotalBytes  int64    `json:"totalBytes"`
	SafeTensors []string `json:"safetensors"`
}

const resolveBase = "https://huggingface.co/"

// wantFile decides whether a given repo file should be downloaded for the
// purposes of this demo: model weights (safetensors), the shard index, and
// the model config.
func wantFile(name string) bool {
	base := filepath.Base(name)
	switch {
	case strings.HasSuffix(base, ".safetensors"):
		return true
	case base == "model.safetensors.index.json":
		return true
	case base == "config.json":
		return true
	default:
		return false
	}
}

// Download fetches the relevant files for repoID into destDir (created if
// needed). It returns a summary of what was written to disk.
func Download(repoID, destDir string, progress ProgressFunc) (*Result, error) {
	if progress == nil {
		progress = func(string) {}
	}

	progress(fmt.Sprintf("Looking up %s on the HuggingFace Hub API...", repoID))
	info, err := models.FetchRepoInfo(repoID)
	if err != nil {
		return nil, err
	}
	if info.Private || info.Disabled {
		return nil, fmt.Errorf("model %q is private or disabled and cannot be downloaded anonymously", repoID)
	}

	var toFetch []string
	for _, s := range info.Siblings {
		if wantFile(s.RFilename) {
			toFetch = append(toFetch, s.RFilename)
		}
	}
	if len(toFetch) == 0 {
		return nil, fmt.Errorf("no .safetensors weight files were found in %q; pick a model that publishes safetensors weights", repoID)
	}

	hasIndex := false
	for _, f := range toFetch {
		if filepath.Base(f) == "model.safetensors.index.json" {
			hasIndex = true
		}
	}
	if !hasIndex {
		count := 0
		for _, f := range toFetch {
			if strings.HasSuffix(f, ".safetensors") {
				count++
			}
		}
		if count > 1 {
			progress("Warning: multiple safetensors shards found without an index file; will attempt to load all of them.")
		}
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, fmt.Errorf("creating destination directory: %w", err)
	}

	client := &http.Client{Timeout: 20 * time.Minute}
	res := &Result{RepoID: repoID, Dir: destDir}

	for _, fname := range toFetch {
		url := resolveBase + repoID + "/resolve/main/" + fname
		progress(fmt.Sprintf("Downloading %s ...", fname))

		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", "hf-mergekit-demo/1.0")

		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("downloading %s: %w", fname, err)
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, fmt.Errorf("downloading %s: server returned HTTP %d", fname, resp.StatusCode)
		}

		outPath := filepath.Join(destDir, filepath.Base(fname))
		outFile, err := os.Create(outPath)
		if err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("creating %s: %w", outPath, err)
		}

		n, err := io.Copy(outFile, resp.Body)
		outFile.Close()
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("writing %s: %w", outPath, err)
		}

		progress(fmt.Sprintf("  -> saved %s (%s)", filepath.Base(fname), humanBytes(n)))
		res.Files = append(res.Files, filepath.Base(fname))
		res.TotalBytes += n
		if strings.HasSuffix(fname, ".safetensors") {
			res.SafeTensors = append(res.SafeTensors, outPath)
		}
	}

	progress(fmt.Sprintf("Download complete: %d file(s), %s total.", len(res.Files), humanBytes(res.TotalBytes)))
	return res, nil
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
