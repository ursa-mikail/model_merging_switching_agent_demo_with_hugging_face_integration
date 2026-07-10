// Command server runs the hf-mergekit-demo backend API.
package main

import (
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"hfmergekit/internal/api"
)

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		log.Printf("warning: invalid %s=%q (expected an integer), using default %d", key, v, fallback)
		return fallback
	}
	return n
}

func main() {
	port := getenv("PORT", "8080")
	dataDir := getenv("DATA_DIR", "/data")

	srv := api.NewServer(dataDir)
	// See CAVEAT.md: these caps are a basic, built-in mitigation against
	// resource-exhaustion abuse if this backend is ever reachable by
	// untrusted clients. Set to 0 to disable (unlimited).
	srv.MaxConcurrentDownloads = getenvInt("MAX_CONCURRENT_DOWNLOADS", srv.MaxConcurrentDownloads)
	srv.MaxConcurrentMerges = getenvInt("MAX_CONCURRENT_MERGES", srv.MaxConcurrentMerges)

	httpServer := &http.Server{
		Addr:         ":" + port,
		Handler:      srv.Routes(),
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 0, // downloads/merges + SSE can run long
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("hf-mergekit-demo backend listening on :%s (data dir: %s, max concurrent downloads: %d, max concurrent merges: %d)",
		port, dataDir, srv.MaxConcurrentDownloads, srv.MaxConcurrentMerges)
	if err := httpServer.ListenAndServe(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
