// Command server runs the hf-mergekit-demo backend API.
package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"hfmergekit/internal/api"
)

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	port := getenv("PORT", "8080")
	dataDir := getenv("DATA_DIR", "/data")

	srv := api.NewServer(dataDir)

	httpServer := &http.Server{
		Addr:         ":" + port,
		Handler:      srv.Routes(),
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 0, // downloads/merges + SSE can run long
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("hf-mergekit-demo backend listening on :%s (data dir: %s)", port, dataDir)
	if err := httpServer.ListenAndServe(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
