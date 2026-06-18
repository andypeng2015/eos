package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSparseEmbeddingSmokeConfigAdds32KPreflight(t *testing.T) {
	cfg, err := parseSparseEmbeddingSmokeConfig([]string{
		"-run-root", t.TempDir(),
		"-seq-len", "16",
		"-query-len", "2",
		"-dim", "4",
		"-top-k", "2",
		"-route-block-size", "4",
		"-route-top-blocks", "1",
		"-preflight-key-lens", "64,128",
	})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if !containsInt(cfg.PreflightKeyLens, 32768) {
		t.Fatalf("preflight key lens = %v, want 32768 included", cfg.PreflightKeyLens)
	}
	if cfg.ValueDim != cfg.Dim {
		t.Fatalf("value_dim = %d, want dim %d", cfg.ValueDim, cfg.Dim)
	}
}

func TestRunSmokeSparseEmbeddingEncoderWritesArtifacts(t *testing.T) {
	runRoot := t.TempDir()
	if err := runSmokeSparseEmbeddingEncoder([]string{
		"-run-root", runRoot,
		"-seq-len", "32",
		"-query-len", "2",
		"-dim", "8",
		"-top-k", "2",
		"-route-top-blocks", "2",
		"-preflight-key-lens", "4096,32768",
		"-max-score-fraction", "0.2",
	}); err != nil {
		t.Fatalf("run smoke: %v", err)
	}
	matches, err := filepath.Glob(filepath.Join(runRoot, "eos-sparse-embedding-encoder-smoke-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("run dirs = %v, want one", matches)
	}
	manifestPath := filepath.Join(matches[0], "manifest.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest sparseEmbeddingSmokeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Embedding.Dimension != 8 || len(manifest.Embedding.Vector) != 8 {
		t.Fatalf("embedding dimension/vector = %d/%d, want 8/8", manifest.Embedding.Dimension, len(manifest.Embedding.Vector))
	}
	if manifest.Runtime.OutputShape[0] != 2 || manifest.Runtime.OutputShape[1] != 8 {
		t.Fatalf("output shape = %v, want [2 8]", manifest.Runtime.OutputShape)
	}
	if !manifest.ThirtyTwoKPreflight.Present || !manifest.ThirtyTwoKPreflight.Passed {
		t.Fatalf("32k preflight = %+v, want present pass", manifest.ThirtyTwoKPreflight)
	}
	if !manifest.ThirtyTwoKPreflightOnly {
		t.Fatal("32k_preflight_only = false, want true for runtime seq_len 32")
	}
	if manifest.Runtime.AttentionMetadata["routing"] != "block_anchor" {
		t.Fatalf("routing metadata = %v, want block_anchor", manifest.Runtime.AttentionMetadata["routing"])
	}
	summary, err := os.ReadFile(filepath.Join(matches[0], "summary.tsv"))
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(summary)), "\n")
	if len(lines) != 2 {
		t.Fatalf("summary lines = %d, want 2", len(lines))
	}
	if got, want := len(strings.Split(lines[0], "\t")), len(strings.Split(lines[1], "\t")); got != want {
		t.Fatalf("summary columns header=%d row=%d", got, want)
	}
}
