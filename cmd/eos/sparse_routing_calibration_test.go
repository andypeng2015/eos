package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSparseRoutingCalibrationConfigDefaultsAndPaths(t *testing.T) {
	runRoot := t.TempDir()
	cfg, err := parseSparseRoutingCalibrationConfig([]string{
		"-run-root", runRoot,
		"-seq-len", "64",
		"-query-len", "2",
		"-dim", "8",
		"-top-k", "4",
		"-route-block-size", "8",
		"-route-top-blocks", "1,2,8",
	})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.ValueDim != cfg.Dim {
		t.Fatalf("value_dim = %d, want dim %d", cfg.ValueDim, cfg.Dim)
	}
	if cfg.RunDir == "" || !strings.HasPrefix(cfg.RunDir, runRoot) {
		t.Fatalf("run_dir = %q, want under %q", cfg.RunDir, runRoot)
	}
	if filepath.Base(cfg.JSONPath) != "calibration.json" || filepath.Base(cfg.TSVPath) != "calibration.tsv" {
		t.Fatalf("artifact paths json=%q tsv=%q", cfg.JSONPath, cfg.TSVPath)
	}
	if got, want := joinInts(cfg.RouteTopBlocks), "1,2,8"; got != want {
		t.Fatalf("route_top_blocks = %s, want %s", got, want)
	}
}

func TestRunCalibrateSparseRoutingWritesArtifacts(t *testing.T) {
	runRoot := t.TempDir()
	if err := runCalibrateSparseRouting([]string{
		"-run-root", runRoot,
		"-seq-len", "64",
		"-query-len", "2",
		"-dim", "8",
		"-top-k", "4",
		"-route-block-size", "8",
		"-route-top-blocks", "1,2,4,8",
		"-max-score-fraction", "1.2",
		"-min-exact-topk-recall", "1",
		"-min-output-cosine", "0.999",
		"-require-pass",
	}); err != nil {
		t.Fatalf("run calibration: %v", err)
	}
	matches, err := filepath.Glob(filepath.Join(runRoot, "eos-sparse-routing-calibration-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("run dirs = %v, want one", matches)
	}
	jsonPath := filepath.Join(matches[0], "calibration.json")
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read calibration json: %v", err)
	}
	var report sparseRoutingCalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("decode calibration json: %v", err)
	}
	if report.Schema != "manta.sparse_routing_calibration.v1" {
		t.Fatalf("schema = %q", report.Schema)
	}
	if len(report.Rows) != 4 || report.Summary.Rows != 4 {
		t.Fatalf("rows len=%d summary=%d, want 4", len(report.Rows), report.Summary.Rows)
	}
	if report.Summary.PassingRows == 0 || report.Summary.BestPassing == nil {
		t.Fatalf("summary = %+v, want at least one passing row", report.Summary)
	}
	best := report.Summary.BestPassing
	if best.RouteTopBlocks != 1 || best.ExactTopKRecallAvg != 1 || best.OutputCosine < 0.999 {
		t.Fatalf("best passing = %+v, want lowest-budget exact-match row", best)
	}
	tsv, err := os.ReadFile(filepath.Join(matches[0], "calibration.tsv"))
	if err != nil {
		t.Fatalf("read calibration tsv: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(tsv)), "\n")
	if len(lines) != 5 {
		t.Fatalf("tsv lines = %d, want header plus 4 rows", len(lines))
	}
	headerCols := len(strings.Split(lines[0], "\t"))
	for i, line := range lines[1:] {
		if got := len(strings.Split(line, "\t")); got != headerCols {
			t.Fatalf("tsv row %d columns = %d, want %d", i+1, got, headerCols)
		}
	}
}

func TestSparseRoutingCalibrationRequirePassFailsOnlyWhenRequested(t *testing.T) {
	common := []string{
		"-run-root", t.TempDir(),
		"-seq-len", "64",
		"-query-len", "2",
		"-dim", "8",
		"-top-k", "4",
		"-route-block-size", "8",
		"-route-top-blocks", "1",
		"-max-score-fraction", "0.01",
		"-min-exact-topk-recall", "1",
		"-min-output-cosine", "1",
	}
	if err := runCalibrateSparseRouting(common); err != nil {
		t.Fatalf("calibration without require-pass returned error: %v", err)
	}
	withRequire := append(append([]string{}, common...), "-require-pass", "-run-root", t.TempDir())
	if err := runCalibrateSparseRouting(withRequire); err == nil {
		t.Fatal("calibration with require-pass succeeded despite impossible thresholds")
	}
}
