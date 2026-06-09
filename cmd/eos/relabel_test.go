package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosruntime "m31labs.dev/eos/runtime"
)

func writeRelabelFixture(t *testing.T, dir string) (string, string) {
	t.Helper()
	minedPath := filepath.Join(dir, "mined-scored.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(minedPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "alpha",
			Positive:      "labeled positive",
			Negatives:     []string{"false negative", "ambiguous doc"},
			TeacherScores: []float32{0.90, 0.97, 0.88},
		},
	}); err != nil {
		t.Fatalf("write mined fixture: %v", err)
	}
	poolPath := filepath.Join(dir, "random-scored.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(poolPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa:random",
			Query:         "alpha",
			Positive:      "labeled positive",
			Negatives:     []string{"random doc one", "random doc two"},
			TeacherScores: []float32{0.90, 0.31, 0.44},
		},
	}); err != nil {
		t.Fatalf("write pool fixture: %v", err)
	}
	return minedPath, poolPath
}

func TestRunRelabelTeacherNegativesHardNegativesMode(t *testing.T) {
	dir := t.TempDir()
	minedPath, poolPath := writeRelabelFixture(t, dir)
	outputPath := filepath.Join(dir, "relabeled.jsonl")

	output := captureRunOutput(t, []string{
		"relabel-teacher-negatives",
		"-promote-min", "0.95",
		"-negative-max", "0.80",
		"-negatives-per-row", "2",
		"-negatives-file", poolPath,
		minedPath,
		outputPath,
	})
	for _, want := range []string{"promoted=1", "output: " + outputPath} {
		if !strings.Contains(output, want) {
			t.Fatalf("output missing %q:\n%s", want, output)
		}
	}

	rows, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read relabeled: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want original + promoted", len(rows))
	}
	for _, row := range rows {
		if len(row.Negatives) != 2 || len(row.TeacherScores) != 3 {
			t.Fatalf("row = %+v", row)
		}
		for _, negative := range row.Negatives {
			if !strings.HasPrefix(negative, "random doc") {
				t.Fatalf("expected pool negatives, got %q", negative)
			}
		}
	}
	var promoted *eosruntime.EmbeddingTextHardNegativeExample
	for i := range rows {
		if rows[i].Positive == "false negative" {
			promoted = &rows[i]
		}
	}
	if promoted == nil || promoted.Source != "fiqa:promoted" {
		t.Fatalf("promoted row missing or untagged: %+v", rows)
	}

	manifestPayload, err := os.ReadFile(outputPath + ".relabel.manifest.json")
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest relabelTeacherNegativesManifest
	if err := json.Unmarshal(manifestPayload, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.Summary.Promoted != 1 || manifest.Summary.DroppedAmbiguous != 1 {
		t.Fatalf("manifest summary = %+v", manifest.Summary)
	}
}

func TestRunRelabelTeacherNegativesPairsMode(t *testing.T) {
	dir := t.TempDir()
	minedPath, _ := writeRelabelFixture(t, dir)
	outputPath := filepath.Join(dir, "relabeled-pairs.jsonl")

	captureRunOutput(t, []string{
		"relabel-teacher-negatives",
		"-emit", "pairs",
		minedPath,
		outputPath,
	})

	payload, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read pairs: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(payload)), "\n")
	if len(lines) != 2 {
		t.Fatalf("pair rows = %d, want 2", len(lines))
	}
	for _, line := range lines {
		var row struct {
			Source   string `json:"source"`
			Query    string `json:"query"`
			Positive string `json:"positive"`
			Negatives []string `json:"negatives"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("parse pair row: %v", err)
		}
		if row.Query == "" || row.Positive == "" || row.Source == "" {
			t.Fatalf("pair row incomplete: %s", line)
		}
		if len(row.Negatives) != 0 {
			t.Fatalf("pair row carries negatives: %s", line)
		}
	}
}

func TestRunRelabelTeacherNegativesDropsRowsWithoutNegatives(t *testing.T) {
	dir := t.TempDir()
	minedPath, _ := writeRelabelFixture(t, dir)
	outputPath := filepath.Join(dir, "relabeled-no-pool.jsonl")

	// Without a pool and with no surviving own negatives, hard-negative mode
	// must drop rows rather than write invalid zero-negative records.
	output, err := captureRunOutputAndError(t, []string{
		"relabel-teacher-negatives",
		"-promote-min", "0.95",
		"-negative-max", "0.50",
		minedPath,
		outputPath,
	})
	if err == nil {
		t.Fatalf("expected failure when no rows survive, output:\n%s", output)
	}
}

func TestRunSampleCorpusNegativesWritesRowsAndManifest(t *testing.T) {
	dir := t.TempDir()
	dataset := filepath.Join(dir, "beir")
	if err := os.MkdirAll(filepath.Join(dataset, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	var corpus strings.Builder
	corpus.WriteString(`{"_id":"d1","title":"T","text":"relevant"}` + "\n")
	for i := 2; i <= 12; i++ {
		corpus.WriteString(`{"_id":"d` + string(rune('a'+i)) + `","text":"filler"}` + "\n")
	}
	if err := os.WriteFile(filepath.Join(dataset, "corpus.jsonl"), []byte(corpus.String()), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dataset, "queries.jsonl"), []byte(`{"_id":"q1","text":"query one"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dataset, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath := filepath.Join(dir, "sampled.jsonl")

	output := captureRunOutput(t, []string{
		"sample-corpus-negatives",
		"-split", "train",
		"-per-query", "3",
		"-seed", "5",
		"-source", "demo:random",
		dataset,
		outputPath,
	})
	if !strings.Contains(output, "sampled_queries=1") {
		t.Fatalf("output:\n%s", output)
	}
	rows, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read sampled: %v", err)
	}
	if len(rows) != 1 || len(rows[0].Negatives) != 3 || rows[0].Source != "demo:random" {
		t.Fatalf("rows = %+v", rows)
	}
	if _, err := os.Stat(outputPath + ".sample.manifest.json"); err != nil {
		t.Fatalf("manifest: %v", err)
	}
}
